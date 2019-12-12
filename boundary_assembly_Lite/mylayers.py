import os
import re
import codecs

from data_utils import create_dico, create_mapping, zero_digits
from data_utils import iob2, iob_iobes, get_seg_features
from numpy import *
from keras import backend as Kb
import sys 
from keras.engine import Layer
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from keras.engine import InputSpec

from keras_contrib import backend as K
from keras_contrib import activations
from keras_contrib import initializers
from keras_contrib import regularizers
from keras_contrib import constraints
from tensorflow.contrib.layers.python.layers import initializers as Init
from data_utils import load_word2vec, my_load_word2vec
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.init_ops import *
from head import *
from tensorflow.contrib.crf import viterbi_decode
from keras.objectives import categorical_crossentropy
from keras.objectives import binary_crossentropy
from keras.objectives import sparse_categorical_crossentropy


def init_emb_weights(shape, id_to_char):
    #emb_shape = (len(id_to_char), 100)
    initializer = Init.xavier_initializer()
    emb_weights = Kb.variable(initializer(shape),
                              dtype=None,
                              name="char_embedding")
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        emb_weights = sess.run(emb_weights.read_value())
        emb_weights = load_word2vec(FLAGS.emb_file, id_to_char, FLAGS.char_dim, emb_weights)
    return emb_weights


class MyInitializer(Initializer):
    """
    Initializer that generates tensors initialized to 0.
    """
    def __init__(self, id_to_char, dtype=dtypes.float32):
        self.dtype = dtypes.as_dtype(dtype)
        self.id_to_char = id_to_char
    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        emb_weights = my_load_word2vec(shape, FLAGS.emb_file, self.id_to_char, FLAGS.char_dim)
        return tf.constant(emb_weights)

    def get_config(self):
        return {"dtype": self.dtype.name}

class Embeding_Layer(Layer):
    def __init__(self, id_to_char, **kwargs):
        self.id_to_char = id_to_char
        #self.output_dim = x.shape[0]
        self.char_dim = FLAGS.char_dim
        self.initializer = MyInitializer(id_to_char)
        super(Embeding_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[-1]
        self.input_length = input_shape[-2]
        self.char_lookup = self.add_weight(
            name="char_embedding",
            shape=[len(self.id_to_char), self.char_dim],
            initializer=self.initializer,
            trainable=False)

        self.built = True

    def call(self, x, mask=None):
        #(None, 20, 64)
        #embedding = []
        #embedding.append(tf.nn.embedding_lookup(self.char_lookup, x))
        #embed = np.array(embedding).reshape(-1, 20, self.char_dim)
        input_embedding = tf.nn.embedding_lookup(self.char_lookup, x)
        input_embedding = tf.reshape(input_embedding, shape=[-1, self.input_length, self.char_dim])
        return input_embedding

    def compute_output_shape(self, input_shape):
        return (None, self.input_length, self.char_dim)




class TimePRO(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(TimePRO, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[-1]

        self.phidden = self.add_weight(name='phidden',
                                      shape=[input_shape[-1], int(input_shape[-1]/2)],
                                      initializer='uniform',
                                      trainable=True)
        self.pb = self.add_weight(name='pb',
                                       shape=[int(input_shape[-1]/2)],
                                       initializer='uniform',
                                       trainable=True)
        self.logits = self.add_weight(name='logits',
                                       shape=[int(input_shape[-1]/2), self.output_dim],
                                       initializer='uniform',
                                       trainable=True)
        self.lb = self.add_weight(name='lb',
                                      shape=[self.output_dim],
                                      initializer='uniform',
                                      trainable=True)
        self.built = True

    def call(self, x, mask=None):

        #(None, 20, 64)
        output = tf.reshape(x, shape=[-1, self.input_spec[0].shape[-1]])

        hidden = tf.tanh(tf.nn.xw_plus_b(output, self.phidden, self.pb))

        pred = tf.nn.xw_plus_b(hidden, self.logits, self.lb)

        return tf.reshape(pred, [-1, self.input_spec[0].shape[1], self.output_dim])

    def compute_output_shape(self, input_shape):
        return (None, input_shape[1], self.output_dim)


class MyCRF(Layer):
    def __init__(self, units,
                 learn_mode='join',
                 test_mode=None,
                 sparse_target=False,
                 use_boundary=True,
                 use_bias=True,
                 activation='linear',
                 kernel_initializer='glorot_uniform',
                 chain_initializer='orthogonal',
                 bias_initializer='zeros',
                 boundary_initializer='zeros',
                 kernel_regularizer=None,
                 chain_regularizer=None,
                 boundary_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 chain_constraint=None,
                 boundary_constraint=None,
                 bias_constraint=None,
                 input_dim=None,
                 unroll=False,
                 **kwargs):
        super(MyCRF, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.num_tags = units
        self.learn_mode = learn_mode
        assert self.learn_mode in ['join', 'marginal']
        self.test_mode = test_mode
        if self.test_mode is None:
            self.test_mode = 'viterbi' if self.learn_mode == 'join' else 'marginal'
        else:
            assert self.test_mode in ['viterbi', 'marginal']
        self.sparse_target = sparse_target
        self.use_boundary = use_boundary
        self.use_bias = use_bias

        self.activation = activations.get(activation)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.chain_initializer = initializers.get(chain_initializer)
        self.boundary_initializer = initializers.get(boundary_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.chain_regularizer = regularizers.get(chain_regularizer)
        self.boundary_regularizer = regularizers.get(boundary_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.chain_constraint = constraints.get(chain_constraint)
        self.boundary_constraint = constraints.get(boundary_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.unroll = unroll

    def build(self, input_shape):

        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[-1]
        self.batch_size = input_shape[0]
        self.num_steps = input_shape[1]

        self.kernel = self.add_weight((self.input_dim, self.units),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.chain_kernel = self.add_weight((self.units, self.units),
                                            name='chain_kernel',
                                            initializer=self.chain_initializer,
                                            regularizer=self.chain_regularizer,
                                            constraint=self.chain_constraint)
        if self.use_bias:
            self.bias = self.add_weight((self.units,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        if self.use_boundary:
            self.left_boundary = self.add_weight((self.units,),
                                                 name='left_boundary',
                                                 initializer=self.boundary_initializer,
                                                 regularizer=self.boundary_regularizer,
                                                 constraint=self.boundary_constraint)
            self.right_boundary = self.add_weight((self.units,),
                                                  name='right_boundary',
                                                  initializer=self.boundary_initializer,
                                                  regularizer=self.boundary_regularizer,
                                                  constraint=self.boundary_constraint)
        self.built = True

    def call(self, x, mask=None):
        # (None, 50, 3)
        #lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
        """
                :param logits: [batch_size, num_steps, num_tags]float32, logits
                :param lengths: [batch_size]int32, real length of each sequence
                :param matrix: transaction matrix for inference
                :return:
                """
        # inference final labels usa viterbi Algorithm
        print(mask)
        print(x)
        used = tf.sign(tf.abs(K.cast(mask, K.floatx())))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)


        paths = []
        small = -1000.0
        start = np.asarray([[small] * self.units + [0]])
        #for score, length in zip(x, self.lengths):
        print(self.lengths)
        for i in range(20):
            score = x[i]
            length = self.lengths[i]
            score = score[:length]
            print(length)
            pad = small * K.ones((30, 1))
            logits = K.concatenate([score, pad], axis=1)
            logits = K.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, self.kernel)

            paths.append(path[1:])
        return paths

    @property
    def loss_function(self):
        if self.learn_mode == 'join':
            def loss(y_true, y_pred):
                assert self.inbound_nodes, 'CRF has not connected to any layer.'
                assert not self.outbound_nodes, 'When learn_model="join", CRF must be the last layer.'
                small = -1000.0
                # pad logits for crf loss
                project_logits = self.inbound_nodes[0].input_tensors[0]
                start_logits = tf.concat(
                    [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]),
                     tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
                pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)

                logits = tf.concat([project_logits, pad_logits], axis=-1)
                logits = tf.concat([start_logits, logits], axis=1)
                targets = tf.concat(
                    [tf.cast(self.num_tags * tf.ones([self.batch_size, 1]), tf.int32), self.y_pred], axis=-1)


                log_likelihood, _ = crf_log_likelihood(
                    inputs=logits,
                    tag_indices=targets,
                    transition_params=self.trans,
                    sequence_lengths=self.lengths + 1)

                return tf.reduce_mean(-log_likelihood)

            return loss
        else:
            if self.sparse_target:
                return sparse_categorical_crossentropy
            else:
                return categorical_crossentropy



    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"  if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)

            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)


    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (self.units,)

    def compute_mask(self, input, mask=None):
        if mask is not None and self.learn_mode == 'join':
            return K.any(mask, axis=1)
        return mask
