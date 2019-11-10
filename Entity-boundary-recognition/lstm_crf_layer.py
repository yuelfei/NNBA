# encoding=utf-8

"""
bert-blstm-crf layer
@Author:
"""

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf

class BLSTM_CRF(object):
    def __init__(self, embedded_chars, hidden_unit, cell_type,num_layers, droupout_rate,
                 initializers,num_labels, seq_length, labels, lengths, is_training):

        self.hidden_unit = hidden_unit
        self.droupout_rate = droupout_rate
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.embedded_chars = embedded_chars
        self.initializers = initializers
        self.seq_length = seq_length
        self.num_labels = num_labels
        self.labels = labels
        self.lengths = lengths

        self.is_training = is_training

    def add_blstm_crf_layer(self):


        # lstm input dropout rate set 0.5 will get best score

        # embedded_chars_yuefei=self.embedded_chars

        if self.is_training:
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.droupout_rate,name="emb_2")
        #blstm
        lstm_output = self.blstm_layer(self.embedded_chars)
        #project
        logits = self.project_bilstm_layer(lstm_output)
        #crf
        loss, trans = self.crf_layer(logits)
        # CRF decode, pred_ids
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)
        return ((loss, logits, trans, pred_ids))

    def get_seq_embedding(self):
        """
        blstm-crf网络
        :return:
        """
        return self.embedded_chars

    def _witch_cell(self):
        """
        RNN 类型
        :return: 
        """
        cell_tmp = None
        if self.cell_type == 'lstm':
            cell_tmp = rnn.BasicLSTMCell(self.hidden_unit)
        elif self.cell_type == 'gru':
            cell_tmp = rnn.GRUCell(self.hidden_unit)
        # 是否需要进行dropout
        if self.droupout_rate is not None:
            cell_tmp = rnn.DropoutWrapper(cell_tmp, output_keep_prob=self.droupout_rate)
        return cell_tmp

    def _bi_dir_rnn(self):
        """
        双向RNN
        :return:
        """
        cell_fw = self._witch_cell()
        cell_bw = self._witch_cell()
        if self.droupout_rate is not None:
            cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.droupout_rate)
            cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.droupout_rate)
        return cell_fw, cell_bw

    def blstm_layer(self, embedding_chars):

        with tf.variable_scope('rnn_layer'):
            cell_fw, cell_bw = self._bi_dir_rnn()
            if self.num_layers > 1:
                cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
                cell_bw = rnn.MultiRNNCell([cell_bw] * self.num_layers, state_is_tuple=True)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedding_chars,dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)
        return outputs

    def project_bilstm_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.hidden_unit * 2, self.hidden_unit],dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.hidden_unit], dtype=tf.float32,initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.hidden_unit * 2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.hidden_unit, self.num_labels],dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    def crf_layer(self, logits):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable("transitions",shape=[self.num_labels, self.num_labels],initializer=self.initializers.xavier_initializer())
            log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(inputs=logits,tag_indices=self.labels,transition_params=trans,sequence_lengths=self.lengths)
            return tf.reduce_mean(-log_likelihood), trans