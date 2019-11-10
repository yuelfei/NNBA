#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bert_base import *

import tf_metrics
import pickle
import numpy as np

from Flag_config import *
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def is_train():
    if FLAGS.clean and FLAGS.do_train:
        if os.path.exists(FLAGS.output_dir):
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)
            try:
                del_file(FLAGS.output_dir)
            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and data.conf')
                exit(-1)
        if os.path.exists(FLAGS.data_config_path):
            try:
                os.remove(FLAGS.data_config_path)
            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and data.conf')
                exit(-1)

def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {"ner": NerProcessor}
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError("Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    is_train()

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    # is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    is_per_host=3
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=FLAGS.iterations_per_loop,num_shards=FLAGS.num_tpu_cores,per_host_input_for_training=is_per_host))
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if os.path.exists(FLAGS.data_config_path):
        with codecs.open(FLAGS.data_config_path) as fd:
            data_config = json.load(fd)
    else:
        data_config = {}

    if FLAGS.do_train:
        # 加载训练数据
        if len(data_config) == 0:
            train_examples = processor.get_train_examples(FLAGS.data_dir)
            num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
            num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

            data_config['num_train_steps'] = num_train_steps
            data_config['num_warmup_steps'] = num_warmup_steps
            data_config['num_train_size'] = len(train_examples)
        else:
            num_train_steps = int(data_config['num_train_steps'])
            num_warmup_steps = int(data_config['num_warmup_steps'])
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        if data_config.get('train.tf_record_path', '') == '':
            train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
            filed_based_convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        else:
            train_file = data_config.get('train.tf_record_path')
        num_train_size = num_train_size = int(data_config['num_train_size'])
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", num_train_size)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        train_input_fn = file_based_input_fn_builder(input_file=train_file,seq_length=FLAGS.max_seq_length,is_training=True,drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        if data_config.get('eval.tf_record_path', '') == '':
            eval_examples = processor.get_dev_examples(FLAGS.data_dir)
            eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
            filed_based_convert_examples_to_features(eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)#将数据写入tf_record
            data_config['eval.tf_record_path'] = eval_file
            data_config['num_eval_size'] = len(eval_examples)
        else:
            eval_file = data_config['eval.tf_record_path']
        num_eval_size = data_config.get('num_eval_size', 0)
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", num_eval_size)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_steps = None
        if FLAGS.use_tpu:
            eval_steps = int(num_eval_size / FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(input_file=eval_file,seq_length=FLAGS.max_seq_length,is_training=False,drop_remainder=eval_drop_remainder)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with codecs.open(output_eval_file, "w", encoding='utf-8') as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if not os.path.exists(FLAGS.data_config_path):
        with codecs.open(FLAGS.data_config_path, 'a', encoding='utf-8') as fd:
            json.dump(data_config, fd)

    if FLAGS.do_predict:
        token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
        if os.path.exists(token_path):
            os.remove(token_path)

        with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        filed_based_convert_examples_to_features(predict_examples, label_list,FLAGS.max_seq_length, tokenizer,predict_file, mode="test")

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        if FLAGS.use_tpu:
            raise ValueError("Prediction in TPU not supported")
        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(input_file=predict_file,seq_length=FLAGS.max_seq_length,is_training=False,drop_remainder=predict_drop_remainder)


        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")

        def result_to_pair(writer):
            for predict_line, prediction in zip(predict_examples, result):
                print(prediction)
                idx = 0
                line = ''
                line_token = str(predict_line.text).split(' ')
                label_token = str(predict_line.label).split(' ')
                if len(line_token) != len(label_token):
                    tf.logging.info(predict_line.text)
                    tf.logging.info(predict_line.label)
                for id in prediction:
                    if id == 0:
                        continue
                    curr_labels = id2label[id]
                    if curr_labels in ['[CLS]', '[SEP]']:
                        continue
                    try:
                        line += line_token[idx] + ' ' + label_token[idx] + ' ' + curr_labels + '\n'
                    except Exception as e:
                        tf.logging.info(e)
                        tf.logging.info(predict_line.text)
                        tf.logging.info(predict_line.label)
                        line = ''
                        break
                    idx += 1
                writer.write(line + '\n')

        with codecs.open(output_predict_file, 'w', encoding='utf-8') as writer:
            result_to_pair(writer)
        from conlleval import return_report
        eval_result = return_report(output_predict_file)
        print(eval_result)


if __name__ == "__main__":
    main()