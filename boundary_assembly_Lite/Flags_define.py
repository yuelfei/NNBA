'''
@文件名称: Flags_define.py
@作者: 武乐飞
@创建时间: 2018/12/17 - 20:47
@描述: 
'''
import tensorflow as tf
import os
from collections import OrderedDict

flags = tf.app.flags
flags.DEFINE_boolean("clean",       True,      "clean train folder")
flags.DEFINE_boolean("train",       True,      "Wither train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim",     0,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM")
flags.DEFINE_string("tag_schema",   "iobes",    "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_integer("batch_size",    256,         "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "rmsprop",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       True,       "Wither lower case")

flags.DEFINE_integer("max_epoch",   10,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 500,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "./log/train.log",    "File for log")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "result",       "Path for results")
flags.DEFINE_string("emb_file",     "wiki_100.utf8", "Path for pre_trained embedding")
flags.DEFINE_string("train_file",   os.path.join("data", "START_Training_train"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join("data", "START_Training_dev"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join("data", "START_Testing_0.txt"),   "Path for test data")

flags.DEFINE_float("rms_rho",            0.9,      "Initial learning rate")

flags.DEFINE_string("pad_emb",    os.path.join("bert_sentence_array", "pad_emb"),   "Path for pad_emb")

flags.DEFINE_integer("seq_langth",    80,         "seq_langth")
flags.DEFINE_integer("emb_dim",    768,         "embedding dim")
flags.DEFINE_integer("reshape_dim",    1,         "reshape_dim")
flags.DEFINE_integer("start_N",   2,         "sseq_langth")
flags.DEFINE_integer("end_N",    2,         "sseq_langth")

flags.DEFINE_integer("entity_length",    12,         "max_entity_length")

flags.DEFINE_float("start_a_threshold",    0.99,   "start_threshold")
flags.DEFINE_float("end_a_threshold",    0.99,   "end_threshold")

flags.DEFINE_integer("define_N",    1,         "seq_langth")
flags.DEFINE_integer("define_fb_N",    1,         "ssseq_langth")

flags.DEFINE_float("correct_false_a",    0.5,         "ssseq_langth")

flags.DEFINE_integer("read_max_seq_langth",    80,         "Maximum length read from a file")

FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad","rmsprop"]


def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size
    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config