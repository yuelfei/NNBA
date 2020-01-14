'''
@文件名称: boundary_bert_emb.py
@作者: 武乐飞
@创建时间: 2019/1/19 - 20:29
@描述:
'''



import shutil
from new_yuefei_tf import *
from yuefei_utils import *
from Flags_define import *
from utils import print_config, save_config, load_config, test_ner,make_path, clean, create_model, save_model
from data_utils import load_word2vec,init_emb_weights
import itertools
import pickle
from keras.callbacks import ModelCheckpoint

from keras import optimizers
from keras.layers import Input, Bidirectional,merge,TimeDistributed,Dense,Dropout,Conv1D,Flatten,Concatenate
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Model
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train():

    logger=get_logger("./log/train_bert_keras_batch_size.txt ")
    collected_num = -1

    logger.info("\n\n\n全匹配方式生成三个数据集，开始训练\n\n")

    logger.info("学习率:{0}".format(FLAGS.lr))
    logger.info("batch_size的数值是:{0}\n".format(FLAGS.batch_size))

    #训练集和验证集由ACE原始文件得出，由所有的正例和，随意匹配的负例组成
    logger.info("生成训练集.........")
    train_bert_emb_array=pickle.load(open("./bert_sentence_array/train_bert_array","rb"))
    train_left_list, train_entity_left, train_entity_right, train_right_list, train_label = E2E_main_train_dev_bert_emb(train_bert_emb_array[:collected_num], "./datas_crf_nn/mul_class/BIO_train", "./datas_crf_nn/mul_class/B_train","./datas_crf_nn/mul_class/E_train","训练",logger)

    logger.info("生成验证集.........")
    dev_bert_emb_array = pickle.load(open("./bert_sentence_array/dev_bert_array", "rb"))
    dev_left_list, dev_entity_left, dev_entity_right, dev_right_list, dev_label = E2E_main_train_dev_bert_emb(dev_bert_emb_array[:collected_num],"./datas_crf_nn/mul_class/BIO_dev","./datas_crf_nn/mul_class/B_dev","./datas_crf_nn/mul_class/E_dev","验证",logger)

    #测试集由bert输出文件得出，采用B+B匹配策略
    logger.info("生成测试集.........")
    test_bert_emb_array = pickle.load(open("./bert_sentence_array/test_bert_array", "rb"))
    test_left_list, test_entity_left, test_entity_right, test_right_list,test_label= C_method_main_test_bert_emb(test_bert_emb_array[:collected_num], "./datas_crf_nn/bert_start_end/start_test", "./datas_crf_nn/bert_start_end/end_test","./datas_crf_nn/mul_class/BIO_test", "./datas_crf_nn/mul_class/B_test", "./datas_crf_nn/mul_class/E_test",logger)

    # new_train_label = oneTo_eight(train_label)
    # new_dev_label = oneTo_eight(dev_label)
    # new_test_label = oneTo_eight(test_label)

    new_train_label, pos_count_train = eightTo_two_array(train_label)  # 二分类
    new_dev_label, pos_count_dev = eightTo_two_array(dev_label)
    new_test_label, pos_count_test = eightTo_two_array(test_label)
    part_left_input = Input(shape=(FLAGS.seq_langth,FLAGS.emb_dim,), name="left_part")
    lstm_left = LSTM(100, name="lstm_left")(part_left_input)

    part_entity_left_input = Input(shape=(FLAGS.seq_langth,FLAGS.emb_dim,), name="entity__left_part")
    lstm_left_entity = LSTM(100, name="left_entity")(part_entity_left_input)

    part_entity_right_input = Input(shape=(FLAGS.seq_langth,FLAGS.emb_dim,), name="entity_right_part")
    lstm_right_entity = LSTM(100, name="right_entity")(part_entity_right_input)

    part_right_input = Input(shape=(FLAGS.seq_langth,FLAGS.emb_dim,), name="right_part")
    lstm_right = LSTM(100, name="lstm_right")(part_right_input)

    x = Concatenate(axis=-1)([lstm_left, lstm_left_entity, lstm_right_entity, lstm_right])
    x = Dense(100)(x)
    x = Dense(20)(x)
    x_out = Dense(2,activation='relu',name="dense_ss")(x)

    model=Model(inputs=[part_left_input,part_entity_left_input,part_entity_right_input,part_right_input],outputs=x_out)
    model.summary()

    optimiaer_1=optimizers.RMSprop(lr=0.001, rho=FLAGS.rms_rho, epsilon=None, decay=0.0)
    model.compile(optimizer=optimiaer_1, loss='binary_crossentropy', metrics=['accuracy'])

    model_path = './keras_model/best_model_bert_emb_binary_batch_size.hdf5'
    if (os.path.exists(model_path)):
        os.remove(model_path)
        logger.info("删除已存在的模型文件.......")

    check_pointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True, monitor='val_acc')
    hist = model.fit([train_left_list, train_entity_left, train_entity_right, train_right_list], new_train_label, batch_size=FLAGS.batch_size, epochs=10, verbose=2,validation_data=[[dev_left_list, dev_entity_left, dev_entity_right, dev_right_list], new_dev_label],callbacks=[check_pointer])

    model.load_weights(model_path)

    print("\n测试集..............")
    y_pre = model.predict([test_left_list, test_entity_left, test_entity_right, test_right_list])
    performance_on_positives_bert_logger(new_test_label.argmax(-1), y_pre.argmax(-1),logger)#二分类性能

    print("\n"+"验证集。。。。。。。")
    y_pre = model.predict([dev_left_list, dev_entity_left, dev_entity_right, dev_right_list])
    performance_on_positives_bert_logger(new_dev_label.argmax(-1), y_pre.argmax(-1),logger)#二分类性能

    # print("--------------------多分类------------------------")
    # y_pre = model.predict([test_left_list, test_entity_left, test_entity_right, test_right_list])
    # performance_on_positives_boundry_mul_class(new_test_label.argmax(-1), y_pre.argmax(-1))


if __name__ == '__main__':
    train()












