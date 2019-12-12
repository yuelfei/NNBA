import math
import random
import tensorflow as tf
import numpy as np

#读文本 ，获取嵌套标签
def nested_read(path,collection):

    file=open(path,"r",encoding='utf-8')
    file_line=file.readlines()

    word=[]
    nested_label=[]

    word_temp=[]
    nested_label_temp=[]

    assert collection>=-1 ,"collection的值为-1或者大于等于零的整数"

    if(collection==-1):#-1时读取全部数据
        line_temp=file_line[0:]
    else:#不为-1时，按行读取
        line_temp=file_line[:collection]

    for node in line_temp:
        node_list=str(node).strip().split("\t")
        if(node_list[0]==""):
            word.append(word_temp)
            nested_label.append(nested_label_temp)
            word_temp=[]
            nested_label_temp=[]
        else:
            word_temp.append(node_list[0])
            nested_label_temp.append(node_list[-1])
    return word,nested_label

#从嵌套标签中获取实体正例index和type
def match_entity(B_label,E_label):
    true_entity_type=[]
    true_entity_index=[]
    for B_label_node,E_label_node in zip(B_label,E_label):
        B_node_temp = []
        B_node_index=[]
        for B_sub,B_sub_index in zip(B_label_node,range(len(B_label_node))):
            if('B' in B_sub):
                B_node_temp.append(B_sub)
                B_node_index.append(B_sub_index)

        E_node_temp = []
        E_node_index = []
        for E_sub, E_sub_index in zip(E_label_node, range(len(E_label_node))):
            if ('B' in E_sub):
                E_node_temp.append(E_sub)
                E_node_index.append(E_sub_index)
        if(len(E_node_temp)==0):
            true_entity_type.append([])
            true_entity_index.append([])
            continue
        match_index=[]
        match_type=[]
        for E_match,E_match_index in zip(E_node_temp,E_node_index):
            corrent_list=str(E_match).strip().replace("B-","").split(",")
            for B_match,B_match_index in zip(B_node_temp,B_node_index):
                if(corrent_list[0] in B_match):
                    match_index.append([B_match_index,E_match_index])
                    match_type.append(corrent_list[-1])

        true_entity_type.append(match_type)
        true_entity_index.append(match_index)
        match_index = []
        match_type = []
    return true_entity_type,true_entity_index

#生成所有情况的匹配word和对应的index
def all_index_match(word):

    total_match_word=[]
    total_sentence_order=[]
    for word_list in word:

        match_word = []
        single_word_temp=[]
        start_temp = []
        end_temp = []
        sentence_order = []

        for a_node in range(len(word_list)):
            for b_node in range(len(word_list)):
                if (a_node < b_node):
                    sentence_order.append([a_node, b_node])

        for index_node in sentence_order:
            start_index = index_node[0]
            end_index = index_node[-1]
            single_word_temp.append(word_list[:start_index+1])
            single_word_temp.append(word_list[start_index:])
            single_word_temp.append(word_list[:end_index+1])
            single_word_temp.append(word_list[end_index:])
            match_word.append(single_word_temp)
            single_word_temp=[]
        total_match_word.append(match_word)
        match_word=[]
        total_sentence_order.append(sentence_order)
        sentence_order=[]

    return total_match_word,total_sentence_order

#获得任意情况的字符序列和对应的type
def get_word_type(total_match_word,total_sentence_order,true_entity_index,true_entity_type):
    result_word_type=[]
    for total_word_node,total_sentence_node,true_index_node,true_type_node in zip(total_match_word,total_sentence_order,true_entity_index,true_entity_type):
        temp_corrent_word=[]
        for corrent_word_node,corrent_sentence in zip(total_word_node,total_sentence_node):
            if(corrent_sentence in true_index_node):
                temp_corrent_word.append(corrent_word_node)
                corrent_type=true_type_node[true_index_node.index(corrent_sentence)]
                temp_corrent_word.append(corrent_type)
            else:
                temp_corrent_word.append(corrent_word_node)
                corrent_type = "FALSE"
                temp_corrent_word.append(corrent_type)
            result_word_type.append(temp_corrent_word)
            temp_corrent_word = []
    return result_word_type

#生成四个部分的word序列和对应的类型
def generator_input(result):
    start_left=[]
    start_right=[]
    end_left=[]
    end_right=[]
    target_type=[]

    for result_node in result:
        target_type.append(result_node[-1])
        corrent_node=result_node[0]
        start_left.append(corrent_node[0])
        corrent_node[1].reverse()
        start_right.append(corrent_node[1])
        end_left.append(corrent_node[2])
        corrent_node[3].reverse()
        end_right.append(corrent_node[3])
    return start_left,start_right,end_left,end_right,target_type

#调用上述函数
def generate_sentence(B_file_path,E_file_path,collection):

    word,B_label=nested_read(B_file_path,collection)
    _,E_label=nested_read(E_file_path,collection)

    total_match_word,total_sentence_order=all_index_match(word)
    true_entity_type,true_entity_index=match_entity(B_label,E_label)

    result=get_word_type(total_match_word,total_sentence_order,true_entity_index,true_entity_type)

    start_left,start_right,end_left,end_right,target_type=generator_input(result)

    return start_left,start_right,end_left,end_right,target_type

# 将字符表示为id，实现char和id之间的转换
def yuefei_prepare_dataset(sentences, char_to_id):
    data=[]
    for node in sentences:
        string=[w for w in node]
        id=[char_to_id[w if w in char_to_id else '<UNK>'] for w in string]
        data.append([string,id])
    return data

#类型转换的ID
def get_type_id(target_type):
    type = []
    for node in target_type:
        if (str(node) == "FALSE"):
            type.append([0, 1])
        else:
            type.append([1, 0])
    return type

#转换ID的总调用方法
def yuefei_sentence_to_id(start_left,start_right,end_left,end_right,char_to_id,target_type):

    start_left_id_data = yuefei_prepare_dataset(start_left,char_to_id)
    start_right_id_data = yuefei_prepare_dataset(start_right, char_to_id)
    end_left_id_data = yuefei_prepare_dataset(end_left, char_to_id)
    end_right_id_data = yuefei_prepare_dataset(end_right, char_to_id)
    train_new_type=get_type_id(target_type)

    return start_left_id_data,start_right_id_data,end_left_id_data,end_right_id_data,train_new_type

#测试预备集批次管理器
class yuefei_eval_BatchManager(object):
    def __init__(self, left_list, entity_left, entity_right, right_list,  batch_size):

        self.left_batch_data=self.split_by_size(left_list,batch_size)
        self.entity_left_batch_data = self.split_by_size(entity_left, batch_size)
        self.entity_right_batch_data = self.split_by_size(entity_right, batch_size)
        self.right_batch_data = self.split_by_size(right_list, batch_size)

    def split_by_size(self,data,batch_size):
        num_batch=int(math.ceil(len(data) /batch_size))
        batch_data=list()
        for i in range(num_batch):
            batch_data.append(data[i*batch_size : (i+1)*batch_size])
        return batch_data


#批次管理器
class yuefei_BatchManager(object):
    def __init__(self,left_list, entity_left, entity_right, right_list, target_type, batch_size):
        self.left_batch_data = self.split_by_size(left_list, batch_size)
        self.left_len_data = len(self.left_batch_data)
        self.left_max_sen_length = max([len(sentence) for sentence in left_list])

        self.entity_left_batch_data = self.split_by_size(entity_left, batch_size)
        self.entity_left_len_data = len(self.entity_left_batch_data)
        self.entity_left_max_sen_length = max([len(sentence) for sentence in entity_left])

        self.entity_right_batch_data = self.split_by_size(entity_right, batch_size)
        self.entity_right_len_data = len(self.entity_right_batch_data)
        self.entity_right_max_sen_length = max([len(sentence) for sentence in entity_right])

        self.right_batch_data = self.split_by_size(right_list, batch_size)
        self.right_len_data = len(self.right_batch_data)
        self.right_max_sen_length = max([len(sentence) for sentence in right_list])

        self.batch_type = self.split_type_by_size(target_type, batch_size)

    def split_by_size(self,data,batch_size):
        num_batch=int(math.ceil(len(data) /batch_size))
        batch_data=list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(data[i*batch_size : (i+1)*batch_size]))
        return batch_data

    def split_type_by_size(self,data,batch_size):
        num_batch=int(math.ceil(len(data) /batch_size))
        batch_data=list()
        for i in range(num_batch):
            batch_data.append(data[i*batch_size : (i+1)*batch_size])
        return batch_data

    @staticmethod
    def pad_data(data):
        chars = []
        max_length = max([len(sentence) for sentence in data])
        for line in data:
            char= line
            padding = [0] * (max_length - len(char))
            chars.append( char+padding )
        return [chars]


#批次管理器
class seq_BatchManager(object):
    def __init__(self,left_seq,left_list, entity_left, entity_right, right_list, right_seq,target_type, batch_size):
        self.left_seq_batch_data = self.split_by_size(left_seq, batch_size)
        self.left_seq_len_data = len(self.left_seq_batch_data)
        self.left_seq_max_sen_length = max([len(sentence) for sentence in left_seq])

        self.left_batch_data = self.split_by_size(left_list, batch_size)
        self.left_len_data = len(self.left_batch_data)
        self.left_max_sen_length = max([len(sentence) for sentence in left_list])

        self.entity_left_batch_data = self.split_by_size(entity_left, batch_size)
        self.entity_left_len_data = len(self.entity_left_batch_data)
        self.entity_left_max_sen_length = max([len(sentence) for sentence in entity_left])

        self.entity_right_batch_data = self.split_by_size(entity_right, batch_size)
        self.entity_right_len_data = len(self.entity_right_batch_data)
        self.entity_right_max_sen_length = max([len(sentence) for sentence in entity_right])

        self.right_batch_data = self.split_by_size(right_list, batch_size)
        self.right_len_data = len(self.right_batch_data)
        self.right_max_sen_length = max([len(sentence) for sentence in right_list])

        self.right_seq_batch_data = self.split_by_size(right_seq, batch_size)
        self.right_seq_len_data = len(self.right_seq_batch_data)
        self.right_seq_max_sen_length = max([len(sentence) for sentence in right_seq])

        self.batch_type = self.split_type_by_size(target_type, batch_size)

    def split_by_size(self,data,batch_size):
        num_batch=int(math.ceil(len(data) /batch_size))
        batch_data=list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(data[i*batch_size : (i+1)*batch_size]))
        return batch_data

    def split_type_by_size(self,data,batch_size):
        num_batch=int(math.ceil(len(data) /batch_size))
        batch_data=list()
        for i in range(num_batch):
            batch_data.append(data[i*batch_size : (i+1)*batch_size])
        return batch_data

    @staticmethod
    def pad_data(data):
        chars = []
        max_length = max([len(sentence) for sentence in data])
        for line in data:
            char= line
            padding = [0] * (max_length - len(char))
            chars.append( padding+char )
        return [chars]



#建立计算图
def yuefei_create_model(session, Model_class, path, load_vec, config, id_to_char, logger):
    '''
    :param session: 当前会话
    :param Model_class: 计算图
    :param path: FLAGS.ckpt_path路径
    :param load_vec:加载预训练embedding的函数
    :param config:计算图参数
    :param id_to_char:id_to_char
    :param logger:日志操作
    :return:
    '''
    # create model, reuse parameters if exists
    model = Model_class(config)

    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())#初始化所有变量，自动处理变量之间的依赖关系
        if config["pre_emb"]:
            temp=model.char_lookup
            emb_weights = session.run(model.char_lookup.read_value())#得到初始随机赋值的emb权重矩阵
            emb_weights = load_vec(config["emb_file"],id_to_char, config["char_dim"], emb_weights)
            session.run(model.char_lookup.assign(emb_weights))
            logger.info("Load pre-trained embedding.")
    return model

#生成数据
def yuefei_create_feed_dict(self, is_train, batch):



    _, chars, segs, tags = batch
    feed_dict = {self.char_inputs: np.asarray(chars), self.seg_inputs: np.asarray(segs), self.dropout: 1.0, }
    if is_train:
        temp = np.asarray(tags)
        feed_dict[self.targets] = np.asarray(tags)
        feed_dict[self.dropout] = self.config["dropout_keep"]
    return feed_dict


def yuefei_run_step(self, sess, is_train, batch):
    """
    :param sess: session to run the batch
    :param is_train: a flag indicate if it is a train batch
    :param batch: a dict containing batch data
    :return: batch result, loss of the batch or logits
    """
    feed_dict = self.create_feed_dict(is_train, batch)
    if is_train:
        global_step, loss, _ = sess.run([self.global_step, self.loss, self.train_op], feed_dict)
        return global_step, loss
    else:
        lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
        return lengths, logits

