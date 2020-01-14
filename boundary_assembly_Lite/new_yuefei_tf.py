'''
@文件名称: new_yuefei_tf.py
@作者: 武乐飞
@创建时间: 2018/12/31 - 13:47
@描述: 
'''

import re
import numpy as np
import chardet
import uuid
import itertools
import operator
import pickle
import random
from keras.preprocessing.sequence import pad_sequences
from Flags_define import *
import logging
import codecs
from a_correct_utils import *

def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def p_n_count(false_entity_e2e, entity_type_e2e,name):
    positive_num=0
    for sentence_type_node in entity_type_e2e:
        for type_node in sentence_type_node:
            if(len(type_node)!=0):
                positive_num+=1

    negative_num=0
    for sentence_false_node in false_entity_e2e:
        for false_node in sentence_false_node:
            if(len(false_node)!=0):
                negative_num+=1
    print(name+"集中正例数量:",positive_num,"         负例数量:",negative_num)


def pad_string(str,max_lenght):#直截断，不补零
    str_lenght=len(str)
    if(str_lenght>max_lenght):
        str=str[:max_lenght]
    return str

def pad_label(str_1,str_2,str_3,str_4):
    lenght=FLAGS.read_max_seq_langth
    new_str_1=pad_string(str_1,lenght)
    new_str_2 = pad_string(str_2, lenght)
    new_str_3 = pad_string(str_3, lenght)
    new_str_4 = pad_string(str_4, lenght)
    return new_str_1,new_str_2,new_str_3,new_str_4

#得到文本中的字符，不要标签,限制每句话的长度
def read_for_word_max(file_path,collected_num):
    lenght=FLAGS.read_max_seq_langth
    f=open(file_path,"r",encoding="utf-8")
    linelsit=f.readlines()

    result=[]
    sentense_temp=[]

    if(collected_num==-1):
        use_list=linelsit
    else:
        use_list=linelsit[:collected_num]
    for node in use_list:
        node_list=str(node).strip().split("\t")
        if(node_list[0]==""):
            sentense_temp=pad_string(sentense_temp,lenght)
            result.append(sentense_temp)
            sentense_temp=[]
            continue
        else:sentense_temp.append(node_list[0])

    return result

def get_sentense_max(train_path,dev_path,test_path,collected_num):

    H_train_sentence=read_for_word_max(train_path,collected_num)
    H_dev_sentence=read_for_word_max(dev_path,collected_num)
    H_test_sentence=read_for_word_max(test_path,collected_num)

    return H_train_sentence,H_dev_sentence,H_test_sentence

#把每个候选实体使用uuid做成全空间唯一表示
def uuid_laebl_percormance_new(left_list_node,entity_left_node,entity_right_node,right_list_node):

    left_list_string=return_str_list_test(left_list_node)
    entity_left_string=return_str_list_test(entity_left_node)
    entity_right_string = return_str_list_test(entity_right_node)
    right_list_string = return_str_list_test(right_list_node)
    # try:
    #     a=left_list_string[0]
    #     b=entity_left_string[0]
    #     c=entity_left_string[-1]
    #     d=right_list_string[-1]
    #     first_end_string=a+b+c+d
    # except:first_end_string="苍井空爱"

    uuid_left = uuid.uuid3(uuid.NAMESPACE_DNS, left_list_string)
    uuid_entity_left=uuid.uuid3(uuid.NAMESPACE_DNS, entity_left_string)
    uuid_entity_right = uuid.uuid3(uuid.NAMESPACE_DNS, entity_right_string)
    uuid_right = uuid.uuid3(uuid.NAMESPACE_DNS, right_list_string)
    # uuid_fe=uuid.uuid3(uuid.NAMESPACE_DNS, first_end_string)
    uuid_str=str(uuid_left)+"*"+str(uuid_entity_left)+"*"+str(uuid_right)+"*"+str(uuid_entity_right)
    return uuid_str

#得到文本中的字符，不要标签，不限制长度
def read_for_word(file_path,collected_num):

    f=open(file_path,"r",encoding="utf-8")
    linelsit=f.readlines()

    result=[]
    sentense_temp=[]

    if(collected_num==-1):
        use_list=linelsit
    else:
        use_list=linelsit[:collected_num]
    for node in use_list:
        node_list=str(node).strip().split("\t")
        if(node_list[0]==""):
            result.append(sentense_temp)
            sentense_temp=[]
            continue
        else:sentense_temp.append(node_list[0])

    return result

def get_sentense(train_path,dev_path,test_path,collected_num):

    H_train_sentence=read_for_word(train_path,collected_num)
    H_dev_sentence=read_for_word(dev_path,collected_num)
    H_test_sentence=read_for_word(test_path,collected_num)

    return H_train_sentence,H_dev_sentence,H_test_sentence


#统计各类数目
def get_num_class(label_list):
    if("array" in str(type(label_list))):
        label_list=label_list.tolist()
        # label_list=[node[0] for node in label_list]
    class_type={}
    for node in label_list:
        class_name=node[:3]
        if(class_name not in class_type):
            class_type[class_name]=1
        else:class_type[class_name]+=1
    b = sorted(class_type.items(), key=lambda item: item[0])
    test = {}
    for node in b:
        test[node[0]] = node[1]
    return test

#统计各类数目
def get_num_class_array(label_list):
    if("array" in str(type(label_list))):
        label_list=label_list.tolist()
        label_list=[node[0] for node in label_list]
    class_type={}
    for node in label_list:
        class_name=node[:3]
        if(class_name not in class_type):
            class_type[class_name]=1
        else:class_type[class_name]+=1
    b = sorted(class_type.items(), key=lambda item: item[0])
    test = {}
    for node in b:
        test[node[0]] = node[1]
    return test

#统计各类数目
def get_num_class_dict(label_dict):
    class_type={}
    for key,value in label_dict.items():
        if(value not in class_type):
            class_type[value]=1
        else:class_type[value]+=1
    return class_type


def get_line(a_list,line_num):
    return [node for node in a_list if float(node)>=line_num]

def diff_list(a,b,N):
    return sorted([a_node for a_node in a if (a_node not in b)], reverse=1)[:N]

def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    return dico

def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))  # 按照字出现的频率进行排序
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}  # 按照字出现的频率进行编号，出现频率最高的先行编号
    item_to_id = {v: k for k, v in id_to_item.items()}  # 按照字出现的频率进行编号，出现频率最高的先行编号，然后生成id_to-char字典
    return item_to_id, id_to_item



def generate_id_char_new(H_train_sentence, H_dev_sentence, H_test_sentence):
    # sub_word = H_train_sentence+H_dev_sentence+H_test_sentence
    sub_word = H_train_sentence+H_dev_sentence
    dico = create_dico(sub_word)
    char_to_id, id_to_char = create_mapping(dico)
    return char_to_id, id_to_char

def augment_with_pretrained_keras(dictionary, ext_emb_path, chars):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([line.rstrip().split()[0].strip() for line in codecs.open(ext_emb_path, 'r', 'utf-8') if len(ext_emb_path) > 0 ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            aa=[char,char.lower(),re.sub('\d', '0', char.lower())]
            if any(x in pretrained for x in [char,char.lower(),re.sub('\d', '0', char.lower())]) and char not in dictionary:
                dictionary[char] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word

def char_mapping_keras(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    chars = [[x.lower() if lower else x[0] for x in s] for s in sentences]

    dico = create_dico(chars)

    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in chars)
    ))
    return dico, char_to_id, id_to_char

def generate_id_char_keras(H_train_sentence, H_dev_sentence, H_test_sentence):

    dico_chars_train = char_mapping_keras(H_train_sentence, FLAGS.lower)[0]

    dico_chars, char_to_id, id_to_char = \
        augment_with_pretrained_keras(dico_chars_train.copy(),
                                FLAGS.emb_file,
                                list(itertools.chain.from_iterable([[w for w in s] for s in H_dev_sentence+ H_test_sentence])))
    return char_to_id, id_to_char

def sentence_transfor_id(sentence,char_to_id):

    new_sentence=[[word for word in a_node] for a_node in sentence]
    for x,sentence_node in enumerate(new_sentence):
        for y,word_node in enumerate(sentence_node):
            if(new_sentence[x][y] in char_to_id):
                new_sentence[x][y]=char_to_id[new_sentence[x][y]]
            else:
                new_sentence[x][y] = 0
    return new_sentence

def sub_transfor_id(train_sentence,dev_sentence,test_sentence,char_to_id):

    x_train=sentence_transfor_id(train_sentence,char_to_id)
    x_dev=sentence_transfor_id(dev_sentence,char_to_id)
    x_test=sentence_transfor_id(test_sentence,char_to_id)

    return x_train,x_dev,x_test


#从边界预测结果中得到的候选实体，按照和真实实体对比得出的结果，生成数据的生成
def generate_train_dev_input(true_entity,true_entity_type,false_entity,BIO_seq):

    left_input=[]
    entity_left_input=[]
    entity_right_input=[]
    right_input=[]

    total_type=[]

    for entity_node,type_node,seq_node in zip(true_entity,true_entity_type,BIO_seq):
        if(len(entity_node)==0):
            continue
        else:
            for entity,type in zip(entity_node ,type_node):
                a = entity[0]
                b = entity[-1]

                left_input.append(seq_node[:a+1])
                entity_left_input.append(seq_node[a:b+1])

                entity_right_temp=seq_node[a:b+1]
                entity_right_temp.reverse()
                entity_right_input.append(entity_right_temp)

                right_temp = seq_node[b:]
                right_temp.reverse()
                right_input.append(right_temp)

                total_type.append(type)

    for false_entity_node ,false_seq_node in zip(false_entity,BIO_seq):
        if(len(false_entity_node)==0):
            continue
        else:
            for false_entity in false_entity_node:
                c=false_entity[0]
                d=false_entity[-1]

                left_input.append(false_seq_node[:a + 1])
                entity_left_input.append(false_seq_node[a:b + 1])

                false_entity_right_temp = false_seq_node[a:b + 1]
                false_entity_right_temp.reverse()
                entity_right_input.append(false_entity_right_temp)

                false_right_temp = false_seq_node[b:]
                false_right_temp.reverse()
                right_input.append(false_right_temp)

                total_type.append("NEG")


    total_input=[]
    for aleft_node,aentity_left_node ,aentity_right_node,aright_node,atype_node in zip(left_input,entity_left_input,entity_right_input,right_input,total_type):
        temp=[]
        temp.append(aleft_node)
        temp.append(aentity_left_node)
        temp.append(aentity_right_node)
        temp.append(aright_node)
        temp.append(atype_node)
        total_input.append(temp)

    random.shuffle(total_input)

    shit_left_input = []
    shit_entity_left_input = []
    shit_entity_right_input = []
    shit_right_input = []

    shit_total_type = []

    for shit_name in total_input:
        shit_left_input.append(shit_name[0])
        shit_entity_left_input.append(shit_name[1])
        shit_entity_right_input.append(shit_name[2])
        shit_right_input.append(shit_name[3])
        shit_total_type.append(shit_name[-1])

    # new_left_input=pad_sequences(shit_left_input,FLAGS.seq_langth)
    # new_entity_left_input=pad_sequences(shit_entity_left_input,FLAGS.seq_langth)
    # new_entity_right_input=pad_sequences(shit_entity_right_input,FLAGS.seq_langth)
    # new_right_input=pad_sequences(shit_right_input,FLAGS.seq_langth)

    new_left_input = shit_left_input
    new_entity_left_input = shit_entity_left_input
    new_entity_right_input =shit_entity_right_input
    new_right_input = shit_right_input


    return new_left_input,new_entity_left_input,new_entity_right_input,new_right_input,shit_total_type

#求列表的差集
def list_difference(a,b):
    '''
    描述:求两个列表的差集
    '''

    false=[]
    for a_node,b_node in zip(a,b):
        temp=[]
        for node in a_node:
            if(node not in b_node):
                temp.append(node)
        false.append(temp)
        temp=[]
    return false

#求列表的交集，返回的是一个列表
def get_same_elem(aList,bList):
    '''
    描述:求两个列表的交集
    '''

    aset = set(aList)
    bset = set(bList)
    result=list(aset.intersection(bset))
    return result

#crf正例性能
def performance_on_positives_binary(y_test, y_pred):

    y_pred_index = y_pred.reshape(-1, 1)#预测的结果集
    y_test_index = y_test.reshape(-1, 1)#测试集中的数据

    if len(y_pred_index) != len(y_test_index):
        print("y_pred_index and y_test_index should be equal...")
        os.system("pause")

    true_index = 0#（1,0）
    false_index = 1

    positive_true_test_num = 0#测试集中的正例
    positive_pred_num = 0#模型总共预测了多少个正例
    positive_true_pre_num=0#模型预测的正例中有多少个是正确的


    negative_true_test_num = 0#测试集中的负例
    negative_pred_num = 0#模型总共预测了多少个负例
    negative_true_pre_num=0#模型预测的负例中有多少个是正确的

    for test_node in y_test_index:
        if(test_node==true_index):
            positive_true_test_num+=1
        else:negative_true_test_num+=1

    for pre_node in y_pred_index:
        if (pre_node == true_index):
            positive_pred_num += 1
        else:negative_pred_num+=1

    for pre,test in zip(y_pred_index,y_test_index):
        if((pre==true_index)and(test==true_index)and(pre==test)):
            positive_true_pre_num+=1
        if ((pre == false_index) and (test == false_index) and (pre == test)):
            negative_true_pre_num += 1

    print("-------------------------正例性能---------------------------------")
    print("测试集中的正例: " + str(positive_true_test_num))
    print("预测结果中的正例:" + str(positive_pred_num))
    print("预测为正例的结果中真正的正例:" + str(positive_true_pre_num))
    P = 0 if positive_pred_num == 0 else 100. * positive_true_pre_num / positive_pred_num
    R = 0 if positive_true_pre_num == 0 else 100. * positive_true_pre_num / positive_true_test_num
    F = 0 if P + R == 0 else 2 * P * R / (P + R)
    print("Precision: %.2f" % (P),"%")
    print("Recall: %.2f" % (R),"%")
    print("F1: %.2f" % (F),"%")

    print()
    print()

    print("-------------------------负例性能---------------------------------")
    print("测试集中的负例: " + str(negative_true_test_num))
    print("预测结果中的负例:" + str(negative_pred_num))
    print("预测为负例的结果中真正的负例:" + str(negative_true_pre_num))
    P = 0 if negative_pred_num == 0 else 100. * negative_true_pre_num / negative_pred_num
    R = 0 if negative_true_pre_num == 0 else 100. * negative_true_pre_num / negative_true_test_num
    F = 0 if P + R == 0 else 2 * P * R / (P + R)
    print("Precision: %.2f" % (P), "%")
    print("Recall: %.2f" % (R), "%")
    print("F1: %.2f" % (F), "%")

#多分类性能测试
def performance_on_positives_boundry_mul_class(y_test, y_pred):

    y_pred_index = y_pred.reshape(-1, 1)#预测的结果集
    y_test_index = y_test.reshape(-1, 1)#测试集中的数据

    if len(y_pred_index) != len(y_test_index):
        print("y_pred_index and y_test_index should be equal...")
        os.system("pause")

    true_index_list=[0,1,2,3,4,5,6]
    type_list=["VEH","LOC","WEA","GPE","PER","ORG","FAC"]

    list_true_test_num=[0,0,0,0,0,0,0]#测试集中的正例，需要统计

    list_pred_num=[]##模型预测了多少个正例
    list_true_pre_num=[]#模型预测的正例中有多少个是正确的
    temp=[]

    for true_index,type,true_test_num in zip(true_index_list,type_list,list_true_test_num):

        pred_num = 0#模型总共预测了多少个正例
        true_pre_num=0#模型预测的正例中有多少个是正确的

        for pre_node in y_pred_index:
            if (pre_node == true_index):
                pred_num += 1

        for pre,test in zip(y_pred_index,y_test_index):
            if((pre==true_index)and(test==true_index)and(pre==test)):
                true_pre_num+=1

        list_pred_num.append(pred_num)
        list_true_pre_num.append(true_pre_num)
        temp.append(type)

        print("-------------------------*",type,"*---------------------------------")
        print("|测试集中的正例: " + str(true_test_num))
        print("|预测结果中的正例:" + str(pred_num))
        print("|预测为正例的结果中真正的正例:" + str(true_pre_num))
        P = 0 if pred_num == 0 else 100. * true_pre_num / pred_num
        R = 0 if true_pre_num == 0 else 100. * true_pre_num / true_test_num
        F = 0 if P + R == 0 else 2 * P * R / (P + R)
        print("|Precision: %.2f" % (P),"%")
        print("|Recall: %.2f" % (R),"%")
        print("|F1: %.2f" % (F),"%")
        print()
        print()

    total_true_test_num=sum(list_true_test_num)
    total_pred_num=sum(list_pred_num)
    total_true_pre_num=sum(list_true_pre_num)

    print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_")
    print("各类总性能：",temp)
    print("-------------------------*Total Performance*---------------------------------")
    print("|测试集中各类的正例总数: " + str(total_true_test_num))
    print("|预测结果中各类的正例总数:" + str(total_pred_num))
    print("|预测为正例的结果中各类真正的正例总数:" + str(total_true_pre_num))
    P = 0 if total_pred_num == 0 else 100. * total_true_pre_num / total_pred_num
    R = 0 if total_true_pre_num == 0 else 100. * total_true_pre_num / total_true_test_num
    F = 0 if P + R == 0 else 2 * P * R / (P + R)
    print("|Precision: %.2f" % (P), "%")
    print("|Recall: %.2f" % (R), "%")
    print("|F1: %.2f" % (F), "%")
    print()
    print()



#crf正例性能
def performance_on_positives_bert(y_test, y_pred):

    print("开始性能评估..............")

    y_pred_index = y_pred.reshape(-1, 1)#预测的结果集
    y_test_index = y_test.reshape(-1, 1)#测试集中的数据

    if len(y_pred_index) != len(y_test_index):
        print("y_pred_index and y_test_index should be equal...")
        os.system("pause")

    true_index = 0#（1,0）
    false_index = 1

    positive_true_test_num = "*"#统计数目
    positive_pred_num = 0#模型总共预测了多少个正例
    positive_true_pre_num=0#模型预测的正例中有多少个是正确的


    negative_true_test_num = 0#测试集中的负例
    negative_pred_num = 0#模型总共预测了多少个负例
    negative_true_pre_num=0#模型预测的负例中有多少个是正确的

    for test_node in y_test_index:#统计测试集中负的个数，正例的个数已经给出
        if(test_node==false_index):
            negative_true_test_num += 1

    for pre_node in y_pred_index:
        if (pre_node == true_index):
            positive_pred_num += 1
        else:negative_pred_num+=1

    for pre,test in zip(y_pred_index,y_test_index):
        if((pre==true_index==test)):
            positive_true_pre_num+=1
        if ((pre == false_index== test)):
            negative_true_pre_num += 1

    print("-------------------------正例性能---------------------------------")
    print("测试集中的正例: " + str(positive_true_test_num))
    print("预测结果中的正例:" + str(positive_pred_num))
    print("预测为正例的结果中真正的正例:" + str(positive_true_pre_num))
    P = 0 if positive_pred_num == 0 else 100. * positive_true_pre_num / positive_pred_num
    R = 0 if positive_true_pre_num == 0 else 100. * positive_true_pre_num / positive_true_test_num
    F = 0 if P + R == 0 else 2 * P * R / (P + R)
    print("Precision: %.2f" % (P),"%")
    print("Recall: %.2f" % (R),"%")
    print("F1: %.2f" % (F),"%")

    print()
    print()

    print("-------------------------负例性能---------------------------------")
    print("测试集中的负例: " + str(negative_true_test_num))
    print("预测结果中的负例:" + str(negative_pred_num))
    print("预测为负例的结果中真正的负例:" + str(negative_true_pre_num))
    P = 0 if negative_pred_num == 0 else 100. * negative_true_pre_num / negative_pred_num
    R = 0 if negative_true_pre_num == 0 else 100. * negative_true_pre_num / negative_true_test_num
    F = 0 if P + R == 0 else 2 * P * R / (P + R)
    print("Precision: %.2f" % (P), "%")
    print("Recall: %.2f" % (R), "%")
    print("F1: %.2f" % (F), "%")

#crf正例性能
def performance_on_positives_bert_logger(y_test, y_pred,logger):

    logger.info("测试集--> 开始性能评估..............")

    y_pred_index = y_pred.reshape(-1, 1)#预测的结果集
    y_test_index = y_test.reshape(-1, 1)#测试集中的数据

    if len(y_pred_index) != len(y_test_index):
        logger.info("y_pred_index and y_test_index should be equal...")
        os.system("pause")

    true_index = 0#（1,0）
    false_index = 1

    positive_true_test_num = "*"#实体总数
    positive_pred_num = 0#模型总共预测了多少个正例
    positive_true_pre_num=0#模型预测的正例中有多少个是正确的


    negative_true_test_num = 0#测试集中的负例
    negative_pred_num = 0#模型总共预测了多少个负例
    negative_true_pre_num=0#模型预测的负例中有多少个是正确的

    for test_node in y_test_index:#统计测试集中负的个数，正例的个数已经给出
        if(test_node==false_index):
            negative_true_test_num += 1

    for pre_node in y_pred_index:
        if (pre_node == true_index):
            positive_pred_num += 1
        else:negative_pred_num+=1

    for pre,test in zip(y_pred_index,y_test_index):
        if((pre==true_index==test)):
            positive_true_pre_num+=1
        if ((pre == false_index== test)):
            negative_true_pre_num += 1

    logger.info("-------------------------正例性能---------------------------------")
    logger.info("测试集中的正例: {0}".format(str(positive_true_test_num)))
    logger.info("预测结果中的正例: {0}".format(str(positive_pred_num)))
    logger.info("预测为正例的结果中真正的正例: {0}".format(str(positive_true_pre_num)))
    P = 0 if positive_pred_num == 0 else 100. * positive_true_pre_num / positive_pred_num
    R = 0 if positive_true_pre_num == 0 else 100. * positive_true_pre_num / positive_true_test_num
    F = 0 if P + R == 0 else 2 * P * R / (P + R)
    P = round(P, 2)
    R = round(R, 2)
    F = round(F, 2)
    logger.info("Precision:  {0}".format(str(P)))
    logger.info("Recall: {0}".format(str(R)))
    logger.info("F1: {0}".format(str(F)))

    print()
    print()

    logger.info("-------------------------负例性能---------------------------------")
    logger.info("测试集中的负例:  {0}".format(str(negative_true_test_num)))
    logger.info("预测结果中的负例: {0}".format(str(negative_pred_num)))
    logger.info("预测为负例的结果中真正的负例: {0}".format(str(negative_true_pre_num)))
    P = 0 if negative_pred_num == 0 else 100. * negative_true_pre_num / negative_pred_num
    R = 0 if negative_true_pre_num == 0 else 100. * negative_true_pre_num / negative_true_test_num
    F = 0 if P + R == 0 else 2 * P * R / (P + R)
    P = round(P, 2)
    R = round(R, 2)
    F = round(F, 2)
    logger.info("Precision:  {0}".format(str(P)))
    logger.info("Recall: {0}".format(str(R)))
    logger.info("F1: {0}".format(str(F)))

#描述:如果a和b两个标签属于一个实体，返回true和该实体的类型（8种之一）
def get_state_type(a,b):
    '''
    描述:如果a和b两个标签属于一个实体，返回true和该实体的类型（8种之一）
    ----	VEH		[1,0,0,0,0,0,0,0]
    地名	LOC		[0,1,0,0,0,0,0,0]
    ----	WEA		[0,0,1,0,0,0,0,0]
    ----	GPE		[0,0,0,1,0,0,0,0]
    人名	PER		[0,0,0,0,1,0,0,0]
    组织	ORG		[0,0,0,0,0,1,0,0]
    ----	FAC		[0,0,0,0,0,0,1,0]
    负例	NEG		[0,0,0,0,0,0,0,1]
    '''
    state=False
    entity_Type=""
    same_a_List=str(a.split(",")[0]).split("-")[1:]
    same_b_List = str(b.split(",")[0]).split("-")[1:]
    same_result=get_same_elem(same_a_List,same_b_List)

    type_a_List=str(a.split(",")[1]).split("-")
    type_b_List = str(b.split(",")[1]).split("-")
    entity_type=get_same_elem(type_a_List,type_b_List)

    if(len(same_result)!=0):
        state=True
        entity_Type=entity_type[0]

    return state,entity_Type

def count_test_example(class_dict):

    class_type = "VEH-LOC-WEA-GPE-PER-ORG-FAC"
    true=0
    false=0

    for k,v in class_dict.items():
        if(k in class_type):
            true+=v
        else:
            false+=v
    return true,false

def count_POS_one(a_list):
    count=0
    for node in a_list:
        if(node==1):
            count+=1
    return count


def count_POS(a_list):
    count=0
    for node in a_list:
        if(node==[1,0]):
            count+=1
    return count

def oneTo_eight(a):
    dict_class={"VEH":[1,0,0,0,0,0,0,0],"LOC":[0,1,0,0,0,0,0,0],"WEA":[0,0,1,0,0,0,0,0],"GPE":[0,0,0,1,0,0,0,0],"PER":[0,0,0,0,1,0,0,0],"ORG":[0,0,0,0,0,1,0,0],"FAC":[0,0,0,0,0,0,1,0],"NEG":[0,0,0,0,0,0,0,1]}
    a=a.tolist()
    result=[]
    for node in a:
        result.append(dict_class[str(node)])
    new_result=np.asanyarray(result)
    final_result = np.reshape(new_result, (-1, 8))
    return final_result

#将包含8种分类的数据按照二分类进行处理
def eightTo_two_array(a):
    dict_class={"NEG":[0,1],"POS":[1,0]}
    result=[]
    for node in a:
        if("NEG" in str(node)):
            result.append(dict_class["NEG"])
        else:result.append(dict_class["POS"])
    pos_count=count_POS(result)
    new_result = np.asanyarray(result)
    final_result = np.reshape(new_result, (-1, 2))
    return final_result,pos_count

#左边补零
def add_zeo_left_bert(left_list):
    pad_embedding=pickle.load(open(FLAGS.pad_emb,"rb"))
    result = []
    for x in range(len(left_list)):
        left_temp=left_list[x]
        length=left_temp.shape[0]

        if (length == FLAGS.seq_langth):
            result.append(left_temp)
        if (length > FLAGS.seq_langth):
            result.append(left_temp[:FLAGS.seq_langth])
        if (length < FLAGS.seq_langth):
            pad_array=np.reshape(np.tile(pad_embedding,(FLAGS.seq_langth-length)),(-1,pad_embedding.shape[-1]))
            paded=np.concatenate([pad_array,left_temp],axis=0)
            result.append(paded)
    result_reshape=np.reshape(np.concatenate(result,axis=0),(-1,FLAGS.seq_langth,FLAGS.emb_dim))
    return result_reshape

#左边补零
def add_zeo_left(left_list):
    temp = [0]
    result = []
    for x in range(len(left_list)):
        left_temp=left_list[x]
        left_length=len(left_temp)
        if (left_length == FLAGS.seq_langth):
            result.append(left_temp)
        if (left_length > FLAGS.seq_langth):
            result.append(left_temp[:FLAGS.seq_langth])
        if (left_length < FLAGS.seq_langth):
            result.append((FLAGS.seq_langth - len(left_list[x])) * temp+left_temp)
    return result

#列表无连接复制
def list_copy(old_list):
    '''
    描述:用以复制新列表，该新列表与旧列表之间没有关系
    '''

    new_list=[]
    for node in old_list:
        new_list.append(node)
    return new_list

#将包含8种分类的数据按照二分类进行处理
def eightTo_two(a):
    dict_class={"NEG":[0,1],"POS":[1,0]}
    result=[]
    for node in a:
        if("NEG" in str(node)):
            result.append(dict_class["NEG"])
        else:result.append(dict_class["POS"])
    pos_count=count_POS(result)
    # new_result = np.asanyarray(result)
    # final_result = np.reshape(new_result, (-1, 2))
    return result,pos_count

#按照候选实体把句子分成三部分，以及生成label,多分类问题使用
def get_entity_input_mul_class_old(BIO_charseq,total_entity,true_entity,entity_type,false_entity):
    '''
    Desc:接受的都是二维列表，所以需要两次循环
    '''
    entity_input=[]

    for true_entity_node,entity_type_node,false_entity_node ,BIO_charseq_node in zip(true_entity,entity_type,false_entity,BIO_charseq):
        entity_input_temp=[]
        true_entity_input=[]
        false_entity_input = []
        for node,type_ndoe in zip(true_entity_node,entity_type_node):
            a=node[0]
            b=node[1]

            true_entity_input.append([[BIO_charseq_node[:a+1],BIO_charseq_node[a:b+1],BIO_charseq_node[b:]],str(type_ndoe)])
        for node in false_entity_node:
            a = node[0]
            b = node[1]
            false_entity_input.append([[BIO_charseq_node[:a+1],BIO_charseq_node[a:b+1],BIO_charseq_node[b:]],"NEG"])
        entity_input_temp=[i for i in true_entity_input+false_entity_input]
        # random.shuffle(entity_input_temp)
        entity_input.append(entity_input_temp)
    return entity_input

#按照候选实体把句子分成三部分，以及生成label,多分类问题使用
def get_entity_input_mul_class_bert_emb(bert_word_emb,total_entity,true_entity,entity_type,false_entity):
    '''
    Desc:接受的都是二维列表，所以需要两次循环
    '''
    entity_input=[]

    for true_entity_node,entity_type_node,false_entity_node ,BIO_charseq_node in zip(true_entity,entity_type,false_entity,bert_word_emb):
        entity_input_temp=[]
        true_entity_input=[]
        false_entity_input = []
        for node,type_ndoe in zip(true_entity_node,entity_type_node):
            a=node[0]
            b=node[1]
            true_entity_input.append([[BIO_charseq_node[:a+1],BIO_charseq_node[a:b+1],BIO_charseq_node[b:]],str(type_ndoe)])
        for node in false_entity_node:
            a = node[0]
            b = node[1]
            false_entity_input.append([[BIO_charseq_node[:a+1],BIO_charseq_node[a:b+1],BIO_charseq_node[b:]],"NEG"])
        entity_input_temp=[i for i in true_entity_input+false_entity_input]
        entity_input.append(entity_input_temp)
    return entity_input

#按照候选实体把句子分成三部分，以及生成label,多分类问题使用
def get_entity_input_mul_class(BIO_charseq,total_entity,true_entity,entity_type,false_entity):
    '''
    Desc:接受的都是二维列表，所以需要两次循环
    '''
    entity_input=[]

    for true_entity_node,entity_type_node,false_entity_node ,BIO_charseq_node in zip(true_entity,entity_type,false_entity,BIO_charseq):
        entity_input_temp=[]
        true_entity_input=[]
        false_entity_input = []
        for node,type_ndoe in zip(true_entity_node,entity_type_node):
            a=node[0]
            b=node[1]
            true_entity_input.append([[BIO_charseq_node,BIO_charseq_node[:a+1],BIO_charseq_node[a:b+1],BIO_charseq_node[b:],BIO_charseq_node],str(type_ndoe)])
        for node in false_entity_node:
            a = node[0]
            b = node[1]
            false_entity_input.append([[BIO_charseq_node,BIO_charseq_node[:a+1],BIO_charseq_node[a:b+1],BIO_charseq_node[b:],BIO_charseq_node],"NEG"])
        entity_input_temp=[i for i in true_entity_input+false_entity_input]
        # random.shuffle(entity_input_temp)
        entity_input.append(entity_input_temp)
    return entity_input

#按照候选实体把句子分成三部分，以及生成label,多分类问题使用
def get_entity_input_mul_class_tf(BIO_charseq,total_entity,true_entity,entity_type,false_entity):
    '''
    Desc:接受的都是二维列表，所以需要两次循环
    '''
    entity_input=[]

    for true_entity_node,entity_type_node,false_entity_node ,BIO_charseq_node in zip(true_entity,entity_type,false_entity,BIO_charseq):
        entity_input_temp=[]
        true_entity_input=[]
        false_entity_input = []
        for node,type_ndoe in zip(true_entity_node,entity_type_node):
            a=node[0]
            b=node[1]

            true_entity_input.append([[BIO_charseq_node[:a+1],BIO_charseq_node[a:b+1],BIO_charseq_node[b:]],str(type_ndoe)])
        for node in false_entity_node:
            a = node[0]
            b = node[1]
            false_entity_input.append([[BIO_charseq_node[:a+1],BIO_charseq_node[a:b+1],BIO_charseq_node[b:]],"NEG"])
        entity_input_temp=[i for i in true_entity_input+false_entity_input]
        entity_input.append(entity_input_temp)
    return entity_input

#把生成三部分的句子左右补零，生成模型需要的形式
def generate_input_lab_emb_bert(aList):
    # random.shuffle(aList)
    # random.shuffle(aList)
    left_list = []
    entity_left = []
    entity_right = []
    right_list = []
    label = []
    for node in aList:
        if (len(node) == 0):
            continue
        for element in node:

            left_list.append(element[0][0])

            entity_left.append(element[0][1])

            temp_array=element[0][1]
            entity_right.append(temp_array[::-1])

            right_temp=element[0][2]
            right_list.append(right_temp[::-1])

            label.append(element[1])

    add_left_list = add_zeo_left_bert(left_list)
    add_entity_left = add_zeo_left_bert(entity_left)
    add_entity_right = add_zeo_left_bert(entity_right)
    add_right_list = add_zeo_left_bert(right_list)

    label = np.asanyarray(label)
    aa=FLAGS.seq_langth
    return add_left_list,add_entity_left,add_entity_right,add_right_list,label

#把生成三部分的句子左右补零，生成模型需要的形式
def generate_input_lab(aList):
    # random.shuffle(aList)
    # random.shuffle(aList)
    left_seq = []
    left_list = []
    entity_left = []
    entity_right = []
    right_list = []
    right_seq = []
    label = []
    for node in aList:
        if (len(node) == 0):
            continue
        for element in node:
            left_seq.append(element[0][0])

            left_list.append(element[0][1])

            entity_left.append(element[0][2])

            temp_list = list_copy(element[0][2])
            temp_list.reverse()
            entity_right.append(temp_list)

            right_temp = list_copy(element[0][3])
            right_temp.reverse()
            right_list.append(right_temp)

            right_seq_temp = list_copy(element[0][0])
            right_seq_temp.reverse()
            right_seq.append(right_seq_temp)

            label.append(element[1])

    add_left_seq = pad_sequences(left_seq, FLAGS.seq_langth, padding="post")
    add_left_list = add_zeo_left(left_list)
    add_entity_left = add_zeo_left(entity_left)
    add_entity_right = add_zeo_left(entity_right)
    add_right_list = add_zeo_left(right_list)
    add_right_seq = pad_sequences(right_seq, FLAGS.seq_langth, padding="pre")

    new_left_seq = np.asanyarray(add_left_seq)
    new_left_list = np.asanyarray(add_left_list)
    new_entity_left = np.asanyarray(add_entity_left)
    new_entity_right = np.asanyarray(add_entity_right)
    new_right_list = np.asanyarray(add_right_list)
    new_right_seq = np.asanyarray(add_right_seq)

    label = np.asanyarray(label)
    aa=FLAGS.seq_langth
    return np.reshape(new_left_seq, (-1, FLAGS.seq_langth)),np.reshape(new_left_list, (-1, FLAGS.seq_langth)),np.reshape(new_entity_left, (-1, FLAGS.seq_langth)),np.reshape(new_entity_right, (-1, FLAGS.seq_langth)),np.reshape(new_right_list, (-1, FLAGS.seq_langth)),np.reshape(new_right_seq, (-1, FLAGS.seq_langth)), np.reshape(label, (-1,FLAGS.reshape_dim))  # 原始语句

#把生成三部分的句子左右补零，生成模型需要的形式
def generate_input_old(aList):

    left_list = []
    entity_left = []
    entity_right = []
    right_list = []
    label = []
    for node in aList:
        if (len(node) == 0):
            continue
        for element in node:

            left_list.append(element[0][0])

            entity_left.append(element[0][1])

            temp_list = list_copy(element[0][1])
            temp_list.reverse()
            entity_right.append(temp_list)

            right_temp = list_copy(element[0][2])
            right_temp.reverse()
            right_list.append(right_temp)


            label.append(element[1])

    add_left_list = add_zeo_left(left_list)
    add_entity_left = add_zeo_left(entity_left)
    add_entity_right = add_zeo_left(entity_right)
    add_right_list = add_zeo_left(right_list)

    new_left_list = np.asanyarray(add_left_list)
    new_entity_left = np.asanyarray(add_entity_left)
    new_entity_right = np.asanyarray(add_entity_right)
    new_right_list = np.asanyarray(add_right_list)

    label = np.asanyarray(label)
    aa=FLAGS.seq_langth
    return np.reshape(new_left_list, (-1, FLAGS.seq_langth)),np.reshape(new_entity_left, (-1, FLAGS.seq_langth)),np.reshape(new_entity_right, (-1, FLAGS.seq_langth)),np.reshape(new_right_list, (-1, FLAGS.seq_langth)), np.reshape(label, (-1,FLAGS.reshape_dim))  # 原始语句


#把生成三部分的句子左右补零，生成模型需要的形式
def generate_input_lab_tf(aList):
    # random.shuffle(aList)
    # random.shuffle(aList)
    left_seq = []
    left_list = []
    entity_left = []
    entity_right = []
    right_list = []
    right_seq = []
    label = []
    for node in aList:
        if (len(node) == 0):
            continue
        for element in node:
            left_seq.append(element[0][0])

            left_list.append(element[0][1])

            entity_left.append(element[0][2])

            temp_list = list_copy(element[0][2])
            temp_list.reverse()
            entity_right.append(temp_list)

            right_temp = list_copy(element[0][3])
            right_temp.reverse()
            right_list.append(right_temp)

            right_seq_temp = list_copy(element[0][0])
            right_seq_temp.reverse()
            right_seq.append(right_seq_temp)

            label.append(element[1])


    return left_seq,left_list, entity_left, entity_right, right_list,right_seq, label


#把生成三部分的句子左右补零，生成模型需要的形式
def generate_input_list(aList):
    # random.shuffle(aList)
    # left_seq=[]
    left_list=[]
    entity_left=[]
    entity_right=[]
    right_list=[]
    # right_seq=[]
    label=[]
    for node in aList:
        if(len(node)==0):
            continue
        for element in node:

            left_list.append(element[0][0])
            entity_left.append(element[0][1])

            temp_list=list_copy(element[0][1])
            temp_list.reverse()
            entity_right.append(temp_list)

            right_temp=list_copy(element[0][2])
            right_temp.reverse()
            right_list.append(right_temp)

            label.append(element[1])

    return left_list,entity_left,entity_right,right_list,label

#把生成三部分的句子左右补零，生成模型需要的形式
def generate_input_list_new(aList):
    # random.shuffle(aList)
    left_seq=[]
    left_list=[]
    entity_left=[]
    entity_right=[]
    right_list=[]
    right_seq=[]
    label=[]
    for node in aList:
        if(len(node)==0):
            continue
        for element in node:
            left_seq.append(element[0][0])
            left_list.append(element[0][1])
            entity_left.append(element[0][2])

            temp_list=list_copy(element[0][2])
            temp_list.reverse()
            entity_right.append(temp_list)

            right_temp=list_copy(element[0][3])
            right_temp.reverse()
            right_list.append(right_temp)

            right_seq_temp = list_copy(element[0][4])
            right_seq_temp.reverse()
            right_seq.append(right_temp)

            label.append(element[1])

    return left_seq,left_list,entity_left,entity_right,right_list,right_seq,label


#把测试候选实体集从id转换为汉字，然后和real_uuid进行对照，求出性能
def transfor_pre_test(test_left_list, test_entity_left, test_entity_right, test_right_list,H_id_to_char):

    new_left=test_left_list
    new_entity_left=test_entity_left
    new_entity_right=test_entity_right
    new_right=test_right_list

    for x_left,new_left_node in enumerate(new_left):
        for y_left,new_left_word in enumerate(new_left_node):
            if(new_left[x_left][y_left]!=0):
                new_left[x_left][y_left]=H_id_to_char[new_left[x_left][y_left]]

    for x_entity_left,new_entity_left_node in enumerate(new_entity_left):
        for y_entity_left,new_entity_left_word in enumerate(new_entity_left_node):
            if(new_entity_left[x_entity_left][y_entity_left]!=0):
                new_entity_left[x_entity_left][y_entity_left]=H_id_to_char[new_entity_left[x_entity_left][y_entity_left]]

    for x_entity_right,new_entity_right_node in enumerate(new_entity_right):
        for y_entity_right,new_entity_right_word in enumerate(new_entity_right_node):
            if(new_entity_right[x_entity_right][y_entity_right]!=0):
                new_entity_right[x_entity_right][y_entity_right]=H_id_to_char[new_entity_right[x_entity_right][y_entity_right]]

    for x_right,new_right_node in enumerate(new_right):
        for y_right,new_right_word in enumerate(new_right_node):
            if(new_right[x_right][y_right]!=0):
                new_right[x_right][y_right]=H_id_to_char[new_right[x_right][y_right]]

    return new_left,new_entity_left,new_entity_right,new_right


#得到候选实体的正例(多分类）和负例,多分类问题使用本函数
def get_real_entity_mul_calss(B_index,E_index,B_label,E_label):


    total_candidate=[]
    true_candidate=[]
    true_candidate_type=[]
    false_candidate=[]

    # total_entity = get_match_N(B_index, E_index)  # greedy匹配
    # total_entity = get_fb_match_N(B_index, E_index)#bi-greedy

    for B_index_node,E_index_node in zip(B_index,E_index):#这部分不是贪心策略，向前全匹配，和上面按个代码二选一
        total_temp=[]
        for x in E_index_node[::-1]:
            for y in B_index_node[::-1]:
                if(y<=x):
                    total_temp.append([y,x])
        total_candidate.append(total_temp)

    for candidate,B_label_node,E_label_node in zip(total_candidate,B_label,E_label):
        true_temp=[]
        true_type_temp=[]
        for candidate_node in candidate:
            a=B_label_node[candidate_node[0]]
            b=E_label_node[candidate_node[1]]
            state,type=get_state_type(B_label_node[candidate_node[0]],E_label_node[candidate_node[1]])
            if(state):
                true_temp.append(candidate_node)
                true_type_temp.append(type)
        true_candidate.append(true_temp)
        true_candidate_type.append(true_type_temp)
    false_candidate=list_difference(total_candidate,true_candidate)
    return total_candidate,true_candidate,true_candidate_type,false_candidate

#得到候选实体的正例(多分类）和负例,多分类问题使用本函数
def get_real_entity_mul_calss_fb_N(B_index,E_index,B_label,E_label):


    total_candidate=[]
    true_candidate=[]
    true_candidate_type=[]
    false_candidate=[]

    total_candidate = get_fb_match_N(B_index, E_index)#bi-greedy

    for candidate,B_label_node,E_label_node in zip(total_candidate,B_label,E_label):
        true_temp=[]
        true_type_temp=[]
        for candidate_node in candidate:
            a=B_label_node[candidate_node[0]]
            b=E_label_node[candidate_node[1]]
            state,type=get_state_type(B_label_node[candidate_node[0]],E_label_node[candidate_node[1]])
            if(state):
                true_temp.append(candidate_node)
                true_type_temp.append(type)
        true_candidate.append(true_temp)
        true_candidate_type.append(true_type_temp)
    false_candidate=list_difference(total_candidate,true_candidate)
    return total_candidate,true_candidate,true_candidate_type,false_candidate

def get_real_entity_mul_calss_fb_N_union(B_index,E_index,B_label,E_label):

    # total_candidate, all_true_candidate, all_true_candidate_type, _=get_real_entity_mul_calss(B_index,E_index,B_label,E_label)
    _, _, _, fb_false_candidate=get_real_entity_mul_calss_fb_N(B_index,E_index,B_label,E_label)
    #total_candidate在下面用不到，随便给一个就行
    return _, _, _,fb_false_candidate

#得到所有候选实体
def find_candidate_bert(B,E):
    '''
    :param BIO: 标签，用不到
    :param B:
    :param E:
    :return: 开始和结束标签的列表
    '''

    B_index=[]
    E_index=[]
    for node in B:
        temp_B=[]
        for x in range(len(node)):
            if("B" in str(node[x])):
                temp_B.append(x)
        B_index.append(temp_B)
    for node in E:
        temp_E=[]
        for x in range(len(node)):
            if("B" in str(node[x])):
                temp_E.append(x)
        E_index.append(temp_E)
    return B_index,E_index

#得到所有候选实体
def find_candidate(BIO,B,E):
    '''
    :param BIO: 标签，用不到
    :param B:
    :param E:
    :return: 开始和结束标签的列表
    '''

    B_index=[]
    E_index=[]
    for node in B:
        temp_B=[]
        for x in range(len(node)):
            if("B" in str(node[x])):
                temp_B.append(x)
        B_index.append(temp_B)
    for node in E:
        temp_E=[]
        for x in range(len(node)):
            if("B" in str(node[x])):
                temp_E.append(x)
        E_index.append(temp_E)
    return B_index,E_index

#得到所有候选实体
def find_candidate_bert(B,E):
    '''
    :param BIO: 标签，用不到
    :param B:
    :param E:
    :return: 开始和结束标签的列表
    '''

    B_index=[]
    E_index=[]
    for node in B:
        temp_B=[]
        for x in range(len(node)):
            if("B" in str(node[x])):
                temp_B.append(x)
        B_index.append(temp_B)
    for node in E:
        temp_E=[]
        for x in range(len(node)):
            if("B" in str(node[x])):
                temp_E.append(x)
        E_index.append(temp_E)
    return B_index,E_index

#从文本文件中获取字符序列和对应的标签1
def get_label_BE(BIO_path,B_path,E_path):
    BIO_file=open(BIO_path,"r",encoding="utf-8")
    B_file = open(B_path, "r", encoding="utf-8")
    E_file = open(E_path, "r", encoding="utf-8")

    BIO_line,B_line,E_line=BIO_file.readlines(),B_file.readlines(),E_file.readlines()

    BIO_seq,BIO_label,B_label,E_label=[],[],[],[]

    BIO_seq_temp=[]
    BIO_label_temp=[]
    B_label_temp = []
    E_label_temp = []

    for BIO_ele,B_ele,E_ele in zip(BIO_line,B_line,E_line):
        if(BIO_ele[0]!="\n"):
            BIO_list=BIO_ele.strip("\n").split("\t")
            BIO_seq_temp.append(BIO_list[0])
            BIO_label_temp.append(BIO_list[1])

            B_list = B_ele.strip("\n").split("\t")
            B_label_temp.append(B_list[1])

            E_list = E_ele.strip("\n").split("\t")
            E_label_temp.append(E_list[1])
        else:
            BIO_seq_temp,BIO_label_temp,B_label_temp,E_label_temp=pad_label(BIO_seq_temp,BIO_label_temp,B_label_temp,E_label_temp)
            BIO_seq.append(BIO_seq_temp)
            BIO_label.append(BIO_label_temp)
            B_label.append(B_label_temp)
            E_label.append(E_label_temp)
            BIO_seq_temp = []
            BIO_label_temp = []
            B_label_temp = []
            E_label_temp = []

    return BIO_seq,BIO_label,B_label,E_label

def index_type(nested_string):
    index, type = nested_string.split(",")
    index = index.split("-")[1:]
    type = type.split("-")
    return index,type


def calculate_entity(B_path,E_path):

    B_line=open(B_path,"r",encoding="utf-8").read()
    E_line=open(E_path,"r",encoding="utf-8").read()

    word_seq = [[seq_word.split("\t")[0] for seq_word in seq_sentence.strip().split("\n")] for seq_sentence in B_line.strip().split("\n" + "\n")]

    B_label=[[B_word.split("\t")[-1] for B_word in B_sentence.strip().split("\n")] for B_sentence in B_line.strip().split("\n"+"\n")]
    E_label = [[E_word.split("\t")[-1] for E_word in E_sentence.strip().split("\n")] for E_sentence in E_line.strip().split("\n" + "\n")]

    result_index=[]
    result_type=[]

    for B_node,E_node in zip(B_label,E_label):
        sentence_index_temp=[]
        sentence_type_temp = []
        a=B_node
        b=E_node
        B_index_single_type = []
        E_index_single_type = []
        for i,nested_B in enumerate(B_node):
            if("B" in nested_B):
                corrent_B_index,corrent_B_type=index_type(nested_B)
                for b_index,b_type in zip(corrent_B_index,corrent_B_type):
                    B_index_single_type.append([i,int(b_index),b_type])
        for j,nested_E in enumerate(E_node):
            if ("B" in nested_E):
                corrent_E_index, corrent_E_type = index_type(nested_E)
                for e_index,e_type in zip(corrent_E_index,corrent_E_type):
                    E_index_single_type.append([j,int(e_index),e_type])

        for B_index_single_type_node in B_index_single_type:
            for E_index_single_type_node in E_index_single_type:
                if(B_index_single_type_node[1]==E_index_single_type_node[1]):
                    sentence_index_temp.append([B_index_single_type_node[0],E_index_single_type_node[0]])
                    assert B_index_single_type_node[2]==E_index_single_type_node[2] ,"对应为值类型不匹配"
                    sentence_type_temp.append(B_index_single_type_node[2])
                    continue
        result_index.append(sentence_index_temp)
        result_type.append(sentence_type_temp)

    return result_index,result_type

#从文本文件中获取字符序列和对应的标签1
def get_label_test(BIO_path,B_path,E_path):
    BIO_file=open(BIO_path,"r",encoding="utf-8")
    B_file = open(B_path, "r", encoding="utf-8")
    E_file = open(E_path, "r", encoding="utf-8")

    BIO_line,B_line,E_line=BIO_file.readlines(),B_file.readlines(),E_file.readlines()

    BIO_seq,BIO_label,B_label,E_label=[],[],[],[]

    BIO_seq_temp=[]
    BIO_label_temp=[]
    B_label_temp = []
    E_label_temp = []

    for BIO_ele,B_ele,E_ele in zip(BIO_line,B_line,E_line):
        if(BIO_ele[0]!="\n"):
            BIO_list=BIO_ele.strip("\n").split("\t")
            BIO_seq_temp.append(BIO_list[0])
            BIO_label_temp.append(BIO_list[1])

            B_list = B_ele.strip("\n").split("\t")
            B_label_temp.append(B_list[1])

            E_list = E_ele.strip("\n").split("\t")
            E_label_temp.append(E_list[1])
        else:
            BIO_seq.append(BIO_seq_temp)
            BIO_label.append(BIO_label_temp)
            B_label.append(B_label_temp)
            E_label.append(E_label_temp)
            BIO_seq_temp = []
            BIO_label_temp = []
            B_label_temp = []
            E_label_temp = []

    return BIO_seq,BIO_label,B_label,E_label

def correct_false(false):
    a=FLAGS.correct_false_a
    new_false=[]
    for sentence_node in false:
        false_temp=[]
        node_len = len(sentence_node)
        if(node_len==0):
            new_false.append([])
            continue
        else:
            new_len=node_len-int(node_len*a)
            new_false.append(sentence_node[:new_len])
    return new_false

def transfor_padding(train_left_list, train_entity_left, train_entity_right, train_right_list):

    new_left = np.asanyarray(pad_sequences(train_left_list,FLAGS.read_max_seq_langth,padding="pre",truncating="post"))
    new_entity_left = np.asanyarray(pad_sequences(train_entity_left, FLAGS.read_max_seq_langth, padding="pre", truncating="post"))
    new_entity_right = np.asanyarray(pad_sequences(train_entity_right, FLAGS.read_max_seq_langth, padding="pre", truncating="post"))
    new_right = np.asanyarray(pad_sequences(train_right_list, FLAGS.read_max_seq_langth, padding="pre", truncating="post"))

    return new_left,new_entity_left,new_entity_right,new_right


#hub函数，调用本方法集的入口2
def hub_main(B_path,E_path,BIO_label,B_label,E_label,BIO_charseq):
    B_index, E_index = find_candidate(BIO_label, B_label, E_label)#11
    _,true_entity, entity_type, false_entity = get_real_entity_mul_calss(B_index, E_index, B_label, E_label)#训练，验证方式不同
    # _, _, _, false_entity = get_real_entity_mul_calss_fb_N_union(B_index, E_index, B_label, E_label)#前后匹配各一生成，在加上所有正例，
    result_index,result_type=calculate_entity(B_path,E_path)
    return _, result_index,result_type, false_entity


def E2E_main_train_dev_bert_emb(bert_word_emb,BIO_path,B_path,E_path,name,logger):

    BIO_seq, BIO_label, B_label, E_label = get_label_BE(BIO_path, B_path, E_path)  # 给三个原值百分百准确率文件得出字符序列，三类标签
    real_total_entity, real_true_entity, real_entity_type, real_false_entity = hub_main(B_path,E_path,BIO_label, B_label, E_label,BIO_seq)

    entity_input=get_entity_input_mul_class_bert_emb(bert_word_emb,real_total_entity,real_true_entity,real_entity_type,real_false_entity)

    left_list, entity_left, entity_right, right_list, label=generate_input_lab_emb_bert(entity_input)

    num_display_train_dev(label,name,logger)

    return left_list, entity_left, entity_right, right_list,label


#从边界预测结果中得到的候选实体，按照和真实实体对比得出的结果，生成数据的生成
def generate_test_input(true_entity_e2e,true_entity_type,false_entity_e2e,BIO_seq_e2e):

    left_input=[]
    entity_left_input=[]
    entity_right_input=[]
    right_input=[]

    total_type=[]

    for entity_node,type_node,seq_node in zip(true_entity_e2e,true_entity_type,BIO_seq_e2e):
        if(len(entity_node)==0):
            continue
        else:
            for entity,type in zip(entity_node ,type_node):
                a = entity[0]
                b = entity[-1]

                left_input.append(seq_node[:a+1])
                entity_left_input.append(seq_node[a:b+1])

                entity_right_temp=seq_node[a:b+1]
                entity_right_temp.reverse()
                entity_right_input.append(entity_right_temp)

                right_temp = seq_node[b:]
                right_temp.reverse()
                right_input.append(right_temp)

                total_type.append(type)

    for false_entity_node ,false_seq_node in zip(false_entity_e2e,BIO_seq_e2e):
        if(len(false_entity_node)==0):
            continue
        else:
            for false_entity in false_entity_node:
                c=false_entity[0]
                d=false_entity[-1]

                left_input.append(false_seq_node[:a + 1])
                entity_left_input.append(false_seq_node[a:b + 1])

                false_entity_right_temp = false_seq_node[a:b + 1]
                false_entity_right_temp.reverse()
                entity_right_input.append(false_entity_right_temp)

                false_right_temp = false_seq_node[b:]
                false_right_temp.reverse()
                right_input.append(false_right_temp)

                total_type.append("NEG")



    # new_left_input=pad_sequences(left_input,FLAGS.seq_langth)
    # new_entity_left_input=pad_sequences(entity_left_input,FLAGS.seq_langth)
    # new_entity_right_input=pad_sequences(entity_right_input,FLAGS.seq_langth)
    # new_right_input=pad_sequences(right_input,FLAGS.seq_langth)

    new_left_input = left_input
    new_entity_left_input = entity_left_input
    new_entity_right_input = entity_right_input
    new_right_input =right_input

    return new_left_input,new_entity_left_input,new_entity_right_input,new_right_input,total_type


#文件内容抽出来返回列表
def file_to_list(data_path):
    '''
    :param data_path: 输入文件
    :return:
    '''

    result_list=[]
    temp=[]
    data_file = open(data_path, "r", encoding='utf-8')

    data_line = data_file.readlines()
    num_order = 0
    for node in data_line:
        node_list=node.strip("\n").split("\t")
        if(node_list[0]!=""):
            temp.append(node_list)
        else:
            result_list.append(temp)
            temp=[]
    return result_list

#文件内容抽出来返回列表
def file_to_list_padding(data_path):
    '''
    :param data_path: 输入文件
    :return:
    '''
    length=FLAGS.read_max_seq_langth
    result_list=[]
    temp=[]
    data_file = open(data_path, "r", encoding='utf-8')

    data_line = data_file.readlines()
    num_order = 0
    for node in data_line:
        node_list=node.strip("\n").split("\t")
        if(node_list[0]!=""):
            temp.append(node_list)
        else:
            temp=pad_string(temp,length)
            result_list.append(temp)
            temp=[]
    return result_list

#仅仅是得到所有的候选实体，材料来源是嵌套模型crf的预测结果
def get_real_entity_mul_calss_E2E(B_index,E_index,B_label,E_label):

    # pickle.dump([B_index,E_index],open("BE_index","wb"))

    total_candidate=[]
    for B_index_node,E_index_node in zip(B_index,E_index):
        total_temp=[]
        for x in E_index_node[::-1]:
            for y in B_index_node[::-1]:
                if(y<=x):
                    total_temp.append([y,x])
        total_candidate.append(total_temp)
    return total_candidate




def get_label(B_list,E_list,start_N,end_N):
    '''
    :param E_index: 结束标签的index列表，以句为单位
    :param B_list: 开始标签列表
    :return: 每句中的候选实体index
    '''

    start_N=end_N=2

    random_label_B=[[w[-1] for w in node] for node in B_list]
    random_label_E = [[w[-1] for w in node] for node in E_list]

    B_weight=[[w[1] for w in node] for node in B_list]
    E_weight = [[w[1] for w in node] for node in E_list]

    for B_node,B_label_index in zip(B_weight,random_label_B):
        B_pre_label=get_line(B_node,0.5)
        B_new_label=diff_list(B_node,B_pre_label,start_N)+B_pre_label
        for x in range(len(B_node)):
            if(B_node[x] in B_new_label):
                B_label_index[x]="B"
            else:B_label_index[x]="O"

    for E_node,E_label_index in zip(E_weight,random_label_E):
        E_pre_label=get_line(E_node,0.5)
        E_new_label=diff_list(E_node,E_pre_label,end_N)+E_pre_label
        for x in range(len(E_node)):
            if(E_node[x] in E_new_label):
                E_label_index[x]="B"
            else:E_label_index[x]="O"

    return random_label_B,random_label_E

#得到去除补零的序列
def return_str_list(data_list):
    str_return=""
    for data_node in data_list:
        if (data_node != '0'):
            str_return += str(data_node)
    return str_return

def return_str_list_test(data_list):
    str_return=""
    for data_node in data_list:
        if (data_node != 0):
            str_return += str(data_node)
    return str_return

#配合性能测试，使用uuid方式表示每一实体
def generate_train_dev_uuid(true_entity,true_entity_type,BIO_seq_e2e):
    left_input=[]
    entity_left_input=[]
    entity_right_input=[]
    right_input=[]

    total_type=[]

    for entity_node ,type_node,seq_node in zip(true_entity,true_entity_type,BIO_seq_e2e):
        if(len(entity_node)==0):
            continue
        else:
            for entity,type in zip(entity_node ,type_node):
                a = entity[0]
                b = entity[-1]

                left_input.append(seq_node[:a+1])
                entity_left_input.append(seq_node[a:b+1])

                entity_right_temp=seq_node[a:b+1]
                entity_right_temp.reverse()
                entity_right_input.append(entity_right_temp)

                right_temp = seq_node[b:]
                right_temp.reverse()
                right_input.append(right_temp)

                total_type.append(type)

    # new_left_input=pad_sequences(left_input,FLAGS.seq_langth)
    # new_entity_left_input=pad_sequences(entity_left_input,FLAGS.seq_langth)
    # new_entity_right_input=pad_sequences(entity_right_input,FLAGS.seq_langth)
    # new_right_input=pad_sequences(right_input,FLAGS.seq_langth)
    #
    # return new_left_input,new_entity_left_input,new_entity_right_input,new_right_input,total_type

    return left_input,entity_left_input,entity_right_input,right_input,total_type


#把每个候选实体使用uuid做成全空间唯一表示
def uuid_laebl_new(left_list_list,entity_left_list,entity_right_list,right_list_list,label_list):

    candidate_dict={}

    for left_list_node,entity_left_node,entity_right_node,right_list_node,label_node in zip(left_list_list,entity_left_list,entity_right_list,right_list_list,label_list):
        label = label_node
        left_list_string=return_str_list(left_list_node)
        entity_left_string=return_str_list(entity_left_node)
        entity_right_string = return_str_list(entity_right_node)
        right_list_string = return_str_list(right_list_node)


        uuid_left = uuid.uuid3(uuid.NAMESPACE_DNS, left_list_string)
        uuid_entity_left=uuid.uuid3(uuid.NAMESPACE_DNS, entity_left_string)
        uuid_entity_right = uuid.uuid3(uuid.NAMESPACE_DNS, entity_right_string)
        uuid_right = uuid.uuid3(uuid.NAMESPACE_DNS, right_list_string)
        uuid_str=str(uuid_left)+"*"+str(uuid_entity_left)+"*"+str(uuid_right)+"*"+str(uuid_entity_right)
        candidate_dict[uuid_str]=label
    temp=len(candidate_dict)

    return candidate_dict

def get_node_index(B_index,corrent_node):
    new=[]
    for node in B_index:
        if(node<=corrent_node):
            new.append(node)
    return sorted(new,reverse=True)

def get_match_N(B_index,E_index):
    define_N = FLAGS.define_N
    total_candidate=[]
    for B_index_node, E_index_node in zip(B_index, E_index):
        total_temp = []
        for x in E_index_node:
            distance=get_node_index(B_index_node,x)
            if(len(distance)==0):
                continue
            else:
                for NB in distance[:define_N]:
                    total_temp.append([NB,x])
        total_candidate.append(total_temp)
    return  total_candidate

def get_node_index_forward(B_index,corrent_node):
    new=[]
    for node in B_index:
        if(node>=corrent_node):
            new.append(node)
    return sorted(new,reverse=False)

def get_match_backqard_N(B_index,E_index):
    define_N = FLAGS.define_fb_N
    total_candidate=[]
    for B_index_node, E_index_node in zip(B_index, E_index):
        total_temp = []
        for x in E_index_node:
            distance=get_node_index(B_index_node,x)
            if(len(distance)==0):
                continue
            else:
                for NB in distance[:define_N]:
                    total_temp.append([NB,x])
        total_candidate.append(total_temp)
    return  total_candidate

def get_match_forward_N(B_index,E_index):
    define_N = FLAGS.define_fb_N
    total_candidate=[]
    for B_index_node, E_index_node in zip(B_index, E_index):
        total_temp = []
        for x in B_index_node:
            distance=get_node_index_forward(E_index_node,x)
            if(len(distance)==0):
                continue
            else:
                for NB in distance[:define_N]:
                    total_temp.append([x,NB])
        total_candidate.append(total_temp)
    return  total_candidate

def union(total_candidate_forward,total_candidate_backward):
    final_candidate=[]

    for forward_node,backward_node in zip(total_candidate_forward,total_candidate_backward):
        candidate_temp=[]
        for forward in forward_node:
            if(forward not in backward_node):
                candidate_temp.append(forward)
        candidate_temp=candidate_temp+backward_node
        final_candidate.append(candidate_temp)
    print()
    return final_candidate

def get_fb_match_N(B_index,E_index):
    #前向，后向取并集
    total_candidate_forward = get_match_backqard_N(B_index, E_index)#从结束，向前匹配一个

    total_candidate_backward = get_match_forward_N(B_index, E_index)#从开始，向后匹配一个
    # total_candidate_backward = total_candidate_forward

    final_candidate = union(total_candidate_forward, total_candidate_backward)#两次匹配求并集

    return final_candidate


#从预测候选实体和真正的候选实体的对比中得到预测候选实体中的真正实体及其类型
def match_candidate(true_candidate,true_entity_type,total_candidate_pre):
    result=[]
    entity_type=[]
    true_temp=[]
    type_temp=[]

    for a_ndoe,b_node in zip(true_candidate,total_candidate_pre):
        for node_b in b_node:
            for node_a in a_ndoe:
                if(operator.eq(node_b,node_a)):
                    true_temp.append(node_b)
        result.append(true_temp)
        true_temp=[]

    for result_node,a_node,type_node in zip(result,true_candidate,true_entity_type):
        for result_node_2  in  result_node :
            index_a=a_node.index(result_node_2)
            temp=type_node[index_a]
            type_temp.append(temp)
        entity_type.append(type_temp)
        type_temp=[]
    true_candidate_pre=result

    false_candidate_pre=list_difference(total_candidate_pre,true_candidate_pre)
    return true_candidate_pre,false_candidate_pre,entity_type

#以按照crf得出的概率，取大于0.5的为B，除此之晚，取余下的最大N个为B，进行测试集候选实体的生成
def hub_generate_candidate_new_assbmle(B_path,E_path):

    start_N = FLAGS.start_N  # 组装策略中选取候选开始边界的权重
    end_N = FLAGS.end_N    #组装策略中选取候选开始边界的权重
    B_list=file_to_list(B_path)#从文件中得到权重
    E_list=file_to_list(E_path)

    B_label,E_label=get_label(B_list,E_list,start_N,end_N)#按照前述策略，重新写入标签

    B_index, E_index = find_candidate([], B_label, E_label)#以下就和训练集、验证集的候选实体生成式一样的
    total_entity = get_real_entity_mul_calss_E2E(B_index, E_index, B_label, E_label)

    return total_entity

#以按照crf得出的概率，取大于0.5的为B，除此之晚，取余下的最大N个为B，进行测试集候选实体的生成
def hub_generate_candidate_new_assbmle_padding(B_path,E_path):

    start_N = FLAGS.start_N  # 组装策略中选取候选开始边界的权重
    end_N = FLAGS.end_N    #组装策略中选取候选开始边界的权重
    B_list=file_to_list_padding(B_path)#从文件中得到权重
    E_list=file_to_list_padding(E_path)

    B_label,E_label=get_label(B_list,E_list,start_N,end_N)#按照前述策略，重新写入标签

    B_index, E_index = find_candidate([], B_label, E_label)#以下就和训练集、验证集的候选实体生成式一样的
    total_entity = get_real_entity_mul_calss_E2E(B_index, E_index, B_label, E_label)

    return total_entity

def num_display_train_dev(label_order,name,logger):
    class_count=get_num_class(label_order)#
    true, false = count_test_example(class_count)
    logger.info("从ACE文件中得出的用于"+name+"的实体集:")
    logger.info("正例数目：{0}      负例数目：{1}".format(str(true),str(false)))
    logger.info("详细情况如下：")
    logger.info(class_count)

def num_display_train_dev_array(label_order,name,logger):
    class_count=get_num_class_array(label_order)#
    true, false = count_test_example(class_count)
    logger.info("从ACE文件中得出的用于"+name+"的实体集:")
    logger.info("正例数目：{0}      负例数目：{1}".format(str(true),str(false)))
    logger.info("详细情况如下：")
    logger.info(class_count)

def num_display_true(class_count_dict,logger):
    true, false = count_test_example(class_count_dict)
    logger.info("从ACE文件中得出的用于性能评价的实体集:")
    logger.info("正例数目：{0}".format(str(true)))
    logger.info("详细情况如下：")
    logger.info(class_count_dict)
    print()
    logger.info("****************************************************")
    print()

def num_display(label_order,logger):
    class_count=get_num_class_array(label_order)#
    true, false = count_test_example(class_count)
    logger.info("从BERT边界识别文件中得出的用于测试的候选实体集数目分布:")
    logger.info("正例数目：{0}      负例数目：{1}".format(str(true),str(false)))
    logger.info("详细情况如下：")
    logger.info(class_count)


#从文本文件中获取字符序列和对应的标签1
def get_label_bert(B_path,E_path):
    B_file = open(B_path, "r", encoding="utf-8")
    E_file = open(E_path, "r", encoding="utf-8")

    B_line,E_line=B_file.readlines(),E_file.readlines()

    B_label,E_label=[],[]


    B_label_temp = []
    E_label_temp = []

    for B_ele,E_ele in zip(B_line,E_line):
        if(B_ele[0]!="\n"):

            B_list = B_ele.strip("\n").split("\t")
            B_label_temp.append(B_list[1])

            E_list = E_ele.strip("\n").split("\t")
            E_label_temp.append(E_list[1])

        else:
            B_label.append(B_label_temp)
            E_label.append(E_label_temp)
            BIO_seq_temp = []
            BIO_label_temp = []
            B_label_temp = []
            E_label_temp = []

    return B_label,E_label

#hub函数，调用本方法集的入口3
def hub_main_bert(B_path,E_path):
    B_label, E_label=get_label_bert(B_path,E_path)
    B_index, E_index = find_candidate_bert(B_label, E_label)
    total_entity= get_real_entity_mul_calss_E2E(B_index, E_index, B_label, E_label)#这个是原来的全匹配
    # total_entity = get_match_N(B_index, E_index)#greedy匹配
    # total_entity = get_fb_match_N(B_index, E_index)#bi-greedy-N
    return total_entity

def C_method_main_test_bert_old(BIO_seq_e2e,B_path,E_path,mul_BIO_path,mul_B_path,mul_E_path,logger):

    total_entity_e2e=hub_main_bert(B_path,E_path)

    BIO_seq, BIO_label, B_label, E_label = get_label_test(mul_BIO_path, mul_B_path,mul_E_path)  # 百分百真实真是边界信息，可以生成百分百正确的候选实体
    real_total_entity, real_true_entity, real_entity_type,real_false_entity= hub_main(BIO_label, B_label, E_label,BIO_seq)#测试集全集中的实体index和type


    true_entity_e2e, false_entity_e2e, entity_type_e2e = match_candidate(real_true_entity, real_entity_type,total_entity_e2e)#从边界预测结果中得到的候选实体，按照和真实实体对比得出的结果,e2e表示根据边界识别结果得到的首选实体中的true实体和false实体

    entity_input = get_entity_input_mul_class_tf(BIO_seq_e2e, total_entity_e2e, true_entity_e2e, entity_type_e2e,false_entity_e2e)
    left_list, entity_left, entity_right, right_list,label = generate_input_old(entity_input)

    num_display(label,logger)

    return left_list, entity_left, entity_right, right_list,label

def C_method_main_test_bert_emb(BIO_seq_e2e,B_path,E_path,mul_BIO_path,mul_B_path,mul_E_path,logger):

    total_entity_e2e=hub_main_bert(B_path,E_path)

    BIO_seq, BIO_label, B_label, E_label = get_label_test(mul_BIO_path, mul_B_path,mul_E_path)  # 百分百真实真是边界信息，可以生成百分百正确的候选实体
    real_total_entity, real_true_entity, real_entity_type,real_false_entity= hub_main(mul_B_path,mul_E_path,BIO_label, B_label, E_label,BIO_seq)#测试集全集中的实体index和type


    true_entity_e2e, false_entity_e2e, entity_type_e2e = match_candidate(real_true_entity, real_entity_type,total_entity_e2e)#从边界预测结果中得到的候选实体，按照和真实实体对比得出的结果,e2e表示根据边界识别结果得到的首选实体中的true实体和false实体

    entity_input = get_entity_input_mul_class_bert_emb(BIO_seq_e2e, total_entity_e2e, true_entity_e2e, entity_type_e2e,false_entity_e2e)
    left_list, entity_left, entity_right, right_list, label = generate_input_lab_emb_bert(entity_input)

    num_display(label,logger)

    return left_list, entity_left, entity_right, right_list, label

def C_method_main_bert_emb_a(BIO_seq_e2e,array_emb,B_path,E_path,mul_BIO_path,mul_B_path,mul_E_path,name,logger):

    # total_entity_e2e=hub_main_bert(B_path,E_path)
    total_entity_e2e=a_main(BIO_seq_e2e,B_path,E_path)

    BIO_seq, BIO_label, B_label, E_label = get_label_test(mul_BIO_path, mul_B_path,mul_E_path)  # 百分百真实真是边界信息，可以生成百分百正确的候选实体
    real_total_entity, real_true_entity, real_entity_type,real_false_entity= hub_main(mul_B_path,mul_E_path,BIO_label, B_label, E_label,BIO_seq)#测试集全集中的实体index和type

    true_entity_e2e, false_entity_e2e, entity_type_e2e = match_candidate(real_true_entity, real_entity_type,total_entity_e2e)#从边界预测结果中得到的候选实体，按照和真实实体对比得出的结果,e2e表示根据边界识别结果得到的首选实体中的true实体和false实体

    # p_n_count(false_entity_e2e, entity_type_e2e,name)

    entity_input = get_entity_input_mul_class_bert_emb(array_emb, total_entity_e2e, true_entity_e2e, entity_type_e2e,false_entity_e2e)
    left_list, entity_left, entity_right, right_list, label = generate_input_lab_emb_bert(entity_input)

    num_display_train_dev(label, name, logger)

    return left_list, entity_left, entity_right, right_list, label

def C_method_main_bert_emb_a_num(BIO_seq_e2e,array_emb,B_path,E_path,mul_BIO_path,mul_B_path,mul_E_path,name,logger):

    total_entity_e2e=a_main(BIO_seq_e2e,B_path,E_path)

    BIO_seq, BIO_label, B_label, E_label = get_label_test(mul_BIO_path, mul_B_path,mul_E_path)  # 百分百真实真是边界信息，可以生成百分百正确的候选实体
    real_total_entity, real_true_entity, real_entity_type,real_false_entity= hub_main(mul_B_path,mul_E_path,BIO_label, B_label, E_label,BIO_seq)#测试集全集中的实体index和type

    true_entity_e2e, false_entity_e2e, entity_type_e2e = match_candidate(real_true_entity, real_entity_type,total_entity_e2e)#从边界预测结果中得到的候选实体，按照和真实实体对比得出的结果,e2e表示根据边界识别结果得到的首选实体中的true实体和false实体

    p_n_count(false_entity_e2e, entity_type_e2e,name)

def C_method_main_test_bert(BIO_seq_e2e,B_path,E_path,mul_BIO_path,mul_B_path,mul_E_path,logger):

    total_entity_e2e=hub_main_bert(B_path,E_path)

    BIO_seq, BIO_label, B_label, E_label = get_label_test(mul_BIO_path, mul_B_path,mul_E_path)  # 百分百真实真是边界信息，可以生成百分百正确的候选实体
    real_total_entity, real_true_entity, real_entity_type,real_false_entity= hub_main(mul_B_path,mul_E_path,BIO_label, B_label, E_label,BIO_seq)#测试集全集中的实体index和type


    true_entity_e2e, false_entity_e2e, entity_type_e2e = match_candidate(real_true_entity, real_entity_type,total_entity_e2e)#从边界预测结果中得到的候选实体，按照和真实实体对比得出的结果,e2e表示根据边界识别结果得到的首选实体中的true实体和false实体

    entity_input = get_entity_input_mul_class(BIO_seq_e2e, total_entity_e2e, true_entity_e2e, entity_type_e2e,false_entity_e2e)
    left_seq, left_list, entity_left, entity_right, right_list, right_seq, label = generate_input_lab(entity_input)

    num_display(label,logger)

    return left_seq, left_list, entity_left, entity_right, right_list, right_seq, label

#获得可以用来性能测试的数据
def C_method_main_test_new_assemble(BIO_seq_e2e,weight_B_path,weight_E_path,mul_BIO_path,mul_B_path,mul_E_path,logger):

    total_entity_e2e=hub_generate_candidate_new_assbmle(weight_B_path,weight_E_path)

    BIO_seq, BIO_label, B_label, E_label = get_label_test(mul_BIO_path, mul_B_path,mul_E_path)  # 百分百真实真是边界信息，可以生成百分百正确的候选实体
    real_total_entity, real_true_entity, real_entity_type,real_false_entity= hub_main(BIO_label, B_label, E_label,BIO_seq)#测试集全集中的实体index和type


    #以下四行是为了使用测试集全集进行最后的性能评价，使用uuid表示每个实体
    test_left_list, test_entity_left, test_entity_right, test_right_list, label = generate_train_dev_uuid(real_true_entity,real_entity_type, BIO_seq)
    real_class_count = get_num_class(label)#测试集中的实体数目
    real_uuid_candidate_label = uuid_laebl_new(test_left_list, test_entity_left, test_entity_right, test_right_list,label)
    class_count_dict = get_num_class_dict(real_uuid_candidate_label)#uuid可以独立唯一表示的实体数目
    num_display_true(class_count_dict,logger)


    true_entity_e2e, false_entity_e2e, entity_type_e2e = match_candidate(real_true_entity, real_entity_type,total_entity_e2e)#从边界预测结果中得到的候选实体，按照和真实实体对比得出的结果,e2e表示根据边界识别结果得到的首选实体中的true实体和false实体

    left_list, entity_left, entity_right, right_list, label_order=generate_test_input(true_entity_e2e, entity_type_e2e,false_entity_e2e, BIO_seq_e2e)#测试数据的生成

    num_display(label_order,logger)


    return left_list, entity_left, entity_right, right_list,label_order,real_uuid_candidate_label

#GUID方式计算性能
def performance_uuid_binary_new(y_pred, test_left_list, test_entity_left, test_entity_right, test_right_list, real_uuid):
    y_pred_index = y_pred.reshape(-1, 1)  # 预测的结果集

    true_index = 0
    false_index = 1

    positive_true_test_num = 0  # 测试集中的正例
    positive_pred_num = 0  # 模型总共预测了多少个正例
    positive_true_pre_num = 0  # 模型预测的正例中有多少个是正确的


    class_type = "VEH-LOC-WEA-GPE-PER-ORG-FAC"

    class_dict_num = get_num_class_dict(real_uuid)

    for key, value in class_dict_num.items():
        if (key in class_type):
            positive_true_test_num += value

    for y_pred_node, left_list_node, entity_left_node, entity_right_node, right_list_node in zip(y_pred, test_left_list,test_entity_left,test_entity_right,test_right_list):
        uuid_check = uuid_laebl_percormance_new(left_list_node, entity_left_node, entity_right_node, right_list_node)
        # uuid_check = uuid_laebl_percormance(left_list_node, entity_left_node, entity_right_node, right_list_node)
        try:
            real_label = real_uuid[uuid_check]
        except:
            real_label="NEG"
        if ((real_label in class_type) and (y_pred_node == true_index)):
            positive_true_pre_num += 1
        if (y_pred_node == true_index):
            positive_pred_num += 1

    print("-------------------------正例性能---------------------------------")
    print("测试集中的正例: " + str(positive_true_test_num))
    print("预测结果中的正例:" + str(positive_pred_num))
    print("预测为正例的结果中真正的正例:" + str(positive_true_pre_num))
    P = 0 if positive_pred_num == 0 else 100. * positive_true_pre_num / positive_pred_num
    R = 0 if positive_true_pre_num == 0 else 100. * positive_true_pre_num / positive_true_test_num
    F = 0 if P + R == 0 else 2 * P * R / (P + R)
    print("Precision: %.2f" % (P), "%")
    print("Recall: %.2f" % (R), "%")
    print("F1: %.2f" % (F), "%")

#获得可以用来性能测试的数据
def C_method_main_test_new_assemble_padding(BIO_seq_e2e,weight_B_path,weight_E_path,mul_BIO_path,mul_B_path,mul_E_path,logger):

    total_entity_e2e=hub_generate_candidate_new_assbmle_padding(weight_B_path,weight_E_path)

    BIO_seq, BIO_label, B_label, E_label = get_label_test(mul_BIO_path, mul_B_path,mul_E_path)  # 百分百真实真是边界信息，可以生成百分百正确的候选实体
    real_total_entity, real_true_entity, real_entity_type,real_false_entity= hub_main(BIO_label, B_label, E_label,BIO_seq)#测试集全集中的实体index和type


    #以下四行是为了使用测试集全集进行最后的性能评价，使用uuid表示每个实体
    test_left_list, test_entity_left, test_entity_right, test_right_list, label = generate_train_dev_uuid(real_true_entity,real_entity_type, BIO_seq)
    real_class_count = get_num_class(label)#测试集中的实体数目
    real_uuid_candidate_label = uuid_laebl_new(test_left_list, test_entity_left, test_entity_right, test_right_list,label)
    class_count_dict = get_num_class_dict(real_uuid_candidate_label)#uuid可以独立唯一表示的实体数目
    num_display_true(class_count_dict,logger)


    true_entity_e2e, false_entity_e2e, entity_type_e2e = match_candidate(real_true_entity, real_entity_type,total_entity_e2e)#从边界预测结果中得到的候选实体，按照和真实实体对比得出的结果,e2e表示根据边界识别结果得到的首选实体中的true实体和false实体

    left_list, entity_left, entity_right, right_list, label_order=generate_test_input(true_entity_e2e, entity_type_e2e,false_entity_e2e, BIO_seq_e2e)#测试数据的生成

    num_display(label_order,logger)

    # uuid_candidate_label=uuid_laebl(left_list,entity_left,entity_right,right_list,label)
    # class_count_dict=get_num_class_dict(uuid_candidate_label)
    # print("候选实体测试集中uuid实体情况：")
    # print(class_count_dict)

    return left_list, entity_left, entity_right, right_list,label_order,real_uuid_candidate_label








