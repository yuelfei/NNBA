'''
@文件名称: aa.py
@作者: 武乐飞
@创建时间: 2019/1/1 - 19:44
@描述: 
'''

import pickle

test_left_list, test_entity_left, test_entity_right, test_right_list,id_to_char=pickle.load(open("transfor_pre_id","rb"))

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


per_test_left_list, per_test_entity_left, per_test_entity_right, per_test_right_list=transfor_pre_test(test_left_list, test_entity_left, test_entity_right, test_right_list,id_to_char)

print()



