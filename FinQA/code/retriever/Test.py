#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script
"""
from tqdm import tqdm
import json
import os
from datetime import datetime
import time
import logging
from utils import *
from config import parameters as conf
from torch import nn
import torch
import torch.optim as optim


from Model import Bert_model

if conf.pretrained_model == "bert":
    from transformers import BertTokenizer
    from transformers import BertConfig
    tokenizer = BertTokenizer.from_pretrained(conf.model_size)
    model_config = BertConfig.from_pretrained(conf.model_size)

elif conf.pretrained_model == "roberta":
    from transformers import RobertaTokenizer
    from transformers import RobertaConfig
    tokenizer = RobertaTokenizer.from_pretrained(conf.model_size)
    model_config = RobertaConfig.from_pretrained(conf.model_size)


saved_model_path = os.path.join(conf.output_path, conf.saved_model_path)
model_dir_name = datetime.now().strftime("%Y%m%d%H%M%S") + \
    "_" + conf.model_save_name
model_dir = os.path.join(
    conf.output_path, 'inference_only_' + model_dir_name)
results_path = os.path.join(model_dir, "results")
os.makedirs(results_path, exist_ok=False)
log_file = os.path.join(results_path, 'log.txt')


op_list = read_txt(conf.op_list_file, log_file)
op_list = [op + '(' for op in op_list] # 每个操作符后添加一个左括号 '('
op_list = ['EOF', 'UNK', 'GO', ')'] + op_list
const_list = read_txt(conf.const_list_file, log_file)
const_list = [const.lower().replace('.', '_') for const in const_list]
reserved_token_size = len(op_list) + len(const_list) # 计算操作符列表和常量列表中元素的总数

start_time = time.time()

test_data, test_examples, op_list, const_list = \
    read_examples(input_path=conf.test_file, tokenizer=tokenizer,
                  op_list=op_list, const_list=const_list, log_file=log_file)
    
print("Data loading time: ", time.time() - start_time)

kwargs = {"examples": test_examples,
          "tokenizer": tokenizer,
          "option": conf.option,
          "is_training": False,
          "max_seq_length": conf.max_seq_length,
          }


kwargs["examples"] = test_examples

start_time = time.time()
test_features = convert_examples_to_features(**kwargs)
print("Feature conversion time: ", time.time() - start_time)


def generate(data_ori, data, model, ksave_dir, mode='valid'):

    pred_list = []
    pred_unk = []

    ksave_dir_mode = os.path.join(ksave_dir, mode)
    os.makedirs(ksave_dir_mode, exist_ok=True)

    data_iterator = DataLoader(
        is_training=False, data=data, batch_size=conf.batch_size_test, shuffle=False)

    k = 0
    all_logits = []
    all_filename_id = []
    all_ind = []
    with torch.no_grad(): # 不计算梯度
        for x in tqdm(data_iterator):

            input_ids = x['input_ids']
            input_mask = x['input_mask']
            segment_ids = x['segment_ids']
            label = x['label']
            filename_id = x["filename_id"]
            ind = x["ind"]

            ori_len = len(input_ids)
            
            start_time = time.time()
            for each_item in [input_ids, input_mask, segment_ids]:
                if ori_len < conf.batch_size_test:
                    # 如果批次的大小小于配置的测试批次大小（conf.batch_size_test），则将输入特征填充至该大小
                    # 确保了模型可以接收固定大小的批次，即使实际的数据量可能不足一个完整批次
                    each_len = len(each_item[0])
                    pad_x = [0] * each_len
                    each_item += [pad_x] * (conf.batch_size_test - ori_len)
            print("Padding time: ", time.time() - start_time)

            input_ids = torch.tensor(input_ids).to(conf.device)
            input_mask = torch.tensor(input_mask).to(conf.device)
            segment_ids = torch.tensor(segment_ids).to(conf.device)

            start_time = time.time()
            logits = model(True, input_ids, input_mask,
                           segment_ids, device=conf.device)
            print("Model training time: ", time.time() - start_time)

            all_logits.extend(logits.tolist())
            all_filename_id.extend(filename_id)
            all_ind.extend(ind)

    output_prediction_file = os.path.join(ksave_dir_mode,
                                          "predictions.json")

    if mode == "valid":
        start_time = time.time()
        print_res = retrieve_evaluate(
            all_logits, all_filename_id, all_ind, output_prediction_file, conf.valid_file, topn=conf.topn)
        print("Evaluation time: ", time.time() - start_time)
    elif mode == "test":
        start_time = time.time()
        print_res = retrieve_evaluate(
            all_logits, all_filename_id, all_ind, output_prediction_file, conf.test_file, topn=conf.topn)
        print("Evaluation time: ", time.time() - start_time)
    else:
        start_time = time.time()
        # private data mode
        print_res = retrieve_evaluate_private(
            all_logits, all_filename_id, all_ind, output_prediction_file, conf.test_file, topn=conf.topn)
        print("Evaluation time: ", time.time() - start_time)

    write_log(log_file, print_res)
    print(print_res)
    return


def generate_test():
    start_time = time.time()
    model = Bert_model(hidden_size=model_config.hidden_size,
                       dropout_rate=conf.dropout_rate,)
    print("Model training time: ", time.time() - start_time)

    # 自动分割数据，使得每个GPU处理一部分数据
    model = nn.DataParallel(model)
    model.to(conf.device)
    model.load_state_dict(torch.load(conf.saved_model_path))
    model.eval() # 评估模式（如dropout层和批归一化层）
    generate(test_data, test_features, model, results_path, mode='test')


if __name__ == '__main__':

    generate_test()
