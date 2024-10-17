#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script

Total number of entries in the training data: 6251
Total number of entries in the valid data: 883
Total number of entries in the test data: 1147
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
    # 分词, 将单词转换为id
    from transformers import BertTokenizer
    # 模型架构配置
    from transformers import BertConfig
    tokenizer = BertTokenizer.from_pretrained(conf.model_size)
    model_config = BertConfig.from_pretrained(conf.model_size)

elif conf.pretrained_model == "roberta":
    from transformers import RobertaTokenizer
    from transformers import RobertaConfig
    tokenizer = RobertaTokenizer.from_pretrained(conf.model_size)
    model_config = RobertaConfig.from_pretrained(conf.model_size)

# create output paths
if conf.mode == "train":
    model_dir_name = conf.model_save_name + "_" + \
        datetime.now().strftime("%Y%m%d%H%M%S")
    model_dir = os.path.join(conf.output_path, model_dir_name)
    results_path = os.path.join(model_dir, "results")
    saved_model_path = os.path.join(model_dir, "saved_model")
    os.makedirs(saved_model_path, exist_ok=False)
    os.makedirs(results_path, exist_ok=False)
    log_file = os.path.join(results_path, 'log.txt')

else:
    saved_model_path = os.path.join(conf.output_path, conf.saved_model_path)
    model_dir_name = datetime.now().strftime("%Y%m%d%H%M%S")
    model_dir = os.path.join(
        conf.output_path, 'inference_only_' + model_dir_name)
    results_path = os.path.join(model_dir, "results")
    os.makedirs(results_path, exist_ok=False)
    log_file = os.path.join(results_path, 'log.txt')


op_list = read_txt(conf.op_list_file, log_file)
op_list = [op + '(' for op in op_list]
op_list = ['EOF', 'UNK', 'GO', ')'] + op_list
const_list = read_txt(conf.const_list_file, log_file)
const_list = [const.lower().replace('.', '_') for const in const_list]
reserved_token_size = len(op_list) + len(const_list)

# 读取部分
""" train_data, train_examples, op_list, const_list = \
    read_partial_examples(input_path=conf.train_file, tokenizer=tokenizer,
                  op_list=op_list, const_list=const_list, log_file=log_file,
                  max_examples=100)

valid_data, valid_examples, op_list, const_list = \
    read_partial_examples(input_path=conf.valid_file, tokenizer=tokenizer,
                  op_list=op_list, const_list=const_list, log_file=log_file,
                  max_examples=10)

test_data, test_examples, op_list, const_list = \
    read_partial_examples(input_path=conf.test_file, tokenizer=tokenizer,
                  op_list=op_list, const_list=const_list, log_file=log_file,
                  max_examples=10) """


train_data, train_examples, op_list, const_list = read_examples(input_path=conf.train_file, tokenizer=tokenizer,op_list=op_list, const_list=const_list, log_file=log_file)

valid_data, valid_examples, op_list, const_list = read_examples(input_path=conf.valid_file, tokenizer=tokenizer,op_list=op_list, const_list=const_list, log_file=log_file)

test_data, test_examples, op_list, const_list = read_examples(input_path=conf.test_file, tokenizer=tokenizer, op_list=op_list, const_list=const_list, log_file=log_file)



kwargs = {"examples": train_examples,
          "tokenizer": tokenizer,
          "option": conf.option,
          "is_training": True,
          "max_seq_length": conf.max_seq_length,
          }

print("len(train_examples): ", len(train_examples))

# 将一系列的 examples 返回格式化的特征
train_features = convert_examples_to_features(**kwargs)
kwargs["examples"] = valid_examples
kwargs["is_training"] = False
valid_features = convert_examples_to_features(**kwargs)
kwargs["examples"] = test_examples
test_features = convert_examples_to_features(**kwargs)


print("len(train_features): ", len(train_features))

def train():
    # keep track of all input parameters
    write_log(log_file, "####################INPUT PARAMETERS###################")
    for attr in conf.__dict__:
        # .__dict__是一个特殊的属性，它包含了一个对象的所有属性和它们的值
        value = conf.__dict__[attr]
        write_log(log_file, attr + " = " + str(value))
    write_log(log_file, "#######################################################")

    # 给定的隐藏层大小和dropout率初始化BERT模型
    model = Bert_model(hidden_size=model_config.hidden_size,
                       dropout_rate=conf.dropout_rate,)
    # 多个GPU上并行
    model = nn.DataParallel(model)
    # 将模型发送到指定的设备
    model.to(conf.device)
    optimizer = optim.Adam(model.parameters(), conf.learning_rate)
    # 使用交叉熵损失函数，忽略标签为-1的样本
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    model.train()
    
    print("len(train_features): ", len(train_features))

    # 加载训练数据，数据被分成多个批次
    train_iterator = DataLoader(
        is_training=True, data=train_features, batch_size=conf.batch_size, shuffle=True)
    
    print("len(train_iterator): ", len(train_iterator))
    # assert False, "Stop here"
    
    # 初始化用于记录训练进度和性能的变量，如批次计数器、损失记录等
    k = 0
    record_k = 0
    record_loss_k = 0
    loss, start_time = 0.0, time.time()
    record_loss = 0.0
    
    # raise Exception("stop here")
    for _ in range(conf.epoch):
        train_iterator.reset()
        # reset 方法能够将迭代器重置到初始状态，确保数据能够从头开始重新迭代
        for x in train_iterator:
            
            print("current batch: ", k)
            start_time = time.time()

            input_ids = torch.tensor(x['input_ids']).to(conf.device)
            input_mask = torch.tensor(x['input_mask']).to(conf.device)
            segment_ids = torch.tensor(x['segment_ids']).to(conf.device)
            label = torch.tensor(x['label']).to(conf.device)
            
            print("complete tensor: ", time.time() - start_time)
            start_time = time.time()

            # 清除之前的梯度
            model.zero_grad()
            optimizer.zero_grad()

            this_logits = model(True, input_ids, input_mask,
                                segment_ids, device=conf.device)
            
            print("complete inference: ", time.time() - start_time)
            start_time = time.time()        

            # 计算损失
            this_loss = criterion(
                this_logits.view(-1, this_logits.shape[-1]), label.view(-1))

            this_loss = this_loss.sum()
            record_loss += this_loss.item() * 100
            record_k += 1
            k += 1

            # 执行反向传播，更新模型参数
            this_loss.backward()
            optimizer.step()
            
            print("complete backward: ", time.time() - start_time)

            if k > 1 and k % conf.report_loss == 0:
                # 每处理一定数量的批次后，记录当前的平均损失
                print("Report loss, k: ", k // conf.report_loss)
                write_log(log_file, "%d : loss = %.3f" %
                          (k, record_loss / record_k))
                record_loss = 0.0
                record_k = 0

            if k > 1 and k % conf.report == 0:
                # 每隔一定数量的批次，评估模型在验证集上的性能，并保存当前的模型
                print("Evaluate model, k: ", k // conf.report)
                model.eval()
                cost_time = time.time() - start_time
                write_log(log_file, "%d : time = %.3f " %
                          (k // conf.report, cost_time))
                start_time = time.time()
                if k // conf.report >= 1:
                    print("Val test")
                    # save model
                    saved_model_path_cnt = os.path.join(
                        saved_model_path, 'loads', str(k // conf.report))
                    os.makedirs(saved_model_path_cnt, exist_ok=True)
                    torch.save(model.state_dict(),
                               saved_model_path_cnt + "/model.pt")
                    # .state_dict() 是一个包含整个模型的参数和持久化缓冲区（比如批量归一化的运行平均值）的Python字典
                    # .pt 或 .pth 文件是PyTorch的标准文件扩展名，用于保存模型的参数

                    results_path_cnt = os.path.join(
                        results_path, 'loads', str(k // conf.report))
                    os.makedirs(results_path_cnt, exist_ok=True)
                    validation_result = evaluate(
                        valid_examples, valid_features, model, results_path_cnt, 'valid')
                    write_log(log_file, validation_result)

                model.train()


def evaluate(data_ori, data, model, ksave_dir, mode='valid'):
    # 评估模型在验证集（或测试集）上的性能
    # ksave_dir: 保存目录

    pred_list = []
    pred_unk = []

    ksave_dir_mode = os.path.join(ksave_dir, mode)
    os.makedirs(ksave_dir_mode, exist_ok=True)
    
    start_time = time.time()

    data_iterator = DataLoader(
        is_training=False, data=data, batch_size=conf.batch_size_test, shuffle=False)
    
    print("Data iterator complete: ", time.time() - start_time)

    k = 0
    all_logits = []
    all_filename_id = []
    all_ind = []
    with torch.no_grad():
        # 关闭梯度计算
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
                # 检查输入数据的长度，并在必要时对其进行填充，以确保每个批次的数据大小都与模型期望的一致
                if ori_len < conf.batch_size_test:
                    each_len = len(each_item[0])
                    pad_x = [0] * each_len
                    each_item += [pad_x] * (conf.batch_size_test - ori_len)
            print("Complete padding: ", time.time() - start_time)

            input_ids = torch.tensor(input_ids).to(conf.device)
            input_mask = torch.tensor(input_mask).to(conf.device)
            segment_ids = torch.tensor(segment_ids).to(conf.device)

            # 使用模型进行预测，得到logits（模型的原始输出，通常在应用softmax之前的值）
            start_time = time.time()
            logits = model(True, input_ids, input_mask,
                           segment_ids, device=conf.device)
            print("Complete inference: ", time.time() - start_time)

            # 将所有预测结果、文件名标识符、索引存储起来
            all_logits.extend(logits.tolist())
            all_filename_id.extend(filename_id)
            all_ind.extend(ind)

    output_prediction_file = os.path.join(ksave_dir_mode,
                                          "predictions.json")
    
    # print(all_filename_id)
    
    # 预测结果被写入到predictions.json文件中
    if mode == "valid":
        print_res = retrieve_evaluate(
            all_logits, all_filename_id, all_ind, output_prediction_file, conf.valid_file, topn=conf.topn)
        
    else:
        print_res = retrieve_evaluate(
            all_logits, all_filename_id, all_ind, output_prediction_file, conf.test_file, topn=conf.topn)

    write_log(log_file, print_res)
    print(print_res)
    return print_res


if __name__ == '__main__':
    train()
