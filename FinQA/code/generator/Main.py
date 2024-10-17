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


from Model_new import Bert_model

if conf.pretrained_model == "bert":
    print("Using bert")
    from transformers import BertTokenizer
    from transformers import BertConfig
    tokenizer = BertTokenizer.from_pretrained(conf.model_size)
    model_config = BertConfig.from_pretrained(conf.model_size)

elif conf.pretrained_model == "roberta":
    print("Using roberta")
    from transformers import RobertaTokenizer
    from transformers import RobertaConfig
    tokenizer = RobertaTokenizer.from_pretrained(conf.model_size)
    model_config = RobertaConfig.from_pretrained(conf.model_size)

elif conf.pretrained_model == "finbert":
    print("Using finbert")
    from transformers import BertTokenizer
    from transformers import BertConfig
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model_config = BertConfig.from_pretrained(conf.model_size)

elif conf.pretrained_model == "longformer":
    print("Using longformer")
    from transformers import LongformerTokenizer, LongformerConfig
    tokenizer = LongformerTokenizer.from_pretrained(conf.model_size)
    model_config = LongformerConfig.from_pretrained(conf.model_size)


# create output paths
if conf.mode == "train":
    model_dir_name = conf.model_save_name + "_" + \
        datetime.now().strftime("%Y%m%d%H%M%S")
    model_dir = os.path.join(conf.output_path, model_dir_name)
    results_path = os.path.join(model_dir, "results")
    saved_model_path = os.path.join(model_dir, "saved_model")
    
    print("saved_model_path: ", saved_model_path)
    
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

print("op_list: ", op_list)
print("const_list: ", const_list)

start_time = time.time()
train_data, train_examples, op_list, const_list = \
    read_examples(input_path=conf.train_file, tokenizer=tokenizer,
                  op_list=op_list, const_list=const_list, log_file=log_file)

valid_data, valid_examples, op_list, const_list = \
    read_examples(input_path=conf.valid_file, tokenizer=tokenizer,
                  op_list=op_list, const_list=const_list, log_file=log_file)

test_data, test_examples, op_list, const_list = \
    read_examples(input_path=conf.test_file, tokenizer=tokenizer,
                  op_list=op_list, const_list=const_list, log_file=log_file)
print("Time to read data: ", time.time() - start_time)

kwargs = {"examples": train_examples,
          "tokenizer": tokenizer,
          "max_seq_length": conf.max_seq_length,
          "max_program_length": conf.max_program_length,
          "is_training": True,
          "op_list": op_list,
          "op_list_size": len(op_list),
          "const_list": const_list,
          "const_list_size": len(const_list),
          "verbose": True}

print("len(train_examples): ", len(train_examples))

start_time = time.time()
train_features = convert_examples_to_features(**kwargs)
kwargs["examples"] = valid_examples
kwargs["is_training"] = False
valid_features = convert_examples_to_features(**kwargs)
kwargs["examples"] = test_examples
test_features = convert_examples_to_features(**kwargs)
print("Time to convert examples to features: ", time.time() - start_time)


def train():
    # keep track of all input parameters
    write_log(log_file, "####################INPUT PARAMETERS###################")
    for attr in conf.__dict__:
        value = conf.__dict__[attr]
        write_log(log_file, attr + " = " + str(value))
    write_log(log_file, "#######################################################")

    model = Bert_model(num_decoder_layers=conf.num_decoder_layers,
                       hidden_size=model_config.hidden_size,
                       dropout_rate=conf.dropout_rate,
                       program_length=conf.max_program_length,
                       input_length=conf.max_seq_length,
                       op_list=op_list,
                       const_list=const_list)
    # 多GPU训练
    model = nn.DataParallel(model)
    model.to(conf.device)
    optimizer = optim.Adam(model.parameters(), conf.learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    model.train()
    # torch.autograd.set_detect_anomaly(True)
    
    print("len(train_features): ", len(train_features))

    start_time = time.time()
    train_iterator = DataLoader(
        is_training=True, data=train_features, batch_size=conf.batch_size, reserved_token_size=reserved_token_size, shuffle=True)
    print("Time to create train iterator: ", time.time() - start_time)
    
    # print("train iterator first batch: ", next(iter(train_iterator)).keys())
    print("len(train_iterator): ", len(train_iterator))
    
    # assert False, "Stop here"


    k = 0
    record_k = 0
    record_loss_k = 0
    loss, start_time = 0.0, time.time()
    record_loss = 0.0

    for _ in range(conf.epoch):
        train_iterator.reset()
        
        for x in train_iterator:

            input_ids = torch.tensor(x['input_ids']).to(conf.device)
            input_mask = torch.tensor(x['input_mask']).to(conf.device)
            segment_ids = torch.tensor(x['segment_ids']).to(conf.device)
            program_ids = torch.tensor(x['program_ids']).to(conf.device)
            program_mask = torch.tensor(x['program_mask']).to(conf.device)
            option_mask = torch.tensor(x['option_mask']).to(conf.device)

            model.zero_grad()
            optimizer.zero_grad()

            start_time = time.time()
            # 模型进行前向传播，获取logits
            this_logits = model(True, input_ids, input_mask, segment_ids,
                                option_mask, program_ids, program_mask, device=conf.device)
            # 计算损失，考虑到程序掩码，这意味着只考虑非填充部分的损失
            this_loss = criterion(
                this_logits.view(-1, this_logits.shape[-1]), program_ids.view(-1))
            this_loss = this_loss * program_mask.view(-1)
            # per token loss
            this_loss = this_loss.sum() / program_mask.sum()

            record_loss += this_loss.item()
            record_k += 1
            k += 1

            this_loss.backward()
            optimizer.step()
            print("Batch time: ", time.time() - start_time)

            # 每当处理指定数量的批次后： 报告当前的损失
            if k > 1 and k % conf.report_loss == 0:
                write_log(log_file, "%d : loss = %.3f" %
                          (k, record_loss / record_k))
                record_loss = 0.0
                record_k = 0

            # 检查是否需要进行评估（验证集上的性能评估）
            if k > 1 and k % conf.report == 0:
                print("Round: ", k / conf.report)
                model.eval()
                cost_time = time.time() - start_time
                write_log(log_file, "%d : time = %.3f " %
                          (k // conf.report, cost_time))
                start_time = time.time()
                # 如果是评估时间，则切换模型到评估模式，记录时间，保存当前模型，并在验证集上进行评估
                if k // conf.report >= 1:
                    print("Val test")
                    # save model
                    saved_model_path_cnt = os.path.join(
                        saved_model_path, 'loads', str(k // conf.report))
                    os.makedirs(saved_model_path_cnt, exist_ok=True)
                    
                    print("saved_model_path_cnt: ", saved_model_path_cnt)
                    print("model: ", model)
                    
                    torch.save(model.state_dict(),
                               saved_model_path_cnt + "/model.pt")

                    results_path_cnt = os.path.join(
                        results_path, 'loads', str(k // conf.report))
                    
                    print("results_path_cnt: ", results_path_cnt)
                    
                    os.makedirs(results_path_cnt, exist_ok=True)
                    validation_result = evaluate(
                        valid_examples, valid_features, model, results_path_cnt, 'valid')
                    write_log(log_file, validation_result)
                
                # 完成评估后，将模型切回训练模式
                model.train()
        print("How many ks: ", k)


def evaluate(data_ori, data, model, ksave_dir, mode='valid'):
    # 函数接收原始数据集 (data_ori)，处理后的数据集 (data)，模型实例 (model)，保存目录 (ksave_dir)，以及评估模式 (mode)
    
    pred_list = []
    pred_unk = []

    # 创建一个保存目录，用于存放评估结果
    ksave_dir_mode = os.path.join(ksave_dir, mode)
    os.makedirs(ksave_dir_mode, exist_ok=True)
    
    print("len(data): ", len(data))

    start_time = time.time()
    data_iterator = DataLoader(
        is_training=False, data=data, batch_size=conf.batch_size_test, reserved_token_size=reserved_token_size, shuffle=False)
    print("Time to create data iterator: ", time.time() - start_time)

    print("len(data_iterator): ", len(data_iterator))

    k = 0
    all_results = []
    with torch.no_grad():
        # 开始一个没有梯度计算的环境
        for x in tqdm(data_iterator):

            input_ids = x['input_ids']
            input_mask = x['input_mask']
            segment_ids = x['segment_ids']
            program_ids = x['program_ids']
            program_mask = x['program_mask']
            option_mask = x['option_mask']

            ori_len = len(input_ids)
            
            start_time = time.time()
            for each_item in [input_ids, input_mask, segment_ids, program_ids, program_mask, option_mask]:
                if ori_len < conf.batch_size_test:
                    # 如果批次中的数据量小于预定的批处理大小，进行填充操作，确保输入模型的数据批量一致
                    each_len = len(each_item[0])
                    pad_x = [0] * each_len
                    each_item += [pad_x] * (conf.batch_size_test - ori_len)
            print("Pad time: ", time.time() - start_time)

            input_ids = torch.tensor(input_ids).to(conf.device)
            input_mask = torch.tensor(input_mask).to(conf.device)
            segment_ids = torch.tensor(segment_ids).to(conf.device)
            program_ids = torch.tensor(program_ids).to(conf.device)
            program_mask = torch.tensor(program_mask).to(conf.device)
            option_mask = torch.tensor(option_mask).to(conf.device)

            start_time = time.time()
            # 使用模型进行前向传播，获取logits
            logits = model(False, input_ids, input_mask,
                           segment_ids, option_mask, program_ids, program_mask, device=conf.device)
            print("Infer time: ", time.time() - start_time)
            
            print("len(logits): ", len(logits))
            print("len(x[unique_id]): ", len(x["unique_id"]))

            for this_logit, this_id in zip(logits.tolist(), x["unique_id"]):
                # 将logits和对应的唯一ID保存到结果列表 all_results 中
                # RawResult 是一个namedtuple，包含三个属性：unique_id, logits, loss
                all_results.append(
                    RawResult(
                        unique_id=int(this_id),
                        logits=this_logit,
                        loss=None
                    ))
    
    print("len(all_results): ", len(all_results))

    output_prediction_file = os.path.join(ksave_dir_mode,
                                          "predictions.json")
    output_nbest_file = os.path.join(ksave_dir_mode,
                                     "nbest_predictions.json")
    output_eval_file = os.path.join(ksave_dir_mode, "full_results.json")
    output_error_file = os.path.join(ksave_dir_mode, "full_results_error.json")
    
    print("output_prediction_file: ", output_prediction_file)
    print("output_nbest_file: ", output_nbest_file)
    
    print("data_ori: ", len(data_ori))
    print("data: ", len(data))
    print("all_results: ", len(all_results))    
    

    # 调用 compute_predictions 函数来计算模型的最终预测
    all_predictions, all_nbest = compute_predictions(
        data_ori,
        data,
        all_results,
        n_best_size=conf.n_best_size,
        max_program_length=conf.max_program_length,
        tokenizer=tokenizer,
        op_list=op_list,
        op_list_size=len(op_list),
        const_list=const_list,
        const_list_size=len(const_list))
    
    print("all_predictions: ", len(all_predictions))
    print("all_nbest: ", len(all_nbest))
    
    
    write_predictions(all_predictions, output_prediction_file)
    write_predictions(all_nbest, output_nbest_file)

    if mode == "valid":
        original_file = conf.valid_file
    else:
        original_file = conf.test_file

    # 根据模型的预测和实际的数据计算执行准确率（exe_acc）和程序准确率（prog_acc）
    exe_acc, prog_acc = evaluate_result(
        output_nbest_file, original_file, output_eval_file, output_error_file, program_mode=conf.program_mode)

    prog_res = "exe acc: " + str(exe_acc) + " prog acc: " + str(prog_acc)
    # write_log(log_file, prog_res)

    return prog_res


if __name__ == '__main__':

    if conf.mode == "train":
        train()
