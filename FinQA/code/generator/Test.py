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


saved_model_path = os.path.join(conf.output_path, conf.saved_model_path)
# model_dir_name = datetime.now().strftime("%Y%m%d%H%M%S")
model_dir_name = datetime.now().strftime("%Y%m%d%H%M%S") + \
    "_" + conf.model_save_name
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

start_time = time.time()
test_data, test_examples, op_list, const_list = \
    read_examples(input_path=conf.test_file, tokenizer=tokenizer,
                  op_list=op_list, const_list=const_list, log_file=log_file)
print("Time to read test data: ", time.time() - start_time)

print(const_list)
print(op_list)

print("len of test_examples: ", len(test_examples))

kwargs = {"examples": test_examples,
          "tokenizer": tokenizer,
          "max_seq_length": conf.max_seq_length,
          "max_program_length": conf.max_program_length,
          "is_training": False,
          "op_list": op_list,
          "op_list_size": len(op_list),
          "const_list": const_list,
          "const_list_size": len(const_list),
          "verbose": True}

start_time = time.time()
test_features = convert_examples_to_features(**kwargs)
print("Time to convert test features: ", time.time() - start_time)

print("len of test_features: ", len(test_features))

def generate(data_ori, data, model, ksave_dir, mode='valid'):
    # 处理输入数据，通过模型生成预测，并将预测结果保存到指定的文件中

    pred_list = []
    pred_unk = []

    ksave_dir_mode = os.path.join(ksave_dir, mode)
    os.makedirs(ksave_dir_mode, exist_ok=True)

    data_iterator = DataLoader(
        is_training=False, data=data, batch_size=conf.batch_size_test, reserved_token_size=reserved_token_size, shuffle=False)

    print("len of data_iterator: ", len(data_iterator))
    
    # assert False, "stop here"
    
    k = 0
    all_results = []
    with torch.no_grad():
        for x in tqdm(data_iterator):

            input_ids = x['input_ids']
            input_mask = x['input_mask']
            segment_ids = x['segment_ids']
            program_ids = x['program_ids']
            program_mask = x['program_mask']
            option_mask = x['option_mask']

            ori_len = len(input_ids)
            for each_item in [input_ids, input_mask, segment_ids, program_ids, program_mask, option_mask]:
                if ori_len < conf.batch_size_test:
                    each_len = len(each_item[0])
                    pad_x = [0] * each_len
                    each_item += [pad_x] * (conf.batch_size_test - ori_len)

            input_ids = torch.tensor(input_ids).to(conf.device)
            input_mask = torch.tensor(input_mask).to(conf.device)
            segment_ids = torch.tensor(segment_ids).to(conf.device)
            program_ids = torch.tensor(program_ids).to(conf.device)
            program_mask = torch.tensor(program_mask).to(conf.device)
            option_mask = torch.tensor(option_mask).to(conf.device)

            start_time = time.time()
            logits = model(False, input_ids, input_mask,
                           segment_ids, option_mask, program_ids, program_mask, device=conf.device)
            print("Time to get logits: ", time.time() - start_time)


            # 对于每个logits，创建一个 RawResult 对象，并将其添加到 all_results
            for this_logit, this_id in zip(logits.tolist(), x["unique_id"]):
                all_results.append(
                    RawResult(
                        unique_id=int(this_id),
                        logits=this_logit,
                        loss=None
                    ))

    output_prediction_file = os.path.join(ksave_dir_mode,
                                          "predictions.json")
    output_nbest_file = os.path.join(ksave_dir_mode,
                                     "nbest_predictions.json")
    output_eval_file = os.path.join(ksave_dir_mode, "evals.json")

    # 生成最终的预测 all_predictions 和最佳预测 all_nbest
    start_time = time.time()
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
    print("Time to compute predictions: ", time.time() - start_time)
    write_predictions(all_predictions, output_prediction_file)
    write_predictions(all_nbest, output_nbest_file)

    return


def generate_test():
    start_time = time.time()
    model = Bert_model(num_decoder_layers=conf.num_decoder_layers,
                       hidden_size=model_config.hidden_size,
                       dropout_rate=conf.dropout_rate,
                       program_length=conf.max_program_length,
                       input_length=conf.max_seq_length,
                       op_list=op_list,
                       const_list=const_list)
    print("Time to create model: ", time.time() - start_time)
    
    model = nn.DataParallel(model)
    model.to(conf.device)
    # torch.load 主要用于加载序列化的数据到内存中，这些数据通常是使用 torch.save 函数保存的(能够加载几乎任何类型的对象，包括模型参数、优化器状态等)
    # load_state_dict 用于加载状态字典（state_dict）到模型中, 这个状态字典包含了模型的参数（权重和偏置）
    # torch.load 和 model.load_state_dict 通常结合在一起使用，torch.load 负责从磁盘加载 state_dict，而 model.load_state_dict 负责将这个 state_dict 应用到模型的参数中
    model.load_state_dict(torch.load(conf.saved_model_path))
    model.eval()
    generate(test_examples, test_features, model, results_path, mode='test')

    if conf.mode != "private":
        res_file = results_path + "/test/nbest_predictions.json"
        error_file = results_path + "/test/full_results_error.json"
        all_res_file = results_path + "/test/full_results.json"
        evaluate_score(res_file, error_file, all_res_file)


def evaluate_score(file_in, error_file, all_res_file):

    start_time = time.time()
    exe_acc, prog_acc = evaluate_result(
        file_in, conf.test_file, all_res_file, error_file, program_mode=conf.program_mode)
    print("Time to evaluate result: ", time.time() - start_time)

    prog_res = "exe acc: " + str(exe_acc) + " prog acc: " + str(prog_acc)
    write_log(log_file, prog_res)


if __name__ == '__main__':

    generate_test()
