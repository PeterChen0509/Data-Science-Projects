import argparse
import collections
import json
import os
import sys
import random


'''
convert retriever results to generator test input
'''

sys.path.insert(0, '../utils/')
from general_utils import table_row_to_text

### for single sent retrieve

def convert_test(json_in, json_out, topn, max_len):
    # 处理和转换JSON格式的数据。它从输入文件读取数据，根据分数和长度限制选择和整理文本数据，然后按照特定的顺序构建模型输入，最后将处理后的数据保存到输出文件

    with open(json_in) as f_in:
        data = json.load(f_in)

    for each_data in data:
        table_retrieved = each_data["table_retrieved"]
        text_retrieved = each_data["text_retrieved"]

        pre_text = each_data["pre_text"]
        post_text = each_data["post_text"]
        all_text = pre_text + post_text

        table = each_data["table"]

        # 将检索到的表格数据和文本数据合并到 all_retrieved 中
        all_retrieved = each_data["table_retrieved"] + each_data["text_retrieved"]

        sorted_dict = sorted(all_retrieved, key=lambda kv: kv["score"], reverse=True)

        acc_len = 0
        # 存储选中的文本和表格数据
        all_text_in = {}
        all_table_in = {}

        for tmp in sorted_dict:
            # 按分数选取 topn 个文本片段
            if len(all_table_in) + len(all_text_in) >= topn:
                break
            this_sent_ind = int(tmp["ind"].split("_")[1])

            if "table" in tmp["ind"]:
                this_sent = table_row_to_text(table[0], table[this_sent_ind])
            else:
                this_sent = all_text[this_sent_ind]

            if acc_len + len(this_sent.split(" ")) < max_len:
                # 检查当前文本长度加上新文本长度是否会超过 max_len，如果不超过，则将新文本添加到 all_text_in 或 all_table_in
                if "table" in tmp["ind"]:
                    all_table_in[tmp["ind"]] = this_sent
                else:
                    all_text_in[tmp["ind"]] = this_sent

                acc_len += len(this_sent.split(" "))
            else:
                break

        # 将选中的文本按照原始顺序整理到 this_model_input 中
        this_model_input = []

        # sorted_dict = sorted(all_table_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
        # this_model_input.extend(sorted_dict)

        # sorted_dict = sorted(all_text_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
        # this_model_input.extend(sorted_dict)

        # original_order
        sorted_dict_table = sorted(all_table_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
        sorted_dict_text = sorted(all_text_in.items(), key=lambda kv: int(kv[0].split("_")[1]))

        # 首先添加 pre_text 部分的文本，然后添加表格数据，最后添加 post_text 部分的文本
        for tmp in sorted_dict_text:
            if int(tmp[0].split("_")[1]) < len(pre_text):
                this_model_input.append(tmp)

        for tmp in sorted_dict_table:
            this_model_input.append(tmp)

        for tmp in sorted_dict_text:
            if int(tmp[0].split("_")[1]) >= len(pre_text):
                this_model_input.append(tmp)

        each_data["qa"]["model_input"] = this_model_input


    with open(json_out, "w") as f:
        json.dump(data, f, indent=4)

    print(len(data))


def convert_train(json_in, json_out, topn, max_len):
    # 处理和转换一组问答（QA）数据，这些数据存储在 JSON 格式中

    with open(json_in) as f_in:
        data = json.load(f_in)

    for each_data in data:
        table_retrieved = each_data["table_retrieved"]
        text_retrieved = each_data["text_retrieved"]

        pre_text = each_data["pre_text"]
        post_text = each_data["post_text"]
        all_text = pre_text + post_text

        # 确定正确的索引
        gold_inds = each_data["qa"]["gold_inds"]

        table = each_data["table"]

        all_retrieved = each_data["table_retrieved"] + each_data["text_retrieved"]

        # 从 all_retrieved 中筛选出非正确的检索结果（false_retrieved）
        false_retrieved = []
        for tmp in all_retrieved:
            if tmp["ind"] not in gold_inds:
                false_retrieved.append(tmp)

        # 根据分数对 false_retrieved 进行排序
        sorted_dict = sorted(false_retrieved, key=lambda kv: kv["score"], reverse=True)

        acc_len = 0
        # 创建字典 all_text_in 和 all_table_in 来存储正确的文本和表格数据
        all_text_in = {}
        all_table_in = {}

        for tmp in gold_inds:
            if "table" in tmp:
                all_table_in[tmp] = gold_inds[tmp]
            else:
                all_text_in[tmp] = gold_inds[tmp]

        context = ""
        
        # 初始化 context 和 acc_len 来累计上下文和当前单词总数
        for tmp in gold_inds:
            context += gold_inds[tmp]

        acc_len = len(context.split(" "))

        for tmp in sorted_dict:
            # 逐个添加排序后的 false_retrieved 项，直到达到 topn 限制或累积长度达到 max_len
            if len(all_table_in) + len(all_text_in) >= topn:
                break
            this_sent_ind = int(tmp["ind"].split("_")[1])

            if "table" in tmp["ind"]:
                this_sent = table_row_to_text(table[0], table[this_sent_ind])
            else:
                this_sent = all_text[this_sent_ind]

            if acc_len + len(this_sent.split(" ")) < max_len:
                if "table" in tmp["ind"]:
                    all_table_in[tmp["ind"]] = this_sent
                else:
                    all_text_in[tmp["ind"]] = this_sent

                acc_len += len(this_sent.split(" "))
            else:
                break

        this_model_input = []

        # sorted_dict = sorted(all_table_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
        # this_model_input.extend(sorted_dict)

        # sorted_dict = sorted(all_text_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
        # this_model_input.extend(sorted_dict)

        # 分别对表格和文本数据进行排序，确保它们按照在原始文本中的顺序添加
        sorted_dict_table = sorted(all_table_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
        sorted_dict_text = sorted(all_text_in.items(), key=lambda kv: int(kv[0].split("_")[1]))

        # 将排序后的表格和文本数据合并到 this_model_input 中
        for tmp in sorted_dict_text:
            if int(tmp[0].split("_")[1]) < len(pre_text):
                this_model_input.append(tmp)

        for tmp in sorted_dict_table:
            this_model_input.append(tmp)

        for tmp in sorted_dict_text:
            if int(tmp[0].split("_")[1]) >= len(pre_text):
                this_model_input.append(tmp)
        
        # 将构建好的模型输入添加到当前数据项的 "qa" 字段下的 "model_input"
        each_data["qa"]["model_input"] = this_model_input


    # 将处理后的整个数据写入到 json_out 文件中，使用缩进以便于阅读
    with open(json_out, "w") as f:
        json.dump(data, f, indent=4)


        


### for sliding windows

def convert_generator_sliding(json_in, json_out, max_len, stride, mode="train"):
    '''
    extract examples that can be covered by sliding windows
    exclude others
    '''

    with open(json_in) as f_in:
        data = json.load(f_in)

    good_window = 0
    res = []
    for each_data in data:

        pre_text = each_data["pre_text"]
        post_text = each_data["post_text"]

        gold_inds = each_data["qa"]["gold_inds"]

        table = each_data["table"]

        table_text = ""
        for row in table[1:]:
            this_row_text = table_row_to_text(table[0], row)
            table_text += (this_row_text + " ")

        table_text = table_text.strip()

        pos_windows = []
        neg_windows = []

        all_text = " ".join(pre_text) + " " + table_text + " " + " ".join(post_text)
        all_text = all_text.replace(". . . . . .", "")
        all_text = all_text.replace("* * * * * *", "")
        all_text_list = all_text.split(" ")

        start = 0
        while start < len(all_text_list) - 1:
            this_window = all_text_list[start: start + max_len]
            this_window_text = " ".join(this_window)

            # whether contains all evi
            num_gold = 0
            for tmp in gold_inds:
                if gold_inds[tmp] in this_window_text:
                    num_gold += 1

            if num_gold == len(gold_inds):
                pos_windows.append([this_window_text, num_gold])
            else:
                neg_windows.append([this_window_text, num_gold])

            neg_windows = sorted(neg_windows, key=lambda kv: kv[1], reverse=True)
            start += stride

        if len(pos_windows) > 0:
            good_window += 1

        
        each_data["qa"]["pos_windows"] = pos_windows
        each_data["qa"]["neg_windows"] = neg_windows

        if mode == "train":
            if len(pos_windows) > 0 and "............" not in all_text:
                res.append(each_data)
        else:
            res.append(each_data)


    print("All: ", len(data))
    print("Good windows: ", good_window)
    print(float(good_window) / len(data))

    with open(json_out, "w") as f:
        json.dump(res, f, indent=4)


def convert_test_infer(json_in, json_out, topn, mode):

    with open(json_in) as f_in:
        data = json.load(f_in)

    for each_data in data:
        table_retrieved = each_data["table_retrieved_all"]
        text_retrieved = each_data["text_retrieved_all"]

        pre_text = each_data["pre_text"]
        post_text = each_data["post_text"]
        all_text = pre_text + post_text

        table = each_data["table"]

        # all_retrieved = each_data["table_retrieved"] + each_data["text_retrieved"]
        # sorted_dict = sorted(all_retrieved, key=lambda kv: kv["score"], reverse=True)

        sorted_dict_text = sorted(text_retrieved, key=lambda kv: kv["score"], reverse=True)
        sorted_dict_table = sorted(table_retrieved, key=lambda kv: kv["score"], reverse=True)

        # print(sorted_dict_table)

        acc_len = 0
        all_text_in = {}
        all_table_in = {}

        # if mode == "table":
        for tmp in sorted_dict_table[:topn]:
            this_sent_ind = int(tmp["ind"].split("_")[1])
            this_sent = table_row_to_text(table[0], table[this_sent_ind])
            all_table_in[tmp["ind"]] = this_sent

        for tmp in sorted_dict_text[:topn]:
            this_sent_ind = int(tmp["ind"].split("_")[1])
            all_text_in[tmp["ind"]] = all_text[this_sent_ind]


        this_model_input = []

        # sorted_dict = sorted(all_table_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
        # this_model_input.extend(sorted_dict)

        # sorted_dict = sorted(all_text_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
        # this_model_input.extend(sorted_dict)

        # original_order
        sorted_dict_table = sorted(all_table_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
        sorted_dict_text = sorted(all_text_in.items(), key=lambda kv: int(kv[0].split("_")[1]))

        # for tmp in sorted_dict_text:
        #     if int(tmp[0].split("_")[1]) < len(pre_text):
        #         this_model_input.append(tmp)

        # for tmp in sorted_dict_table:
        #     this_model_input.append(tmp)

        # for tmp in sorted_dict_text:
        #     if int(tmp[0].split("_")[1]) >= len(pre_text):
        #         this_model_input.append(tmp)

        if mode == "table":
            for tmp in sorted_dict_table:
                this_model_input.append(tmp)
        else:
            for tmp in sorted_dict_text:
                this_model_input.append(tmp)

        each_data["qa"]["model_input"] = this_model_input


    with open(json_out, "w") as f:
        json.dump(data, f, indent=4)

    print(len(data))


if __name__ == '__main__':

    root = "/home/peterchen/Model/FinQA/results(original)/generator_output/top5/"

    # json_in = root + "outputs/inference_only_20220504054235_new_correct_retriever_train/results/test/predictions.json"
    # json_out = root + "finQA/dataset/train_retrieve_correct.json"
    # convert_train(json_in, json_out, topn=3, max_len=290)
    
    
    
    # json_in = root + "inference_only_20240130130143_retriever-bert-base-test(Train set)/results/test/predictions.json"
    json_in = "/home/peterchen/Model/FinQA/results(original)/retriever_output/inference_only_20240210012135_retriever-bert-base-test (train)/results/test/predictions.json"
    json_out = root + "data/train_retrieve.json"
    convert_train(json_in, json_out, topn=5, max_len=290)

    # json_in = root + "inference_only_20240130130143_retriever-bert-base-test(Test set)/results/test/predictions.json"
    json_in = "/home/peterchen/Model/FinQA/results(original)/retriever_output/inference_only_20240210032716_retriever-bert-base-test (test)/results/test/predictions.json"
    json_out = root + "data/test_retrieve.json"
    convert_test(json_in, json_out, topn=5, max_len=290)
    
    # json_in = root + "inference_only_20240130130143_retriever-bert-base-test(Dev set)/results/test/predictions.json"
    json_in = "/home/peterchen/Model/FinQA/results(original)/retriever_output/inference_only_20240210032142_retriever-bert-base-test (dev)/results/test/predictions.json"
    json_out = root + "data/dev_retrieve.json"
    convert_test(json_in, json_out, topn=5, max_len=290)


    # ### fake data for infer experiments
    # json_in = root + "outputs/inference_only_20220504051252_new_correct_retriever_test/results/test/predictions.json"
    # json_out = root + "finQA/dataset/test_correct_table_only.json"
    # convert_test_infer(json_in, json_out, topn=3, mode="table")

    # json_in = root + "outputs/inference_only_20220504051252_new_correct_retriever_test/results/test/predictions.json"
    # json_out = root + "finQA/dataset/test_correct_text_only.json"
    # convert_test_infer(json_in, json_out, topn=3, mode="text")


    # train_ori = root + "finQA/dataset/test.json"
    # train_slide = root + "finQA/dataset/test_slide.json"
    # convert_generator_sliding(train_ori, train_slide, max_len=200, stride=30, mode="test")