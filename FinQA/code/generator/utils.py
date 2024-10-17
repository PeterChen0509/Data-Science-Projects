import time
import os
import sys
import shutil
import io
import subprocess
import re
import zipfile
import json
import copy
import torch
import random
import collections
import math
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from config import parameters as conf
from transformers import BertTokenizer, BertModel, BertConfig
import finqa_utils as finqa_utils
from sympy import simplify

# Progress bar

TOTAL_BAR_LENGTH = 100.
last_time = time.time()
begin_time = last_time
print(os.popen('stty size', 'r').read())
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)


all_ops = ["add", "subtract", "multiply", "divide", "exp", "greater", "table_max",
           "table_min", "table_sum", "table_average"]


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def write_word(pred_list, save_dir, name):
    ss = open(save_dir + name, "w+")
    for item in pred_list:
        ss.write(" ".join(item) + '\n')


def get_current_git_version():
    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha


def write_log(log_file, s):
    print(s)
    with open(log_file, 'a') as f:
        f.write(s+'\n')


def _compute_softmax(scores):
    # 计算一组分数（通常是模型输出的logits）的softmax概率
    if not scores:
        # 如果输入的 scores 是空的，立即返回一个空列表
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            # 为了后续的数值稳定性优化，防止计算 exp(score) 时溢出
            max_score = score

    # 初始化 exp_scores 为空列表，用于存储每个分数减去 max_score 后的指数值
    exp_scores = []
    # 累计所有指数分数的和
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = [] # 存储最终的softmax概率
    for score in exp_scores:
        # 得到当前分数对应的softmax概率, 添加到 probs
        probs.append(score / total_sum)
    return probs


def read_txt(input_path, log_file):
    """Read a txt file into a list."""

    write_log(log_file, "Reading: %s" % input_path)
    with open(input_path) as input_file:
        input_data = input_file.readlines()
    items = []
    for line in input_data:
        items.append(line.strip())
    return items


def read_examples(input_path, tokenizer, op_list, const_list, log_file):
    """Read a json file into a list of examples."""

    write_log(log_file, "Reading " + input_path)
    with open(input_path) as input_file:
        input_data = json.load(input_file)

    examples = []
    for entry in tqdm(input_data):
        examples.append(finqa_utils.read_mathqa_entry(entry, tokenizer))
        program = examples[-1].program
        # for tok in program:
        #     if 'const_' in tok and not (tok in const_list):
        #         const_list.append(tok)
        #     elif '(' in tok and not (tok in op_list):
        #         op_list.append(tok)
    return input_data, examples, op_list, const_list


def convert_examples_to_features(examples,
                                 tokenizer,
                                 max_seq_length,
                                 max_program_length,
                                 is_training,
                                 op_list,
                                 op_list_size,
                                 const_list,
                                 const_list_size,
                                 verbose=True):
    # 将一系列的示例（比如文本或者数据记录）转换成模型输入特征
    # 函数开始时，初始化一个唯一ID（unique_id）和一个空列表res，用于存储转换后的特征
    unique_id = 1000000000
    res = []
    # example是一个MathQAExample,里面定义了convert_single_example函数
    for (example_index, example) in enumerate(examples):
        features = example.convert_single_example(
            is_training=is_training,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_program_length=max_program_length,
            op_list=op_list,
            op_list_size=op_list_size,
            const_list=const_list,
            const_list_size=const_list_size,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token)

        # 对于生成的每个特征，设置其唯一ID和示例索引，然后将其添加到res列表中
        for feature in features:
            feature.unique_id = unique_id
            feature.example_index = example_index
            res.append(feature)
            unique_id += 1

    return res


RawResult = collections.namedtuple(
    "RawResult",
    "unique_id logits loss")


def compute_prog_from_logits(logits, max_program_length, example,
                             template=None):
    # 根据模型的输出（logits）计算预测的程序ID序列和对应的损失值
    pred_prog_ids = [] # 存储预测的程序ID序列
    op_stack = []
    loss = 0 # 累计损失
    for cur_step in range(max_program_length): # 程序ID序列的最大长度
        # 获取当前步骤的logits
        cur_logits = logits[cur_step]
        # 计算softmax（_compute_softmax(cur_logits)），将logits转换为概率分布
        cur_pred_softmax = _compute_softmax(cur_logits)
        # 找出概率最高的预测标记
        cur_pred_token = np.argmax(cur_logits)
        # 累加损失：使用预测标记的概率的对数的负值进行累加
        loss -= np.log(cur_pred_softmax[cur_pred_token])
        # 将预测的标记（cur_pred_token）添加到 pred_prog_ids
        pred_prog_ids.append(cur_pred_token)
        if cur_pred_token == 0:
            # 如果预测的标记是 0（通常代表特定的结束标记或填充标记），则结束循环
            break
    return pred_prog_ids, loss


def compute_predictions(all_examples, all_features, all_results, n_best_size,
                        max_program_length, tokenizer, op_list, op_list_size,
                        const_list, const_list_size):
    # 将模型的原始输出（logits）转换为更直观、可理解的预测结果
    # defaultdict 为字典提供一个默认值工厂函数，这个函数会在你尝试访问字典中不存在的键时被调用，以提供默认值
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        # 将每个样本（example）和它对应的特征（feature）关联起来
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        # 将每个结果（通过唯一ID标识）与其对应的logits关联起来
        unique_id_to_result[result.unique_id] = result

    # 定义 _PrelimPrediction 用于存储初步预测结果
    # 存储每个初步预测的相关信息。包含了 feature_index（特征索引）和 logits（模型输出的原始预测得分）
    # 下划线 _ 作为变量名或类名的前缀通常用于指示这个变量或类是“内部使用的”或“非公开的”
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", [
            "feature_index", "logits"
        ])

    all_predictions = collections.OrderedDict()
    all_predictions["pred_programs"] = collections.OrderedDict()
    all_predictions["ref_programs"] = collections.OrderedDict()
    all_nbest = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        # 获取该样本对应的所有特征
        features = example_index_to_features[example_index]
        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            # 对于每个特征，获取对应的结果（result），提取logits，创建一个 _PrelimPrediction 对象，并将其添加到 prelim_predictions 列表中
            result = unique_id_to_result[feature.unique_id]
            logits = result.logits
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=feature_index,
                    logits=logits))
        # _NbestPrediction 用于存储最佳预测结果
        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", "options answer program_ids program")

        nbest = []
        for pred in prelim_predictions:
            # 对于每个样本的初步预测
            if len(nbest) >= n_best_size:
                # 如果已经收集了足够的最佳预测（根据 n_best_size 确定），则停止收集
                break
            program = example.program
            # 根据logits计算程序ID和损失
            pred_prog_ids, loss = compute_prog_from_logits(pred.logits,
                                                           max_program_length,
                                                           example)
            # 将程序ID转换为可读的程序
            pred_prog = finqa_utils.indices_to_prog(pred_prog_ids,
                                                    example.numbers,
                                                    example.number_indices,
                                                    conf.max_seq_length,
                                                    op_list, op_list_size,
                                                    const_list, const_list_size
                                                    )
            # 创建一个 _NbestPrediction 对象，并将其添加到 nbest 列表中
            nbest.append(
                _NbestPrediction(
                    options=example.options,
                    answer=example.answer,
                    program_ids=pred_prog_ids,
                    program=pred_prog))

        assert len(nbest) >= 1

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            # 对于每个样本的最佳预测，构建一个详细的输出字典，包含了ID、选项、参考答案、预测程序、参考程序等信息
            # OrderedDict 确保元素的顺序和它们被添加到字典中的顺序一致, 即使在更改现有键的值或删除并重新添加键后，它也会保持原有的顺序
            output = collections.OrderedDict()
            output["id"] = example.id
            output["options"] = entry.options
            output["ref_answer"] = entry.answer
            output["pred_prog"] = [str(prog) for prog in entry.program]
            output["ref_prog"] = example.program
            output["question_tokens"] = example.question_tokens
            output["numbers"] = example.numbers
            output["number_indices"] = example.number_indices
            nbest_json.append(output)

        # 确保每个样本至少有一个最佳预测
        assert len(nbest_json) >= 1

        # 将每个样本的预测程序和参考程序分别添加到 all_predictions 字典中
        all_predictions["pred_programs"][example_index] = nbest_json[0]["pred_prog"]
        all_predictions["ref_programs"][example_index] = nbest_json[0]["ref_prog"]
        # 将每个样本的最佳预测详细信息添加到 all_nbest 字典中
        all_nbest[example_index] = nbest_json

    return all_predictions, all_nbest


def write_predictions(all_predictions, output_prediction_file):
    # 将预测结果以 JSON 格式写入到一个文件中
    """ 
    all_predictions: 包含所有预测结果的字典或其他数据结构。
    output_prediction_file: 要写入预测结果的文件路径
    """

    with open(output_prediction_file, "w") as writer:
        # "w" 模式表示如果文件存在则覆盖，如果不存在则创建新文件
        # 将 all_predictions 转换为格式化的 JSON 字符串
        # indent=4 参数是为了使 JSON 数据具有可读性，每个层级的缩进为 4 个空格
        writer.write(json.dumps(all_predictions, indent=4) + "\n")


class DataLoader:
    # 通过这个类，可以方便地管理数据的加载和批量提供
    def __init__(self, is_training, data, reserved_token_size, batch_size=64, shuffle=True):
        """
        Main dataloader
        """
        self.data = data
        self.batch_size = batch_size
        self.is_training = is_training
        self.data_size = len(data)
        self.reserved_token_size = reserved_token_size
        self.num_batches = int(self.data_size / batch_size) if self.data_size % batch_size == 0 \
            else int(self.data_size / batch_size) + 1
        if shuffle:
            self.shuffle_all_data()
        self.count = 0

    # __iter__ 和 __next__ 方法使得 DataLoader 类成为一个迭代器
    def __iter__(self):
        # 返回迭代器本身
        return self

    def __next__(self):
        # drop last batch
        if self.is_training:
            bound = self.num_batches - 1
        else:
            bound = self.num_batches
        # 如果当前计数小于总批次数，返回下一个批次的数据；否则，引发 StopIteration 异常
        if self.count < bound:
            return self.get_batch()
        else:
            raise StopIteration

    def __len__(self):
        # 返回总批次数，即一次完整的数据遍历会产生的批次数量
        return self.num_batches

    def reset(self):
        # 重置计数器，并且根据需要重新打乱数据。这通常在一个新的周期开始时调用
        self.count = 0
        self.shuffle_all_data()

    def shuffle_all_data(self):
        # 对数据进行随机打乱。这有助于避免模型在训练过程中过度适应数据的特定顺序，从而提高模型的泛化能力
        random.shuffle(self.data)
        return

    def get_batch(self):
        # 根据当前计数器的值计算出要获取的数据批次的起始和结束索引
        start_index = self.count * self.batch_size
        end_index = min((self.count + 1) * self.batch_size, self.data_size)

        self.count += 1
        # print (self.count)

        # 从数据集中抽取对应索引的数据，构造一个包含各种信息的字典 batch_data（如唯一ID、输入ID、输入掩码、问题文本、程序ID等）
        batch_data = {"unique_id": [],
                      "example_index": [],
                      "tokens": [],
                      "question": [],
                      "input_ids": [],
                      "input_mask": [],
                      "option_mask": [],
                      "segment_ids": [],
                      "options": [],
                      "answer": [],
                      "program": [],
                      "program_ids": [],
                      "program_weight": [],
                      "program_mask": []}
        for each_data in self.data[start_index: end_index]:

            batch_data["option_mask"].append(each_data.option_mask)
            batch_data["input_mask"].append(each_data.input_mask)

            batch_data["unique_id"].append(each_data.unique_id)
            batch_data["example_index"].append(each_data.example_index)
            batch_data["tokens"].append(each_data.tokens)
            batch_data["question"].append(each_data.question)
            batch_data["input_ids"].append(each_data.input_ids)
            batch_data["segment_ids"].append(each_data.segment_ids)
            batch_data["options"].append(each_data.options)
            batch_data["answer"].append(each_data.answer)
            batch_data["program"].append(each_data.program)
            batch_data["program_ids"].append(each_data.program_ids)
            batch_data["program_weight"].append(each_data.program_weight)
            batch_data["program_mask"].append(each_data.program_mask)

        return batch_data


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def str_to_num(text):
    # 将文本字符串转换为数值

    text = text.replace(",", "") # 去除文本中的逗号（,）
    # 使用 try...except 结构尝试将文本转换为浮点数，如果失败，则返回 "n/a"
    try:
        num = float(text)
    except ValueError:
        # 检查文本是否包含特殊格式的数值，如百分比、常量等
        if "%" in text:
            text = text.replace("%", "")
            try:
                num = float(text)
                num = num / 100.0
            except ValueError:
                num = "n/a"
        elif "const" in text:
            # 去除前缀 "const_"，并检查是否是特殊常数（如 "const_m1" 表示 -1）。然后转换为浮点数
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            num = float(text)
        else:
            # 如果以上条件都不满足，设置 num 为 "n/a"，表示无法转换
            num = "n/a"
    return num


def process_row(row_in):
    # 处理表格中的一行数据，将其中的文本转换为数值
    # row_in: 输入的一行数据，通常是一个包含数字和可能的货币符号等的字符串列表

    row_out = [] # 存储处理后的数值
    invalid_flag = 0 # 指示行中的数据是否有效。如果数据无效，这个标志将被设置为 1

    for num in row_in:
        # 去除可能存在的美元符号（$）和空白字符
        num = num.replace("$", "").strip()
        # 如果文本包含括号（如表示负数的括号），仅保留括号前的部分
        num = num.split("(")[0].strip()

        # 将文本转换为数值
        num = str_to_num(num)

        # 如果转换结果为 "n/a"，即无法转换为数值，则设置 invalid_flag 为 1 并退出循环
        if num == "n/a":
            invalid_flag = 1
            break

        # 如果转换成功，将转换后的数值添加到 row_out
        row_out.append(num)

    # 如果 invalid_flag 被设置为 1，表示行中有无效数据，函数返回 "n/a"
    if invalid_flag:
        return "n/a"

    # 如果所有数据都有效，返回处理后的数值列表 row_out
    return row_out


def reprog_to_seq(prog_in, is_gold):
    # 将递归定义的程序（以列表形式表示）转换为线性序列的程序
    """
    prog_in 是输入的程序列表
    is_gold 是一个布尔值，指示输入的程序是否是参考（正确）程序 
    """

    st = [] # 作为一个栈来处理程序的嵌套结构
    res = [] # 存储转换后的程序

    # 使用 try...except 来捕捉处理过程中可能发生的异常（例如，如果栈 st 中的元素不够弹出）
    try:
        # 初始化 num 计数器，用于生成占位符
        num = 0
        # 遍历输入的程序 prog_in 中的每个标记 tok
        for tok in prog_in:
            # 如果 tok 不是右括号 )，将其推入栈 st
            if tok != ")":
                st.append(tok)
            else:
                # 初始化一个列表 this_step_vec 来存储当前步骤的操作和操作数
                this_step_vec = [")"]
                for _ in range(3):
                    # 从栈中弹出三个元素（操作数、操作符、操作数）并将它们添加到 this_step_vec，再加上右括号 )
                    this_step_vec.append(st[-1])
                    st = st[:-1]
                # 将 this_step_vec 的元素按相反的顺序添加到结果列表 res，实现操作数和操作符的逆序排列
                res.extend(this_step_vec[::-1])
                # 在栈 st 中添加一个占位符（如 "#0", "#1"），代表当前操作的结果，同时 num 计数器加一
                st.append("#" + str(num))
                num += 1
    except:
        if is_gold:
            # 如果输入程序是参考程序（is_gold 为真）并且发生异常，则抛出 ValueError
            raise ValueError

    return res


def eval_program(program, table):
    # 计算以特定格式表示的程序的数值结果

    # 指示程序是否有效。如果程序在任何地方无效，这个标志将被设置为 1
    invalid_flag = 0
    this_res = "n/a"

    try:
        # 去除程序末尾的 'EOF'
        program = program[:-1]  
        # 检查程序的结构，确保操作符和括号正确放置
        for ind, token in enumerate(program):
            if ind % 4 == 0:
                if token.strip("(") not in all_ops:
                    return 1, "n/a"
            if (ind + 1) % 4 == 0:
                if token != ")":
                    return 1, "n/a"

        # 将程序转换为一个由 "|" 分隔的字符串，并以 ")" 为分隔符分割成步骤
        program = "|".join(program)
        steps = program.split(")")[:-1]

        res_dict = {} # 存储每个步骤的结果

        for ind, step in enumerate(steps):
            step = step.strip() # 去除首位空格

            if len(step.split("(")) > 2:
                invalid_flag = 1
                break
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()

            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()

            if op == "add" or op == "subtract" or op == "multiply" or op == "divide" or op == "exp" or op == "greater":

                # 对于需要数值的操作数，如果它是之前步骤的结果（即以 "#" 开头），则从 res_dict 中获取相应的结果；否则，尝试将其转换为数值
                if "#" in arg1:
                    arg1 = res_dict[int(arg1.replace("#", ""))]
                else:
                    arg1 = str_to_num(arg1)
                    if arg1 == "n/a":
                        invalid_flag = 1
                        break

                if "#" in arg2:
                    arg2 = res_dict[int(arg2.replace("#", ""))]
                else:
                    arg2 = str_to_num(arg2)
                    if arg2 == "n/a":
                        invalid_flag = 1
                        break

                if op == "add":
                    this_res = arg1 + arg2
                elif op == "subtract":
                    this_res = arg1 - arg2
                elif op == "multiply":
                    this_res = arg1 * arg2
                elif op == "divide":
                    this_res = arg1 / arg2
                elif op == "exp":
                    this_res = arg1 ** arg2
                elif op == "greater":
                    this_res = "yes" if arg1 > arg2 else "no"

                # 执行操作，并将结果存储在 res_dict 中，键为步骤的索引
                res_dict[ind] = this_res

            # 如果操作涉及到表格数据（如 table_max），则从 table 中提取相应的行，并执行相应的聚合操作
            elif "table" in op:
                table_dict = {}
                for row in table:
                    table_dict[row[0]] = row[1:]

                if "#" in arg1:
                    arg1 = res_dict[int(arg1.replace("#", ""))]
                else:
                    if arg1 not in table_dict:
                        invalid_flag = 1
                        break

                    cal_row = table_dict[arg1]
                    num_row = process_row(cal_row)

                if num_row == "n/a":
                    invalid_flag = 1
                    break
                # 如果操作涉及到表格数据（如 table_max），则从 table 中提取相应的行，并执行相应的聚合操作
                if op == "table_max":
                    this_res = max(num_row)
                elif op == "table_min":
                    this_res = min(num_row)
                elif op == "table_sum":
                    this_res = sum(num_row)
                elif op == "table_average":
                    this_res = sum(num_row) / len(num_row)

                # 执行操作，并将结果存储在 res_dict 中，键为步骤的索引
                res_dict[ind] = this_res
        if this_res != "yes" and this_res != "no" and this_res != "n/a":

            this_res = round(this_res, 5) # 四舍五入保留五位小数

    except:
        # 如果在处理过程中遇到异常，将 invalid_flag 设置为 1
        invalid_flag = 1

    # 如果 invalid_flag 为 1，则表明程序在执行过程中遇到了问题；this_res 包含了程序的最终执行结果或 "n/a"
    return invalid_flag, this_res


def equal_program(program1, program2):
    '''
    比较两个符号程序（program1 和 program2）是否等价
    如果program2在结构或符号映射上与program1不一致，函数将返回False
    program1: gold
    program2: pred
    '''

    # 字典，用于将程序中的符号映射到标准格式
    sym_map = {}

    program1 = program1[:-1]  # 移除程序末尾的EOF
    program1 = "|".join(program1)
    steps = program1.split(")")[:-1] # 将program1分割成独立的步骤

    # invalid_flag 和 sym_ind: 分别用于标记无效的情况和符号索引
    invalid_flag = 0
    sym_ind = 0
    # step_dict_1: 字典，用于存储program1的步骤
    step_dict_1 = {}

    # symbolic map
    for ind, step in enumerate(steps):

        step = step.strip()

        assert len(step.split("(")) <= 2

        # 提取操作符（op）和参数（args）
        op = step.split("(")[0].strip("|").strip()
        args = step.split("(")[1].strip("|").strip()

        arg1 = args.split("|")[0].strip()
        arg2 = args.split("|")[1].strip()

        step_dict_1[ind] = step

        # 如果操作涉及到“table”，则会在sym_map中为其创建一个新的符号表示（如果之前未映射过）
        if "table" in op:
            if step not in sym_map:
                sym_map[step] = "a" + str(sym_ind)
                sym_ind += 1

        else:
            # 如果参数不是以#开头，也会为其在sym_map中创建或确认映射
            if "#" not in arg1:
                if arg1 not in sym_map:
                    sym_map[arg1] = "a" + str(sym_ind)
                    sym_ind += 1

            if "#" not in arg2:
                if arg2 not in sym_map:
                    sym_map[arg2] = "a" + str(sym_ind)
                    sym_ind += 1

    # check program 2
    step_dict_2 = {}
    try:
        program2 = program2[:-1]  # remove EOF
        # check structure
        for ind, token in enumerate(program2):
            # 验证program2的每四个token的结构是否符合预期（操作符后跟三个参数，参数之间由"|"分隔）
            if ind % 4 == 0:
                if token.strip("(") not in all_ops:
                    print("structure error")
                    return False
            # 如果结构有误，打印错误信息并返回False
            if (ind + 1) % 4 == 0:
                if token != ")":
                    print("structure error")
                    return False

        program2 = "|".join(program2)
        steps = program2.split(")")[:-1]

        # 对于每个步骤，提取操作符和参数
        for ind, step in enumerate(steps):
            step = step.strip()

            if len(step.split("(")) > 2:
                return False
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()

            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()

            step_dict_2[ind] = step

            # 如果操作涉及到“table”，验证sym_map中是否存在对应映射
            if "table" in op:
                if step not in sym_map:
                    return False

            else:
                if "#" not in arg1:
                    if arg1 not in sym_map:
                        return False
                else:
                    # 验证它们的索引是否合理（即不超过当前步骤的索引）
                    if int(arg1.strip("#")) >= ind:
                        return False

                if "#" not in arg2:
                    if arg2 not in sym_map:
                        return False
                else:
                    if int(arg2.strip("#")) >= ind:
                        return False
    except:
        return False

    def symbol_recur(step, step_dict):

        step = step.strip()
        op = step.split("(")[0].strip("|").strip()
        args = step.split("(")[1].strip("|").strip()

        arg1 = args.split("|")[0].strip()
        arg2 = args.split("|")[1].strip()

        if "table" in op:
            # as var
            return sym_map[step]

        if "#" in arg1:
            arg1_ind = int(arg1.replace("#", ""))
            arg1_part = symbol_recur(step_dict[arg1_ind], step_dict)
        else:
            arg1_part = sym_map[arg1]

        if "#" in arg2:
            arg2_ind = int(arg2.replace("#", ""))
            arg2_part = symbol_recur(step_dict[arg2_ind], step_dict)
        else:
            arg2_part = sym_map[arg2]

        if op == "add":
            return "( " + arg1_part + " + " + arg2_part + " )"
        elif op == "subtract":
            return "( " + arg1_part + " - " + arg2_part + " )"
        elif op == "multiply":
            return "( " + arg1_part + " * " + arg2_part + " )"
        elif op == "divide":
            return "( " + arg1_part + " / " + arg2_part + " )"
        elif op == "exp":
            return "( " + arg1_part + " ** " + arg2_part + " )"
        elif op == "greater":
            return "( " + arg1_part + " > " + arg2_part + " )"

    # # derive symbolic program 1
    steps = program1.split(")")[:-1]
    sym_prog1 = symbol_recur(steps[-1], step_dict_1)
    sym_prog1 = simplify(sym_prog1, evaluate=False)

    try:
        # derive symbolic program 2
        steps = program2.split(")")[:-1]
        sym_prog2 = symbol_recur(steps[-1], step_dict_2)
        sym_prog2 = simplify(sym_prog2, evaluate=False)
    except:
        return False

    return sym_prog1 == sym_prog2


def evaluate_result(json_in, json_ori, all_res_file, error_file, program_mode):
    # 评估模型在执行（exe）和程序（prog）准确率
    
    correct = 0

    with open(json_in) as f_in:
        # 预测结果文件
        data = json.load(f_in)

    with open(json_ori) as f_in:
        # 原始数据文件
        data_ori = json.load(f_in)

    data_dict = {}
    # 将原始数据转换为以 ID 为键的字典 data_dict，便于后续按 ID 快速检索数据
    for each_data in data_ori:
        # 如果条件不成立（即条件为假），则会触发 AssertionError 异常。如果条件成立（即条件为真），程序会正常继续执行，不会有任何提示或错误
        assert each_data["id"] not in data_dict
        data_dict[each_data["id"]] = each_data

    exe_correct = 0 # 统计执行准确的数量
    prog_correct = 0 # 程序准确的数量

    res_list = [] # 收集错误的结果
    all_res_list = [] # 所有的结果

    for tmp in data:
        each_data = data[tmp][0]
        each_id = each_data["id"]

        each_ori_data = data_dict[each_id]

        table = each_ori_data["table"]
        gold_res = each_ori_data["qa"]["exe_ans"]

        # 获取预测的程序（pred）和参考程序（gold）
        pred = each_data["pred_prog"]
        gold = each_data["ref_prog"]

        # 根据 program_mode 对预测和参考程序进行必要的处理（例如，处理嵌套程序和去除 'EOF'）
        if program_mode == "nest":
            if pred[-1] == "EOF":
                pred = pred[:-1]
            pred = reprog_to_seq(pred, is_gold=False)
            pred += ["EOF"]
            gold = gold[:-1]
            gold = reprog_to_seq(gold, is_gold=True)
            gold += ["EOF"]

        # 使用 eval_program 函数评估预测的程序，获取执行结果 exe_res 和有效性标志 invalid_flag
        invalid_flag, exe_res = eval_program(pred, table)

        if invalid_flag == 0:
            # 如果执行有效（invalid_flag == 0）
            if exe_res == gold_res:
                # 如果执行结果与参考执行结果相同
                exe_correct += 1

            if equal_program(gold, pred):
                # 判断预测的程序与参考程序是否相同，如果相同，prog_correct 计数增加
                if exe_res != gold_res:
                    print(each_id)
                    print(gold)
                    print(pred)
                    print(gold_res)
                    print(exe_res)
                    print(each_ori_data["id"])
                assert exe_res == gold_res
                prog_correct += 1
                if "".join(gold) != "".join(pred):
                    print(each_id)
                    print(gold)
                    print(pred)
                    print(gold_res)
                    print(exe_res)
                    print(each_ori_data["id"])

        each_ori_data["qa"]["predicted"] = pred

        # 如果执行结果不正确，将该记录添加到 res_list；
        if exe_res != gold_res:
            res_list.append(each_ori_data)
        # 无论执行结果是否正确，都将记录添加到 all_res_list
        all_res_list.append(each_ori_data)

    # 计算执行准确率（exe_acc）和程序准确率（prog_acc），即正确计数除以数据总量
    exe_acc = float(exe_correct) / len(data)
    prog_acc = float(prog_correct) / len(data)

    print("All: ", len(data))
    print("Correct: ", correct)
    print("Exe acc: ", exe_acc)
    print("Prog acc: ", prog_acc)

    # 将错误的结果和所有的结果分别保存到指定的文件 error_file 和 all_res_file
    with open(error_file, "w") as f:
        json.dump(res_list, f, indent=4)

    with open(all_res_file, "w") as f:
        json.dump(all_res_list, f, indent=4)

    return exe_acc, prog_acc


if __name__ == '__main__':

    root = "your_root_path"
    our_data = root + "dataset/"
