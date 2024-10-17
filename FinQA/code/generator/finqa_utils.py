"""MathQA utils.
"""
import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys
import random
import enum
import six
import copy
from six.moves import map
from six.moves import range
from six.moves import zip

from config import parameters as conf


sys.path.insert(0, '../utils/')
from general_utils import table_row_to_text


def str_to_num(text):
    # 将文本字符串转换为数值（整数或浮点数）
    text = text.replace(",", "")
    try:
        num = int(text)
    except ValueError:
        # 如果转换失败，它会检查字符串是否表示百分比或者是无法识别的文本
        try:
            num = float(text)
        except ValueError:
            if text and text[-1] == "%":
                num = text
            else:
                num = None
    return num


def prog_token_to_indices(prog, numbers, number_indices, max_seq_length,
                          op_list, op_list_size, const_list,
                          const_list_size):
    # 将程序从符号表示（例如操作符、常量、数字）转换为索引表示的过程
    prog_indices = [] # 存储程序中每个标记的索引
    for i, token in enumerate(prog):
        if token in op_list:
            # 检查当前标记 token 是否在操作符列表 op_list 中
            prog_indices.append(op_list.index(token))
            # 如果是，将其索引添加到 prog_indices 列表中
        elif token in const_list:
            # 检查它是否在常量列表 const_list 中
            prog_indices.append(op_list_size + const_list.index(token))
            # 如果是，将其索引加上操作符列表的大小 op_list_size，然后添加到 prog_indices 中。这是因为常量的索引是在操作符索引之后的
        else:
            # 如果标记既不是操作符也不是常量，那么它可能是程序中用到的数字
            if token in numbers:
                # 检查这个数字是否在 numbers 列表中。如果是，获取它的索引 cur_num_idx
                cur_num_idx = numbers.index(token)
            else:
                # 如果不在 numbers 列表中，尝试通过比较数字的值找到它的索引
                cur_num_idx = -1
                for num_idx, num in enumerate(numbers):
                    if str_to_num(num) == str_to_num(token):
                        cur_num_idx = num_idx
                        break
            # 使用断言确保找到了数字的索引
            assert cur_num_idx != -1
            # 将数字的索引加上操作符列表和常量列表的大小，添加到 prog_indices 中。这是因为数字的索引是在操作符和常量之后的
            prog_indices.append(op_list_size + const_list_size +
                                number_indices[cur_num_idx])
    return prog_indices


def indices_to_prog(program_indices, numbers, number_indices, max_seq_length,
                    op_list, op_list_size, const_list, const_list_size):
    """ 
    参数
    program_indices: 程序ID序列，是模型预测的输出。
    numbers: 输入相关的数字列表。
    number_indices: 索引列表，指示哪些程序ID与 numbers 中的数字对应。
    max_seq_length: 序列的最大长度。
    op_list: 操作符列表。
    op_list_size: 操作符列表的大小。
    const_list: 常量列表。
    const_list_size: 常量列表的大小 
    """
    # 将程序的索引列表转换为具体的操作符、常量或输入相关的数字
    prog = [] # 存储转换后的程序
    for i, prog_id in enumerate(program_indices):
        if prog_id < op_list_size:
            # 是一个操作符
            prog.append(op_list[prog_id]) 
        elif prog_id < op_list_size + const_list_size:
            # 是一个常量
            prog.append(const_list[prog_id - op_list_size])
        else:
            # 对应于 numbers 中的某个数字
            prog.append(numbers[number_indices.index(prog_id - op_list_size
                                                     - const_list_size)])
    return prog


class MathQAExample(
    # 创建一个命名元组的类
        collections.namedtuple(
            "MathQAExample",
            "id original_question question_tokens options answer \
            numbers number_indices original_program program"
        )):

    def convert_single_example(self, *args, **kwargs):
        return convert_single_mathqa_example(self, *args, **kwargs)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 question,
                 input_ids,
                 input_mask,
                 option_mask,
                 segment_ids,
                 options,
                 answer=None,
                 program=None,
                 program_ids=None,
                 program_weight=None,
                 program_mask=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.question = question
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.option_mask = option_mask
        self.segment_ids = segment_ids
        self.options = options
        self.answer = answer
        self.program = program
        self.program_ids = program_ids
        self.program_weight = program_weight
        self.program_mask = program_mask


def tokenize(tokenizer, text, apply_basic_tokenization=False):
    # 这个函数特别处理了特殊的标记，并根据配置的模型类型使用不同的正则表达式来识别这些特殊标记

    if conf.pretrained_model in ["bert", "finbert"]:
        # re.compile 将一个字符串编译为正则表达式对象
        _SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)
        # 前缀 r 表示 raw string，告诉 Python 不要处理字符串中的转义字符（例如 \n）
        # ^: 匹配字符串的开始
        # $: 匹配字符串的结束
    elif conf.pretrained_model in ["roberta", "longformer"]:
        _SPECIAL_TOKENS_RE = re.compile(r"^<[^ ]*>$", re.UNICODE)
        # [^ ]*: 匹配除了空格之外的任意字符（[^ ] 表示非空格字符，* 表示零次或多次）
        # 通过使用 re.UNICODE 标志位，这个匹配在 Unicode 字符串中也能正确进行。

    tokenize_fn = tokenizer.tokenize
    if apply_basic_tokenization:
        tokenize_fn = tokenizer.basic_tokenizer.tokenize

    tokens = [] # 存储分词结果
    for token in text.split(" "):
        if _SPECIAL_TOKENS_RE.match(token):
            # 如果当前标记与特殊标记的正则表达式匹配
            if token in tokenizer.get_vocab():
                # 如果该标记在分词器的词汇表中（tokenizer.get_vocab()），直接将该标记添加到 tokens 列表
                tokens.append(token)
            else:
                # 否则，将分词器的未知标记 (tokenizer.unk_token) 添加到 tokens 列表
                tokens.append(tokenizer.unk_token)
        else:
            # 如果当前标记不是特殊标记，使用选择的分词函数（tokenize_fn）对该标记进行分词，然后将分词结果添加到 tokens 列表
            tokens.extend(tokenize_fn(token))

    return tokens


def _detokenize(tokens):#
    text = " ".join(tokens)

    text = text.replace(" ##", "")
    text = text.replace("##", "")

    text = text.strip()
    text = " ".join(text.split())
    return text

# 不是很懂这个函数的作用
def program_tokenization(original_program):
    # 对表示程序的字符串进行分词
    # 按逗号和空格分割，得到一个标记列表。每个标记可能是一个操作符、数值、变量或包含括号的表达式
    original_program = original_program.split(', ')
    program = []
    for tok in original_program:
        # 初始化一个临时字符串 cur_tok 用于构建当前的标记
        cur_tok = ''
        # 进一步遍历 tok 中的每个字符 
        for c in tok:
            if c == ')':
                # 如果当前字符 c 是右括号 )，并且 cur_tok 非空，将 cur_tok 添加到 program 列表中，并重置 cur_tok
                if cur_tok != '':
                    program.append(cur_tok)
                    cur_tok = ''
            cur_tok += c
            if c in ['(', ')']:
                # 如果当前字符 c 是左括号 ( 或右括号 )，将 cur_tok（即 '(' 或 ')'）作为独立的标记添加到 program 列表中，并重置 cur_tok
                program.append(cur_tok)
                cur_tok = ''
        if cur_tok != '':
            # 如果在处理完所有字符后 cur_tok 非空，将其作为最后一个标记添加到 program 列表中
            program.append(cur_tok)
    program.append('EOF') # 添加程序结束标记
    # 原始的程序字符串被转换为模型可以直接处理的标记列表
    return program 


def convert_single_mathqa_example(example, is_training, tokenizer, max_seq_length,
                                  max_program_length, op_list, op_list_size,
                                  const_list, const_list_size,
                                  cls_token, sep_token):
    # 将单个数学问答示例（MathQAExample 实例）转换成一个用于模型输入的特征集（InputFeature）
    features = []
    # 分词
    question_tokens = example.question_tokens
    # 如果问题的分词长度超过了 max_seq_length - 2，则截断问题文本以满足长度限制
    if len(question_tokens) >  max_seq_length - 2:
        print("too long")
        question_tokens = question_tokens[:max_seq_length - 2]
    # 在问题分词的开头和结尾分别添加特殊的 cls_token 和 sep_token
    tokens = [cls_token] + question_tokens + [sep_token]
    # 创建 segment_ids，用于区分不同的句子，这里所有的 segment_ids 都被设为0，因为只有一个句子
    segment_ids = [0] * len(tokens)

    # 使用 tokenizer 将分词后的问题文本转换为模型可以理解的 input_ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # 创建 input_mask，在问题文本中的数字对应的位置上标记为2，其余位置标记为1
    input_mask = [1] * len(input_ids)
    for ind, offset in enumerate(example.number_indices):
        if offset < len(input_mask):
            input_mask[offset] = 2
        else:
            # 如果数字的位置超过了最大长度，那么在训练模式下，这个示例会被丢弃
            if is_training == True:
                # print("\n")
                # print("################")
                # print("number not in input")
                # print(example.original_question)
                # print(tokens)
                # print(len(tokens))
                # print(example.numbers[ind])
                # print(offset)

                # invalid example, drop for training
                return features

            # assert is_training == False


    # 对 input_ids、input_mask 和 segment_ids 进行补齐，确保它们的长度等于 max_seq_length
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)

    # 使用断言确保补齐后的长度正确
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    # 创建 number_mask，用于标记问题文本中数字的位置
    number_mask = [tmp - 1 for tmp in input_mask]
    for ind in range(len(number_mask)):
        if number_mask[ind] < 0:
            number_mask[ind] = 0
    # 创建 option_mask，用于标记可选项（如操作符和常量）的位置
    option_mask = [1, 0, 0, 1] + [1] * (len(op_list) + len(const_list) - 4)
    option_mask = option_mask + number_mask
    option_mask = [float(tmp) for tmp in option_mask]

    for ind in range(len(input_mask)):
        if input_mask[ind] > 1:
            input_mask[ind] = 1

    # 如果提供了程序（program），并且处于训练模式，将程序转换为索引，并创建相应的 program_mask
    # 如果程序长度小于 max_program_length，则进行补齐
    numbers = example.numbers
    number_indices = example.number_indices
    program = example.program
    if program is not None and is_training:
        program_ids = prog_token_to_indices(program, numbers, number_indices,
                                            max_seq_length, op_list, op_list_size,
                                            const_list, const_list_size)
        program_mask = [1] * len(program_ids)
        program_ids = program_ids[:max_program_length]
        program_mask = program_mask[:max_program_length]
        if len(program_ids) < max_program_length:
            padding = [0] * (max_program_length - len(program_ids))
            program_ids.extend(padding)
            program_mask.extend(padding)
    else:
        program = ""
        program_ids = [0] * max_program_length
        program_mask = [0] * max_program_length
    assert len(program_ids) == max_program_length
    assert len(program_mask) == max_program_length
    features.append(
        InputFeatures(
            unique_id=-1,
            example_index=-1,
            tokens=tokens,
            question=example.original_question,
            input_ids=input_ids,
            input_mask=input_mask,
            option_mask=option_mask,
            segment_ids=segment_ids,
            options=example.options,
            answer=example.answer,
            program=program,
            program_ids=program_ids,
            program_weight=1.0,
            program_mask=program_mask))
    return features


def read_mathqa_entry(entry, tokenizer):
    # 处理MathQA数据集中的单个条目，并将其转换为方便模型处理的格式
    
    # 提取问题文本 question 和唯一标识符 this_id
    question = entry["qa"]["question"]
    this_id = entry["id"]
    context = ""

    # 根据配置的检索模式（conf.retrieve_mode），上下文 context 会以不同的方式被构建
    if conf.retrieve_mode == "single":
        # "single": 串联 entry["qa"]["model_input"] 中的句子
        for ind, each_sent in entry["qa"]["model_input"]:
            context += each_sent
            context += " "
    elif conf.retrieve_mode == "slide":
        # "slide": 从正面窗口（pos_windows）中随机选择一个或从负面窗口（neg_windows）中选择第一个
        if len(entry["qa"]["pos_windows"]) > 0:
            context = random.choice(entry["qa"]["pos_windows"])[0]
        else:
            context = entry["qa"]["neg_windows"][0][0]
    elif conf.retrieve_mode == "gold":
        # "gold": 串联所有金标准指示的上下文
        for each_con in entry["qa"]["gold_inds"]:
            context += entry["qa"]["gold_inds"][each_con]
            context += " "

    elif conf.retrieve_mode == "none":
        # "none": 没有检索器，使用 entry["pre_text"], entry["post_text"], 和 table_text
        table = entry["table"]
        table_text = ""
        for row in table[1:]:
            this_sent = table_row_to_text(table[0], row)
            table_text += this_sent

        context = " ".join(entry["pre_text"]) + " " + " ".join(entry["post_text"]) + " " + table_text

    context = context.strip()
    # 清理上下文，移除连续的点和星号
    context = context.replace(". . . . . .", "")
    context = context.replace("* * * * * *", "")
        
    # 将问题和上下文合并，中间使用分隔符 sep_token
    original_question = question + " " + tokenizer.sep_token + " " + context.strip()

    if "exe_ans" in entry["qa"]:
        options = entry["qa"]["exe_ans"]
    else:
        options = None

    original_question_tokens = original_question.split(' ')

    numbers = []
    number_indices = []
    question_tokens = []
    for i, tok in enumerate(original_question_tokens):
        # 遍历问题中的每个单词，使用 str_to_num 函数尝试将其转换为数字
        num = str_to_num(tok)
        if num is not None:
            # 如果转换成功，将数字及其在问题中的位置添加到相应的列表中
            numbers.append(tok)
            number_indices.append(len(question_tokens))
            if tok[0] == '.':
                numbers.append(str(str_to_num(tok[1:])))
                number_indices.append(len(question_tokens) + 1)
        tok_proc = tokenize(tokenizer, tok)
        question_tokens.extend(tok_proc)

    if "exe_ans" in entry["qa"]:
        # 如果条目包含执行答案 exe_ans，则提取到 answer，否则它们被设置为 None
        answer = entry["qa"]["exe_ans"]
    else:
        answer = None

    # table headers
    for row in entry["table"]:
        # 遍历表格的每一行
        tok = row[0]  # tok 被赋值为当前行的第一个元素，即表头
        if tok and tok in original_question:
            numbers.append(tok)
            # 找出表头在原始问题文本中的位置索引
            tok_index = original_question.index(tok)
            # 获取位于表头之前的所有文本（即从问题开头到表头位置之前的部分）
            prev_tokens = original_question[:tok_index]
            # 使用 tokenize 函数对位于表头之前的文本进行分词，计算这部分文本的分词数量，并将这个数量（加1）作为表头的索引添加到 number_indices 列表中。这样做是为了在后续处理中能够准确地定位到每个表头的位置
            number_indices.append(len(tokenize(tokenizer, prev_tokens)) + 1)

    # 根据配置的程序模式（conf.program_mode），提取和处理程序
    if conf.program_mode == "seq":
        if 'program' in entry["qa"]:
            original_program = entry["qa"]['program']
            program = program_tokenization(original_program)
        else:
            program = None
            original_program = None
            
    elif conf.program_mode == "nest":
        if 'program_re' in entry["qa"]:
            original_program = entry["qa"]['program_re']
            program = program_tokenization(original_program)
        else:
            program = None
            original_program = None
        
    else:
        program = None
        original_program = None

    return MathQAExample(
        id=this_id,
        original_question=original_question,
        question_tokens=question_tokens,
        options=options,
        answer=answer,
        numbers=numbers,
        number_indices=number_indices,
        original_program=original_program,
        program=program)
