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

_SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)

sys.path.insert(0, '../utils/')
from general_utils import table_row_to_text


def str_to_num(text):
    text = text.replace(",", "")
    try:
        num = int(text)
    except ValueError:
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
    prog_indices = []
    for i, token in enumerate(prog):
        if token in op_list:
            prog_indices.append(op_list.index(token))
        elif token in const_list:
            prog_indices.append(op_list_size + const_list.index(token))
        else:
            if token in numbers:
                cur_num_idx = numbers.index(token)
            else:
                cur_num_idx = -1
                for num_idx, num in enumerate(numbers):
                    if str_to_num(num) == str_to_num(token):
                        cur_num_idx = num_idx
                        break
            assert cur_num_idx != -1
            prog_indices.append(op_list_size + const_list_size +
                                number_indices[cur_num_idx])
    return prog_indices


def indices_to_prog(program_indices, numbers, number_indices, max_seq_length,
                    op_list, op_list_size, const_list, const_list_size):
    prog = []
    for i, prog_id in enumerate(program_indices):
        if prog_id < op_list_size:
            prog.append(op_list[prog_id])
        elif prog_id < op_list_size + const_list_size:
            prog.append(const_list[prog_id - op_list_size])
        else:
            prog.append(numbers[number_indices.index(prog_id - op_list_size
                                                     - const_list_size)])
    return prog


class MathQAExample(
        collections.namedtuple(
            "MathQAExample",
            "filename_id question all_positive \
            pre_text post_text table"
        )):
    # MathQAExample 类代表了一个数学问题及其相关信息的结构
    def convert_single_example(self, *args, **kwargs):
        return convert_single_mathqa_example(self, *args, **kwargs)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 filename_id,
                 retrieve_ind,
                 tokens,
                 input_ids,
                 segment_ids,
                 input_mask,
                 label):

        self.filename_id = filename_id
        self.retrieve_ind = retrieve_ind
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label


def tokenize(tokenizer, text, apply_basic_tokenization=False):
    # 对输入文本进行分词处理

    if conf.pretrained_model in ["bert", "finbert"]:
        # 检测特殊标记 - 不含空格且被方括号或尖括号括起来的文本
        # BERT类模型中表示某些特殊的控制字符，如开始、结束或分隔符号（例如，[CLS]、[SEP]）
        _SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)
    # 对于BERT和FinBERT模型，特殊标记由方括号括起来；而对于RoBERTa和Longformer模型，特殊标记由尖括号括起来
    elif conf.pretrained_model in ["roberta", "longformer"]:
        _SPECIAL_TOKENS_RE = re.compile(r"^<[^ ]*>$", re.UNICODE)

    tokenize_fn = tokenizer.tokenize
    if apply_basic_tokenization:
        tokenize_fn = tokenizer.basic_tokenizer.tokenize

    tokens = []
    for token in text.split(" "):
        # 按空格分割成单词
        # 看[]里有没有空格、不是以方括号开头和结尾，或者是空的
        if _SPECIAL_TOKENS_RE.match(token):
            # 如果是特殊标记且在分词器的词汇表中
            if token in tokenizer.get_vocab():
                tokens.append(token)
            else:
                # 未知标记
                tokens.append(tokenizer.unk_token)
        else:
            # 不是特殊标记的文本，使用指定的 tokenize_fn 函数进行分词
            tokens.extend(tokenize_fn(token))

    # 返回分词后的文本列表 
    return tokens


def _detokenize(tokens):
    text = " ".join(tokens)

    text = text.replace(" ##", "")
    text = text.replace("##", "")

    text = text.strip()
    text = " ".join(text.split())
    return text


def program_tokenization(original_program):
    original_program = original_program.split(', ')
    program = []
    for tok in original_program:
        cur_tok = ''
        for c in tok:
            if c == ')':
                if cur_tok != '':
                    program.append(cur_tok)
                    cur_tok = ''
            cur_tok += c
            if c in ['(', ')']:
                program.append(cur_tok)
                cur_tok = ''
        if cur_tok != '':
            program.append(cur_tok)
    program.append('EOF')
    return program



def get_tf_idf_query_similarity(allDocs, query):
    """
    vectorizer: TfIdfVectorizer model
    docs_tfidf: tfidf vectors for all docs
    query: query doc

    return: cosine similarity between query and all docs
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(stop_words='english')
    docs_tfidf = vectorizer.fit_transform(allDocs)
    
    query_tfidf = vectorizer.transform([query])
    cosineSimilarities = cosine_similarity(query_tfidf, docs_tfidf).flatten()
    
    # print(cosineSimilarities)
    return cosineSimilarities


def wrap_single_pair(tokenizer, question, context, label, max_seq_length,
                    cls_token, sep_token):
    # 将一对文本（即一个问题和一个上下文）及其关联的标签转换成适合输入到一个基于Transformer模型（如BERT）的机器学习模型的格式
    
    question_tokens = tokenize(tokenizer, question)
    this_gold_tokens = tokenize(tokenizer, context)

    tokens = [cls_token] + question_tokens + [sep_token]
    # 对于序列中的每个分词，分配一个段落ID
    segment_ids = [0] * len(tokens)

    tokens += this_gold_tokens
    # 在这个函数中，所有分词的段落ID都设为0，这在仅有单个句子时是常见的
    segment_ids.extend([0] * len(this_gold_tokens))

    if len(tokens) > max_seq_length:
        # 对序列进行截断，以确保其长度不超过模型能够处理的最大长度
        tokens = tokens[:max_seq_length-1]
        tokens += [sep_token]
        segment_ids = segment_ids[:max_seq_length]

    # 将分词转换成模型能够理解的输入ID
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # 创建一个输入掩码，其中每个输入ID对应的是1，表示模型应该考虑这些分词
    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    
    # 创建一个包含上下文、原始分词序列、输入ID、输入掩码、段落ID和标签的字典
    this_input_feature = {
        "context": context,
        "tokens": tokens,
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "label": label
    }
    
    return this_input_feature

def convert_single_mathqa_example(example, option, is_training, tokenizer, max_seq_length,
                                  cls_token, sep_token):
    # 将一个MathQAExample实例转换为用于检索型任务的特征集
    # 初始化正面和负面特征的列表
    pos_features = []
    features_neg = []
    
    question = example.question
    all_text = example.pre_text + example.post_text

    if is_training:
        for gold_ind in example.all_positive:
            # 遍历所有正答案索引, 为每一个正答案创建特征
            this_gold_sent = example.all_positive[gold_ind]
            # 将问题文本和对应的正答案文本转换为模型的输入特征
            this_input_feature = wrap_single_pair(
                tokenizer, question, this_gold_sent, 1, max_seq_length,
                cls_token, sep_token)

            this_input_feature["filename_id"] = example.filename_id
            this_input_feature["ind"] = gold_ind
            # 将这些特征存储在 pos_features 列表中
            pos_features.append(this_input_feature)
            
        num_pos_pair = len(example.all_positive)
        # 计算负面特征的数量
        num_neg_pair = num_pos_pair * conf.neg_rate
            
        pos_text_ids = []
        pos_table_ids = []
        for gold_ind in example.all_positive:
            if "text" in gold_ind:
                pos_text_ids.append(int(gold_ind.replace("text_", "")))
            elif "table" in gold_ind:
                pos_table_ids.append(int(gold_ind.replace("table_", "")))

        # 计算所有文本和表格的索引数
        all_text_ids = range(len(example.pre_text) + len(example.post_text))
        all_table_ids = range(1, len(example.table))
        
        # all_positive = entry["qa"]["gold_inds"]
        all_negs_size = len(all_text) + len(example.table) - len(example.all_positive)
        if all_negs_size < 0:
            all_negs_size = 0
                    
        # 对于每个负答案，也使用 wrap_single_pair 函数来创建特征，并存储在 features_neg 列表中
        for i in range(len(all_text)):
            if i not in pos_text_ids:
                this_text = all_text[i]
                this_input_feature = wrap_single_pair(
                    tokenizer, example.question, this_text, 0, max_seq_length,
                    cls_token, sep_token)
                this_input_feature["filename_id"] = example.filename_id
                this_input_feature["ind"] = "text_" + str(i)
                features_neg.append(this_input_feature)
            # table      
        for this_table_id in range(len(example.table)):
            if this_table_id not in pos_table_ids:
                this_table_row = example.table[this_table_id]
                this_table_line = table_row_to_text(example.table[0], example.table[this_table_id])
                this_input_feature = wrap_single_pair(
                    tokenizer, example.question, this_table_line, 0, max_seq_length,
                    cls_token, sep_token)
                this_input_feature["filename_id"] = example.filename_id
                this_input_feature["ind"] = "table_" + str(this_table_id)
                features_neg.append(this_input_feature)
            
    else:
        # 测试阶段
        pos_features = []
        features_neg = []
        question = example.question

        # 对于所有文本和表格数据，创建标签为 -1 的特征（表示这些是测试数据，不用于训练）
        for i in range(len(all_text)):
            this_text = all_text[i]
            this_input_feature = wrap_single_pair(
                tokenizer, example.question, this_text, -1, max_seq_length,
                cls_token, sep_token)
            this_input_feature["filename_id"] = example.filename_id
            this_input_feature["ind"] = "text_" + str(i)
            features_neg.append(this_input_feature)
            # table      
        for this_table_id in range(len(example.table)):
            this_table_row = example.table[this_table_id]
            this_table_line = table_row_to_text(example.table[0], example.table[this_table_id])
            this_input_feature = wrap_single_pair(
                tokenizer, example.question, this_table_line, -1, max_seq_length,
                cls_token, sep_token)
            this_input_feature["filename_id"] = example.filename_id
            this_input_feature["ind"] = "table_" + str(this_table_id)
            features_neg.append(this_input_feature)

    # 返回正面和负面特征列表
    return pos_features, features_neg


def read_mathqa_entry(entry, tokenizer):
    # 表示数学问题及其相关数据
    filename_id = entry["id"]
    question = entry["qa"]["question"]
    # 是否是正答案的索引列表
    if "gold_inds" in entry["qa"]:
        all_positive = entry["qa"]["gold_inds"]
    else:
        all_positive = []

    pre_text = entry["pre_text"]
    post_text = entry["post_text"]
    table = entry["table"]

    return MathQAExample(
        filename_id=filename_id,
        question=question,
        all_positive=all_positive,
        pre_text=pre_text,
        post_text=post_text,
        table=table)
