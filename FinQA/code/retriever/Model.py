import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from config import parameters as conf

if conf.pretrained_model == "bert":
    from transformers import BertModel
elif conf.pretrained_model == "roberta":
    from transformers import RobertaModel


class Bert_model(nn.Module):

    def __init__(self, hidden_size, dropout_rate):

        super(Bert_model, self).__init__()

        self.hidden_size = hidden_size

        if conf.pretrained_model == "bert":
            # .from_pretrained 是一个类方法，用于加载预训练的模型权重
            # conf.model_size 指定了预训练模型的类型或大小
            # cache_dir=conf.cache_dir 指定了模型和权重下载后存储的位置
            self.bert = BertModel.from_pretrained(
                conf.model_size, cache_dir=conf.cache_dir)
        elif conf.pretrained_model == "roberta":
            self.bert = RobertaModel.from_pretrained(
                conf.model_size, cache_dir=conf.cache_dir)

        self.cls_prj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.cls_dropout = nn.Dropout(dropout_rate)

        self.cls_final = nn.Linear(hidden_size, 2, bias=True)

    def forward(self, is_training, input_ids, input_mask, segment_ids, device):
        # input_ids、input_mask、segment_ids 是BERT模型的输入，分别代表输入的token的id、attention mask（用于区分padding和非padding部分）以及token类型id（用于区分两个句子）
        
        bert_outputs = self.bert(
            input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)

        # 最后一层的隐藏状态
        bert_sequence_output = bert_outputs.last_hidden_state

        # 从 bert_sequence_output 中提取每个序列的第一个token的表示（即CLS token的表示）
        bert_pooled_output = bert_sequence_output[:, 0, :]

        pooled_output = self.cls_prj(bert_pooled_output)
        pooled_output = self.cls_dropout(pooled_output)

        logits = self.cls_final(pooled_output)

        return logits
