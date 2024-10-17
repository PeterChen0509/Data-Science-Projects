
import argparse
import collections
import json
import os
import re
import string
import sys
import random


def remove_space(text_in):
    # 从输入字符串中移除所有多余的空格，只保留单词之间的一个空格
    res = []

    for tmp in text_in.split(" "):
        if tmp != "":
            # 不是空字符串, 则将 tmp 添加到 res 列表中
            res.append(tmp)
    # 将 res 列表中的单词用一个空格连接起来
    return " ".join(res)


def table_row_to_text(header, row):
    # 将表格的一行数据转换为文本描述形式
    # header（表格的表头行，包含列的名称）
    # row（表格中的具体一行数据）
    
    res = ""
    
    if header[0]:
        # 表头的第一列有名称
        res += (header[0] + " ")

    for head, cell in zip(header[1:], row[1:]):
        # 对于每一对表头和单元格数据，函数使用模板 the [第一列的值] of [列名] is [单元格的值] 来生成一个描述性的短句
        res += ("the " + row[0] + " of " + head + " is " + cell + " ; ")
    
    res = remove_space(res)
    return res.strip()