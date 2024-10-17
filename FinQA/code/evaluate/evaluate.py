#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for evaluate results
Usage:
python evaluate.py predicted_file_name test_file_name
"""
import time
import os
import sys
import shutil
import io
import re
import json
import copy
import random
import collections
import math
import numpy as np
from tqdm import tqdm
from sympy import simplify


all_ops = ["add", "subtract", "multiply", "divide", "exp", "greater", "table_max", \
"table_min", "table_sum", "table_average"]

def str_to_num(text):
    
    text = text.replace(",", "")
    try:
        num = float(text)
    except ValueError:
        if "%" in text:
            text = text.replace("%", "")
            try:
                num = float(text)
                num = num / 100.0
            except ValueError:
                num = "n/a"
        elif "const" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            num = float(text)
        else:
            num = "n/a"
    return num

def process_row(row_in):
    # 处理一个包含数值的列表（row_in），并将其转换为一个更规范的数值列表（row_out），同时检测和处理无效或不可处理的数据
    
    row_out = [] # 存储处理过的数值
    invalid_flag = 0 # 初始化为0，用于标记行中是否存在无效数据
    
    for num in row_in:
        # 移除num中的美元符号($)和前后的空格
        num = num.replace("$", "").strip()
        # 从num中取出第一个左括号(()之前的部分，然后移除前后的空格
        num = num.split("(")[0].strip()
        # 将处理后的字符串转换为数值
        num = str_to_num(num)
        
        if num == "n/a":
            # 当前的num无效或不可转换为数值，将invalid_flag设置为1，并退出循环
            invalid_flag = 1
            break
        
        row_out.append(num)
        
    if invalid_flag:
        return "n/a"
    
    return row_out


def eval_program(program, table):
    # 接收两个参数：program（一个代表程序的分词列表）和 table（一个表格数据）
    # 这个函数的目的是计算和返回由 program 参数定义的程序的数值结果
    '''
    calculate the numerical results of the program
    '''

    # 标记程序在执行过程中是否遇到无效或不可解析的情况
    invalid_flag = 0
    # 存储程序的计算结果，默认值为 "n/a"
    this_res = "n/a"
    
    try:
        program = program[:-1] # 移除末尾的 'EOF' 标记
        # 检查程序的结构，确保操作符 (all_ops 中的元素) 和括号出现的位置符合预期的模式。如果不符合，则立即返回 1, "n/a"，表示程序结构无效
        for ind, token in enumerate(program):
            if ind % 4 == 0:
                if token.strip("(") not in all_ops:
                    return 1, "n/a"
            if (ind + 1) % 4 == 0:
                if token != ")":
                    return 1, "n/a"


        # 通过 program.split(")")[:-1] 将程序分解成独立的步骤（steps），每个步骤代表一个操作
        program = "|".join(program)
        steps = program.split(")")[:-1]
        
        # 存储每个步骤的计算结果
        res_dict = {}
        
        # print(program)
        
        # 遍历每个步骤 step
        for ind, step in enumerate(steps):
            step = step.strip()
            
            if len(step.split("(")) > 2:
                invalid_flag = 1
                break
            # 分解操作符 op 和操作数 args
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()
            
            # print(args)
            # print(op)
            
            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()
            
            # 根据操作符的类型，对操作数进行适当的计算。支持的操作包括加、减、乘、除、指数和比较大小
            if op == "add" or op == "subtract" or op == "multiply" or op == "divide" or op == "exp" or op == "greater":
                
                # 如果操作数是之前步骤的结果（标记为 "#"），则从 res_dict 中获取相应的值
                if "#" in arg1:
                    arg1 = res_dict[int(arg1.replace("#", ""))]
                # 否则，尝试将操作数转换为数字
                else:
                    # print(arg1)
                    arg1 = str_to_num(arg1)
                    # 如果遇到任何问题（例如操作数无效或 str_to_num 返回 "n/a"），则将 invalid_flag 设置为 1 并退出循环
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

                    
                # print("ind: ", ind)
                # print(this_res)
                res_dict[ind] = this_res


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
                # 对于特殊的 table 相关操作（table_max, table_min, table_sum, table_average），需要从 table 参数中提取和处理数据
                if op == "table_max":
                    this_res = max(num_row)
                elif op == "table_min":
                    this_res = min(num_row)
                elif op == "table_sum":
                    this_res = sum(num_row)
                elif op == "table_average":
                    this_res = sum(num_row) / len(num_row)
                    
                # this_res = round(this_res, 5)

                res_dict[ind] = this_res

            # print(this_res)

        # 如果 this_res 不是 "yes"、"no" 或 "n/a"（即是数字），则对其进行四舍五入到小数点后五位
        if this_res != "yes" and this_res != "no" and this_res != "n/a":
            # print(this_res)
            this_res = round(this_res, 5)

    except:
        # 如果在执行过程中发生任何异常，将 invalid_flag 设置为 1
        invalid_flag = 1
        
    # 返回 invalid_flag 和 this_res，分别表示程序是否有效以及程序的计算结果
    return invalid_flag, this_res


def equal_program(program1, program2):
    # 验证program2是否在结构和符号映射上与program1一致
    '''
    symbolic program if equal
    program1: gold
    program2: pred
    '''
    
    # 存储程序中符号的映射
    sym_map = {}
    
    program1 = program1[:-1] # remove EOF
    # program1被转换为以|为分隔符的字符串，并以)为分隔符分割成步骤列表steps
    program1 = "|".join(program1)
    steps = program1.split(")")[:-1]
    
    invalid_flag = 0
    sym_ind = 0
    step_dict_1 = {} # 存储program1中的步骤
    
    # 遍历steps，处理每个步骤
    for ind, step in enumerate(steps):
        
        step = step.strip()

        assert len(step.split("(")) <= 2

        # 拆分步骤为操作符（op）和参数（args）
        op = step.split("(")[0].strip("|").strip()
        args = step.split("(")[1].strip("|").strip()
        
        arg1 = args.split("|")[0].strip()
        arg2 = args.split("|")[1].strip()
        
        step_dict_1[ind] = step

        # 如果操作涉及表格，或者参数不是以#开头的数字（可能表示某种引用或索引），则在sym_map中创建或确认符号映射
        if "table" in op:
            if step not in sym_map:
                sym_map[step] = "a" + str(sym_ind)
                sym_ind += 1
        # 符号映射是将实际的参数或操作映射为抽象的符号（如"a0", "a1"等）
        else:
            if "#" not in arg1:
                if arg1 not in sym_map:
                    sym_map[arg1] = "a" + str(sym_ind)
                    sym_ind += 1
                    
            if "#" not in arg2:
                if arg2 not in sym_map:
                    sym_map[arg2] = "a" + str(sym_ind)
                    sym_ind += 1


    # check program 2
    step_dict_2 = {} # 存储program2中的步骤
    try:
        program2 = program2[:-1] # remove EOF
        # 检查program2的结构，确保每个操作符后有三个参数，第四个元素是)
        for ind, token in enumerate(program2):
            if ind % 4 == 0:
                if token.strip("(") not in all_ops:
                    print("structure error")
                    return False
            if (ind + 1) % 4 == 0:
                if token != ")":
                    print("structure error")
                    return False

        program2 = "|".join(program2)
        steps = program2.split(")")[:-1]
        
        for ind, step in enumerate(steps):
            step = step.strip()
            
            if len(step.split("(")) > 2:
                return False
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()
            
            # print(args)
            # print(op)
            
            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()
            
            step_dict_2[ind] = step

            if "table" in op:
                # 验证参数是否在sym_map中有合法的映射
                if step not in sym_map:
                    return False
                    
            else:
                if "#" not in arg1:
                    if arg1 not in sym_map:
                        return False
                else:
                    # 是否是有效的引用（以#开头的数字且数字小于步骤的索引）
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
        
        # print(op)
        # print(arg1)
        # print(arg2)
        
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
    # print(program1)
    steps = program1.split(")")[:-1]
    # print(steps)
    # print(steps)
    # print(sym_map)
    sym_prog1 = symbol_recur(steps[-1], step_dict_1)
    sym_prog1 = simplify(sym_prog1, evaluate=False)
    # print("########")
    # print(sym_prog1)
    
    try:
        # derive symbolic program 2
        steps = program2.split(")")[:-1]
        sym_prog2 = symbol_recur(steps[-1], step_dict_2)
        sym_prog2 = simplify(sym_prog2, evaluate=False)
        # print(sym_prog2)
    except:
        return False

    return sym_prog1 == sym_prog2


def program_tokenization(original_program):
    # 接收一个名为 original_program 的字符串作为输入，并返回一个分词后的列表
    # 根据逗号和空格进行分割，生成一个列表
    original_program = original_program.split(', ')
    # 存储分词后的结果
    program = []
    for tok in original_program:
        # 在循环中，使用 cur_tok 字符串来暂存当前处理的单词或符号
        cur_tok = ''
        for c in tok:
            # 对于初始分割得到的每个片段 tok，代码逐个字符进行处理
            # 在处理每个字符 c 时，代码通过几个条件判断来决定如何处理当前字符，以及是否需要将 cur_tok 添加到 program 列表中
            if c == ')':
                if cur_tok != '':
                    program.append(cur_tok)
                    cur_tok = ''
            cur_tok += c
            # 如果当前字符 c 是左括号 ( 或右括号 )，则将 cur_tok（此时为 ( 或 )）添加到 program 列表，并清空 cur_tok
            if c in ['(', ')']:
                program.append(cur_tok)
                cur_tok = ''
        # 处理完当前片段 tok 的所有字符后，如果 cur_tok 不为空（即还有未处理完的单词或符号），则将其添加到 program 列表
        if cur_tok != '':
            program.append(cur_tok)
    # 在处理完所有输入片段后，代码通过 program.append('EOF') 添加了一个 'EOF' 标记，可能是为了标识程序的结束
    program.append('EOF')
    # 以列表的形式返回这些分词和符号
    return program



def evaluate_result(json_in, json_ori):
    # 评估一个程序生成的结果与一个标准结果之间的一致性
    # json_in是待评估的结果文件，json_ori是标准（参考）结果文件
    '''
    execution acc
    program acc
    '''
    correct = 0
    
    with open(json_in) as f_in:
        data = json.load(f_in)
        
    with open(json_ori) as f_in:
        data_ori = json.load(f_in)
    
    # data_dict用于存储原始数据的字典，以便可以通过ID快速访问
    data_dict = {}
    for each_data in data_ori:
        # 每个ID是唯一的（不应该在data_dict中有重复的ID）
        assert each_data["id"] not in data_dict
        data_dict[each_data["id"]] = each_data
        
    # exe_correct和prog_correct用于跟踪执行正确和程序正确的数量
    exe_correct = 0
    prog_correct = 0

    res_list = []
    all_res_list = []
    
    # 对于data中的每个项
    for each_data in data:
        # 获取该项的ID，并从data_dict中找到对应的原始数据
        each_id = each_data["id"]
        each_ori_data = data_dict[each_id]
        
        # 提取表格数据(table)和正确的执行结果(gold_res)
        table = each_ori_data["table"]
        gold_res = each_ori_data["qa"]["exe_ans"]
        
        # 获取预测程序(pred)和标准程序(gold)
        pred = each_data["predicted"]
        gold = program_tokenization(each_ori_data["qa"]["program"])

        # print("#########")
        # print(pred)
        # print(gold)

        # if program_mode == "nest":
        #     if pred[-1] == "EOF":
        #         pred = pred[:-1]
        #     pred = reprog_to_seq(pred, is_gold=False)
        #     pred += ["EOF"]
        #     gold = gold[:-1]
        #     gold = reprog_to_seq(gold, is_gold=True)
        #     gold += ["EOF"]
        
        # print("\n")
        # print("########")
        
        # 执行预测程序，获取执行结果(exe_res)和有效标志(invalid_flag)
        invalid_flag, exe_res = eval_program(pred, table)

        # print(invalid_flag)
        # print(exe_res)
        
        # 如果invalid_flag为0（表示程序有效），则检查执行结果是否与标准结果相同。如果相同，增加exe_correct计数
        if invalid_flag == 0:
            if exe_res == gold_res:
                exe_correct += 1
                
        # else:
        #     if "".join(gold) == "".join(pred):
        #         print(each_id)
        #         print(gold)
        #         print(pred)
        #         print(gold_res)
        #         print(exe_res)
        #         print(each_ori_data["id"])
                
        
        # 使用equal_program函数比较预测程序和标准程序是否等价
        if equal_program(gold, pred):
            # assert exe_res == gold_res
            if exe_res != gold_res:
                print(each_id)
                print(gold)
                print(pred)
                print(gold_res)
                print(exe_res)
                print(each_ori_data["id"])
            # 如果等价，增加prog_correct计数，并且断言执行结果必须与标准执行结果相同
            assert exe_res == gold_res
            prog_correct += 1
            if "".join(gold) != "".join(pred):
                print(each_id)
                print(gold)
                print(pred)
                print(gold_res)
                print(exe_res)
                print(each_ori_data["id"])

        # if "".join(gold) == "".join(pred):
        #     if not equal_program(gold, pred):
        #         print(each_id)
        #         print(gold)
        #         print(pred)
        #         print(gold_res)
        #         print(exe_res)
        #         print(each_ori_data["id"])
        #     prog_correct += 1

        each_ori_data["qa"]["predicted"] = pred

        # 如果执行结果与标准结果不同，将这些结果添加到res_list中
        if exe_res != gold_res:
            res_list.append(each_ori_data)
        # 将所有结果添加到all_res_list中
        all_res_list.append(each_ori_data)
            
    # 计算执行精度(exe_acc)和程序精度(prog_acc)
    exe_acc = float(exe_correct) / len(data)
    prog_acc = float(prog_correct) / len(data)
            
    # 打印总数、执行精度和程序精度
    print("All: ", len(data))
    print("Exe acc: ", exe_acc)
    print("Prog acc: ", prog_acc)

    # with open(error_file, "w") as f:
    #     json.dump(res_list, f, indent=4)

    # with open(all_res_file, "w") as f:
    #     json.dump(all_res_list, f, indent=4)

    return exe_acc, prog_acc




if __name__ == '__main__':


    json_in = sys.argv[1]
    json_ori = sys.argv[2]

    evaluate_result(json_in, json_ori)
    
  