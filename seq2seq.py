#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   seq2seq.py
@Time    :   2020/05/22 14:19:18
@Author  :   Shushan An 
@Version :   1.0
@Contact :   shushan_an@cbit.com.cn
@Desc    :   实现seq2seq的对话模型
'''
from __future__ import absolute_import  # 绝对引入
from __future__ import division         # 精确除法
from __future__ import print_function   # 输出的标准
from __future__ import unicode_literals  # 字符串都是unicode编码

import torch
from torch.jit import script, trace     # 用来解决部署问题
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

corpus_name = "cornell_movie-dialogs_corpus"
corpus = os.path.join("data", corpus_name)


def printLines(file, n=10):
    """用来展示数据集的部分数据"""
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)


printLines(os.path.join(corpus, 'movie_lines.txt'))


def loadLines(fileName, fields):
    """将每一行都分割成字典存储"""
    # Splits each line of the file into a dictionary of fields(lineID, characterID, movieID, character, text)
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines

# Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
def loadConversations(fileName, lines, fields):
    """把文本信息整合到对话集合中"""
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # 将string转换成list convObj['utteranceIDs'] = "['L1','L2'...]"
            utterance_id_pattern = re.compile('L[0-9]+')
            lineIds = utterance_id_pattern.findall(convObj['utteranceIDs'])
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations

# 从对话中提取句子对
def extractSentencePairs(conversations):
    """从对话集合中提取出来文本对话"""
    qa_pairs = []
    for conversation in conversations:
        for i in range(len(conversation["lines"]) - 1):
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs



def data_processs():
    """数据预处理并保存对话数据"""
    # 定义新文件的路径
    datafile = os.path.join(corpus, 'formatted_movie_lines.txt')

    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # 初始化lines字典，对话集合，属性值
    lines = {}
    conversations = []
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    # 加载字典和处理对话集合
    print("\nProcessing corpus...")
    lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
    print("\nLoading conversations...")
    conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                    lines, MOVIE_CONVERSATIONS_FIELDS)
    # 写入到csv文件
    print("\nWriting newly formatted file..")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter= delimiter, lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)

    # 打印一些数据显示
    print('\nSample lines from file:')
    printLines(datafile)

if __name__ == "__main__":
    data_processs()