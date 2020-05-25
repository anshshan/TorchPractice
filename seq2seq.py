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
# 定义新文件的路径
datafile = os.path.join(corpus, 'formatted_movie_lines.txt')
# 过滤后问答对保存的文件路径
save_dir = os.path.join("data", "save")


MAX_LENGTH = 10     # 考虑句子的最大长度
MIN_COUNT = 3       # 单词出现的最小次数
# 默认单词标记(Default word tokens)
PAD_token = 0   # 短句分隔标记(Used for padding short sentences)
SOS_token = 1   # 句首标记(Start-of-sentence token)
EOS_token = 2   # 句尾标记(End-of-sentence token)


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
    # datafile = os.path.join(corpus, 'formatted_movie_lines.txt')

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



class Voc():
    """记录单词到索引的映射、索引到单词的反向映射、每个单词的次数及总单词数
    提供了增加词，增加句子以及剪枝的函数
    """
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD
    
    def addSentence(self, sentence):
        '''将一个句子增加到字典中'''
        for word in sentence.split(' '):
            self.addWord(word)
    
    def addWord(self, word):
        '''将一个单词增加到字典中'''
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        '''根据最小词频进行剪枝'''
        # 这里可能存在只能进行一次修剪
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []
        
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        
        print('keep_words {} / {} = {:.4f}'.format(len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)))

        # 重新进行初始化
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)

def unicodeToAscii(s):
    '''将Unicode编码的字符串转换成Ascii编码'''
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    '''转换成小写字符、移除前后空格、删除非字符字母'''
    s = unicodeToAscii(s.lower().strip())
    # 匹配标点符号并增加空格
    s = re.sub(r"([.!?])", r" \1", s)
    # 去除非标点和字母的字符
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    # 将连续空格转换成一个空格
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def readVocs(datafile, corpus_name):
    '''读取问答对，然后返回voc对象'''
    print("Reading lines...")
    # 读取文件并分割到行
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    # 分割每一行到问答对，然后归一化 (Split every line into pairs and normalize)
    pairs = [[normalizeString(s) for s in line.split('\t')] for line in lines]
    voc = Voc(corpus_name)
    return voc, pairs

def filterPair(p):
    """判断问答对是否满足不大于最大句子长度"""
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    """过滤文本长度大于最大句子长度的数据"""
    return [pair for pair in pairs if filterPair(pair)]

def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    # 将过滤后的问答对保存到文件中
    # print("\nWriting newly formatted file..")
    # with open(os.path.join(save_dir, 'filter_move_lines.txt'), 'w', encoding='utf-8') as outputfile:
    #     writer = csv.writer(outputfile, delimiter= '\t', lineterminator='\n')
    #     for pair in pairs:
    #         writer.writerow(pair)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Count words:", voc.num_words)
    return voc, pairs

# 为了加速收敛，我们将少于一定数量的单词删除
def trimRareWords(voc, pairs, MIN_COUNT):
    # 根据最小单词数目进行剪枝
    voc.trim(MIN_COUNT)
    # 过滤问答对
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True

        # 过滤输入句子
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # 过滤输出句子
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break
        
        if keep_input and keep_output:
            keep_pairs.append(pair)
    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs



# 为模型准备数据
def indexesFromSentence(voc, sentence):
    """将句子文本数据转换成索引数据
    params:
        voc: 词典
        sentence: 句子
    """
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def zeroPadding(l, fillvalue=PAD_token):
    """对索引数据进行填充并进行转置
    params:
        l: batch大小的句子索引数据
        filvalue: 填充的字符
    """
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    """将小批量句子索引矩阵转换成0-1的mask矩阵
    params:
        l: batch大小的句子索引数据
        value: 填充的字符
    """
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def inputVar(l, voc):
    """返回填充的输入序列的tensor和长度的tensor
    params:
        l: batch大小的句子数据
        voc: 词典
    """
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

def outputVar(l, voc):
    """返回填充的输出序列tensor，mask矩阵和句子最大长度
    params:
        l: batch大小的句子数据
        voc: 词典
    """
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(index) for index in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

def batch2TrainData(voc, pair_batch):
    """对训练数据进行处理，并返回问答对得tensor数据，输入句子长度的tensor，输出句子的mask矩阵和最大长度"""
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

if __name__ == "__main__":
    # data_processs()
    # Load/Assemble voc and pairs
    voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
    # Print some pairs to validate
    pairs = trimRareWords(voc, pairs, MIN_COUNT)
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair)
    
    # Example for validation
    small_batch_size = 5
    batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches

    print("input_variable:", input_variable)
    print("lengths:", lengths)
    print("target_variable:", target_variable)
    print("mask:", mask)
    print("max_target_len:", max_target_len)