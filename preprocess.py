#!/usr/bin/env python
# encoding: utf-8

#import argparse

#parser = argparse.ArgumentParser(description="Caver preprocessing")
#parser.add_argument("--data", type=str, help="the path to training file")
#parser.add_argument("--destdir", type=str, help="path to store the processed data")
#args = parser.parse_args()
#print(args)

import mmap
from tqdm import tqdm


#############
### COUNTING

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

file_path = "/data_hdd/zhihu/nlpcc2018/zhihu_data_sample.csv"

tags_count = {}
tokens_count = {}

with open(file_path) as trainfile:
    for line in tqdm(trainfile, total=get_num_lines(file_path), desc='Counting'):
        line = line.strip().split("\t")
        line_tags = line[1].split("|")
        line_tokens = line[0].split(" ")
        for tag in line_tags:
            if tag not in tags_count:
                tags_count[tag] = 1
            else:
                tags_count[tag] += 1
        for token in line_tokens:
            if token not in tokens_count:
                tokens_count[token] = 1
            else:
                tokens_count[token] += 1

TAG_LIMIT = 200
tags_count = sorted(tags_count.items(), key=lambda kv: -kv[1])
tags = [tag_count_kv[0] for tag_count_kv in tags_count[:TAG_LIMIT]]
print("| Origin {} tags, truncated to {} ({:.2f}%)".format(len(tags_count), TAG_LIMIT, float(TAG_LIMIT)/len(tags_count)*100))
tags.sort()
tag2index = {}
index2tag = {}

for idx, tag in enumerate(tags):
    tag2index[tag] = idx
    index2tag[idx] = tag

TOKEN_LIMIT = len(tokens_count)
tokens_count = sorted(tokens_count.items(), key=lambda kv: -kv[1])
tokens = [tag_count_kv[0] for tag_count_kv in tokens_count][:TOKEN_LIMIT]
print("| Origin {} tokens, truncated to {} ({:.2f}%)".format(len(tokens_count), TOKEN_LIMIT, float(TOKEN_LIMIT)/len(tokens_count)*100))
tokens.sort()
token2index = {}
index2token = {}

for idx, token in enumerate(tokens):
    token2index[token] = idx
    index2token[idx] = token


#############
### CONVERT

converted_file = []
training_features = []
training_target = []

with open(file_path) as trainfile:
    for line in tqdm(trainfile, total=get_num_lines(file_path), desc="Converting"):
        line = line.strip().split("\t")
        line_tags = line[1].split("|")
        line_tokens = line[0].split(" ")
        # for tag in line_tags:
        # line_tags_idx = [tag2index[tag] for tag in line_tags]
        line_tags_idx = []
        for tag in line_tags:
            if tag in tag2index:
                line_tags_idx.append(tag2index[tag])

        line_tokens_idx = []
        for token in line_tokens:
            if token in token2index:
                line_tokens_idx.append(token2index[token])

        if len(line_tokens_idx) == 0 or len(line_tags_idx) == 0: # sample's all tags are low frequency, skipp this sample
            continue
        else:
            training_features.append(line_tokens_idx)
            training_target.append(line_tags_idx)


converted_file = list(zip(training_features, training_target))
print(len(converted_file))

###########
#### SAVE
import pickle

pickle.dump(tag2index, open("dest_dir/tag2index.p", "wb"))
pickle.dump(index2tag, open("dest_dir/index2tag.p", "wb"))
pickle.dump(token2index, open("dest_dir/token2index.p", "wb"))
pickle.dump(index2token, open("dest_dir/index2token.p", "wb"))

pickle.dump(converted_file, open("dest_dir/train_data.p", "wb"))

##########
#### META
meta_config = {}
meta_config["num_of_tags"] = 200

import json

with open('dest_dir/config.json', 'w') as fp:
    json.dump(meta_config, fp)
