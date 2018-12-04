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

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

tags_count = {}

file_path = "/data_hdd/zhihu/nlpcc2018/zhihu_data_sample.csv"
with open(file_path) as trainfile:
    for line in tqdm(trainfile, total=get_num_lines(file_path)):
        line_tags = line.strip().split("\t")[1].split("|")
        for tag in line_tags:
            if tag not in tags_count:
                tags_count[tag] = 1
            else:
                tags_count[tag] += 1

# x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
tags_count = sorted(tags_count.items(), key=lambda kv: -kv[1])

tags = [tag_count_kv[0] for tag_count_kv in tags_count][:100]

tags.sort()

tag2index = {}
index2tag = {}

for idx, tag in enumerate(tags):
    tag2index[tag] = idx
    index2tag[idx] = tag

import pickle

pickle.dump(tag2index, open("dest_dir/tag2index.p", "wb"))
pickle.dump(index2tag, open("dest_dir/index2tag.p", "wb"))

meta_config = {}

meta_config["num_of_tags"] = 100

import json

with open('dest_dir/config.json', 'w') as fp:
    json.dump(meta_config, fp)
