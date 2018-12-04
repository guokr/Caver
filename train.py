#!/usr/bin/env python
# encoding: utf-8

import argparse
import torch

parser = argparse.ArgumentParser(description="Caver training")
parser.add_argument("--model", type=str, choices=["CNN", "LSTM"],
                    help="choose the model", default="CNN")

args = parser.parse_args()

print(args)
