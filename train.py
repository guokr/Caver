#!/usr/bin/env python
# encoding: utf-8

import argparse
import os
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import dill as pickle

from torchtext.data import Field, TabularDataset, BucketIterator
from caver.model import LSTM, FastText, CNN
from caver.utils import MiniBatchWrapper
from caver.evaluator import Evaluator

parser = argparse.ArgumentParser(description="Caver training")
parser.add_argument("--model", type=str, choices=["fastText", "LSTM", "CNN"],
                    help="choose the model", default="LSTM")
parser.add_argument("--input_data_dir", type=str, help="data dir")
parser.add_argument("--train_filename", type=str, default="train.tsv")
parser.add_argument("--valid_filename", type=str, default="valid.tsv")
parser.add_argument("--epoch", type=int, help="number of epoches", default=10)
parser.add_argument("--batch_size", type=int, default=16,
                    help="batch size for each GPU card")
parser.add_argument("--output_data_dir", type=str, default="processed_data")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                    help="dir for checkpoints saving")
parser.add_argument("--master_device", type=int, default=0)
parser.add_argument("--multi_gpu", action="store_true")

args = parser.parse_args()

def check_args():
    print("=============== Command Line Tools Args ===============")
    for arg, value in vars(args).items():
        print("{:>20} <===> {:<20}".format(arg, value))
    print("=======================================================")

    status = True
    if not os.path.exists(os.path.join(args.input_data_dir, args.train_filename)):
        status = False
        print("|ERROR| train file doesn't exist")

    if not os.path.exists(os.path.join(args.input_data_dir, args.valid_filename)):
        status = False
        print("|ERROR| valid file doesn't exist")

    if torch.cuda.is_available() == False:
        status = False
        print("|ERROR| Currently we dont support CPU training")

    if torch.cuda.device_count() == 1 and args.multi_gpu == True:
        status = False
        print("|ERROR| We only detected {} GPU".format(torch.cuda.device_count()))

    if os.path.isdir(args.checkpoint_dir) and len(os.listdir(args.checkpoint_dir)) != 0:
        status = False
        # exist but not empty
        print("|ERROR| save dir must be empty")

    if not os.path.isdir(args.checkpoint_dir):
        print("|NOTE| Doesn't find the save dir, we will create a default one for you")
        os.mkdir(args.checkpoint_dir)

    if not os.path.isdir(args.output_data_dir):
        print("|NOTE| Doesn't find the output data dir, we will create a default one for you")
        os.mkdir(args.output_data_dir)

    return status


def preprocess():
    print("|LOGGING| Processing tokens and datasets...")
    tokenize = lambda x: x.split()
    TEXT = Field(sequential=True, tokenize=tokenize, lower=True, batch_first=True)
    LABEL = Field(sequential=False, use_vocab=False)
    from itertools import islice
    columns = []

    with open(os.path.join(args.input_data_dir, args.train_filename)) as f_input:
        for row in islice(f_input, 0, 1):
            for _ in row.split("\t"):
                columns.append(_)

    x_feature = columns[0]
    y_feature = columns[1:]

    tv_datafields = [(x_feature, TEXT)]

    for col in y_feature:
        tv_datafields.append((col, LABEL))

    train_data, valid_data = TabularDataset.splits(path=args.input_data_dir,
                                                   format="tsv",
                                                   train=args.train_filename,
                                                   validation=args.valid_filename,
                                                   skip_header=True,
                                                   fields=tv_datafields)

    TEXT.build_vocab(train_data)

    pickle.dump(TEXT, open(os.path.join(args.output_data_dir, "TEXT.p"), "wb"))
    pickle.dump(y_feature, open(os.path.join(args.output_data_dir, "y_feature.p"), "wb"))
    print("FEATURE: ", len(y_feature))

    ############# pre-process done

    return train_data, valid_data, TEXT, x_feature, y_feature

def train(train_data, valid_data, TEXT, x_feature, y_feature):
    print("| Building batches...")
    device = torch.device("cuda:{}".format(args.master_device))
    # build dataloader
    train_iter, valid_iter = BucketIterator.splits((train_data, valid_data),
                                batch_size=args.batch_size * torch.cuda.device_count(),
                                device=device,
                                sort_key=lambda x: len(x.tokens),
                                sort_within_batch=True)

    train_dataloader = MiniBatchWrapper(train_iter, x_feature, y_feature)
    valid_dataloader = MiniBatchWrapper(valid_iter, x_feature, y_feature)

    print("| Building model...")

    if args.model == "LSTM":
        model = LSTM(hidden_dim=300,
                     embedding_dim=256,
                     vocab_size=len(TEXT.vocab),
                     label_num=len(y_feature),
                     device=device,
                     layer_num=2)
    elif args.model == "fastText":
        model = FastText(vocab_size=len(TEXT.vocab),
                         embedding_dim=256,
                         label_num=len(y_feature))
    elif args.model == "CNN":
        model = CNN(vocab_size=len(TEXT.vocab),
                    embedding_dim=256,
                    filter_num=100,
                    filter_sizes=[2,3,4],
                    label_num=len(y_feature))

    model_args = model.get_args()

    if torch.cuda.device_count() > 1 and args.multi_gpu is True:
        print("Training on {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    valid_loss_history = []

    print("| Training...")

    for epoch in range(1, args.epoch+1):
        train_step(model, train_dataloader, optimizer, criterion, epoch)
        valid_step(model, model_args, valid_dataloader, criterion, valid_loss_history, epoch)

from evaluate import evaluation

def train_step(model, train_data, opt, criterion, epoch):
    evaluator = Evaluator(criterion)
    model.train()
    tqdm_progress = tqdm.tqdm(train_data, desc="| Training epoch {}/{}".format(epoch, args.epoch))
    for x, y in tqdm_progress:
        opt.zero_grad()
        preds = model(x)
        recall, precision, f_score = evaluation(preds, y)
        ev = evaluator.evaluate(preds, y)
        opt.step()

        tqdm_progress.set_postfix({"Loss":"{:.4f}".format(ev[0]),
                                   "Recall": "{:.4f}".format(ev[1]),
                                   "Precsion": "{:.4f}".format(ev[2]),
                                   "F_Score": "{:.4f}".format(ev[3])
                                   })
        # tqdm_progress.set_postfix({"Recall":"{:.4f}".format(recall)})
    # calculate the validation loss for this epoch


def valid_step(model, model_args, valid_data, criterion, valid_loss_history, epoch):
    evaluator = Evaluator(criterion)
    model.eval()
    tqdm_progress = tqdm.tqdm(valid_data, desc="| Validating epoch {}/{}".format(epoch, args.epoch))
    for x, y in tqdm_progress:
        if x.size(1)<4:
            print("ok minibatch skiped")
            continue

        preds = model(x)
        ev = evaluator.evaluate(preds, y, mode="eval")
        tqdm_progress.set_postfix({"Loss":"{:.4f}".format(ev[0]),
                                   "Recall": "{:.4f}".format(ev[1]),
                                   "Precsion": "{:.4f}".format(ev[2]),
                                   "F_Score": "{:.4f}".format(ev[3])
                                   })
    torch.save({"model_type": args.model,
                "model_args": model_args,
                "model_state_dict": model.state_dict()},
               os.path.join(args.checkpoint_dir, "checkpoint_{}.pt".format(epoch)))

    if len(valid_loss_history) == 0 or ev[0] < valid_loss_history[0]:
        print("| Better checkpoint found !")
        torch.save({"model_type": args.model,
                    "model_args": model_args,
                    "model_state_dict": model.state_dict()},
                   os.path.join(args.checkpoint_dir, "checkpoint_best.pt"))
        valid_loss_history.append(ev[0])
        valid_loss_history.sort()


if __name__ == "__main__":
    status = check_args()
    if status == True:
        tr, vl, TEXT, x_feature, y_feature = preprocess()
        train(tr, vl, TEXT, x_feature, y_feature)
