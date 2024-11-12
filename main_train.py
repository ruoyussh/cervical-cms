# -*- coding: utf-8 -*-
import os
import random
random.seed(0)

import torch
torch.manual_seed(0)

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score as auc_roc
from sklearn.metrics import average_precision_score
import pickle
from arguments import parser
from utils import *
from data_utils import *
from torch.utils.tensorboard import SummaryWriter
from MLP_model import *
from pathlib import Path


def evaluate(args, model, val_X, val_Y, trainval, auc_best=0, pr_best=0):
    model.eval()
    avg_predictions = []

    test = MyDataset(val_X, isTest=True)
    loader_ts = DataLoader(test, batch_size=1, shuffle=False)

    gt_list = []
    for idx_ts, tsbag in enumerate(loader_ts):
        tsbag = tsbag.float()
        tsbag = Variable(tsbag).type(torch.cuda.FloatTensor)
        scores = get_scores(model.forward(tsbag[0]))
        scores = float(scores)
        avg_predictions.append(scores)
        gt_list.append(int(val_Y[idx_ts]))

    auc_avg = auc_roc(gt_list, avg_predictions)

    pr_avg = average_precision_score(gt_list, avg_predictions)
    print(trainval, 'AUC=', auc_avg)
    print(trainval, 'AP=', pr_avg)

    if trainval == 'val':
        if pr_best < pr_avg:
            torch.save(model.state_dict(), os.path.join(args.current_output_dir, 'MLP_pr.pth'))
            pr_best = pr_avg
        if auc_best < auc_avg:
            torch.save(model.state_dict(), os.path.join(args.current_output_dir, 'MLP_auc.pth'))
            auc_best = auc_avg
            
    return {
        "pr_best": pr_best,
        "auc_best": auc_best,
        "pr": pr_avg,
        "auc": auc_avg
    }


def get_scores(predictions):
    score_mean = torch.mean(predictions)
    overall_score = score_mean
    return overall_score.unsqueeze(0)


def main():
    args = parser.parse_args()
    from pprint import pprint
    pprint(args)
    args.feature_path_dict = str_to_dict(args.feature_dict_str)

    with open(args.split_path, 'rb') as handle:
        split = pickle.load(handle)

    path_list = Path(list(args.feature_path_dict.values())[0]).parts
    for item in path_list:
        if 'x_' in item and item.split('x_')[0].isdigit():
            data_info = item
            break
    encoder_info = path_list[-1]
    model_info = 'Triplet'
    save_folder_name = f'{args.exp_id}={data_info}={encoder_info}={model_info}'
    args.output_dir = os.path.join(args.output_dir, save_folder_name)
    os.makedirs(args.output_dir, exist_ok=True)

    for split_id in split:
        args.split_output_dir = os.path.join(args.output_dir, f"split_{split_id}")
        os.makedirs(args.split_output_dir, exist_ok=True)
        args.current_trainval_split = split[split_id]

        eval_dict = run(args)
        if eval_dict is None:
            continue


def run(args):
    args.current_output_dir = os.path.join(args.split_output_dir, args.exp_name)
    FINISH_FLAG_FILE = f'{args.current_output_dir}/FINISH_FLAG'
    if os.path.exists(FINISH_FLAG_FILE):
        return None

    os.makedirs(args.current_output_dir, exist_ok=True)

    tensorboard_writer = SummaryWriter(log_dir=args.current_output_dir)

    train_dict = args.current_trainval_split['train']
    val_dict = args.current_trainval_split['val']

    train_lst = list(train_dict.keys())
    val_lst = list(val_dict.keys())
    patient_labels = {**train_dict, **val_dict}

    train_X, train_Y, _ = read_features(train_lst, patient_labels, args.feature_path_dict)
    val_X, val_Y, _ = read_features(val_lst,  patient_labels, args.feature_path_dict)
    
    pos_bags = train_X[train_Y == 1]
    neg_bags = train_X[train_Y == 0]

    print('pos bags:', len(pos_bags))
    print('neg bags:', len(neg_bags))

    feat_len = pos_bags[0][0].shape[0]
    print('feat_len:', feat_len)


    mlp = Net(d=feat_len, hidden_d=int(2*feat_len), out_dim=1, dropout=args.dropout)
    mlp.cuda()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    all_losses = []
    auc_best = -1
    pr_best = -1

    n_iter = 0
    for e in range(args.epoch):
        print('Epoch:', e)
        l = 0.0
        torch.autograd.set_detect_anomaly(True)

        loss_sum = 0
        # loss_iter = 0
        optimizer.zero_grad()

        train_set = MyTripletDataset(train_X, train_Y, model=mlp, args=args, 
                                     sample_max_num=args.sample_max_num)
        if len(train_set) == 0:
            break
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

        mlp.train()
        for idx_batch, triplet_pair in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Training {e}'):
            pbag = triplet_pair['pos'].float()
            pbag = Variable(pbag, requires_grad=True).type(torch.cuda.FloatTensor)
            p_score = get_scores(mlp.forward(pbag[0]))

            nbag = triplet_pair['neg'].float()
            nbag = Variable(nbag, requires_grad=True).type(torch.cuda.FloatTensor)
            n_score = get_scores(mlp.forward(nbag[0]))

            ancbag = triplet_pair['anc'].float()
            ancbag = Variable(ancbag, requires_grad=True).type(torch.cuda.FloatTensor)
            anc_score = get_scores(mlp.forward(ancbag[0]))

            z = np.array([0.0])
            loss_p_n = torch.max(Variable(torch.from_numpy(z), requires_grad=True).type(torch.cuda.FloatTensor),
                                (n_score - p_score + args.margin_inter))
            loss_p_anc = torch.max(Variable(torch.from_numpy(z), requires_grad=True).type(torch.cuda.FloatTensor),
                                (anc_score - p_score + args.margin_inter))
            loss_n_anc = torch.max(Variable(torch.from_numpy(z), requires_grad=True).type(torch.cuda.FloatTensor),
                                (torch.norm(anc_score - n_score, p=2) - args.margin_intra))
            loss = loss_p_n + loss_p_anc + loss_n_anc
            
            loss_sum += loss
            loss_sum.backward(retain_graph=False)
            nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad()
            loss_sum = 0
            n_iter += 1

            if idx_batch % args.eval_step == 0 and idx_batch != 0:
                mlp.eval()
                eval_dict = evaluate(args, mlp, val_X, val_Y, "val", auc_best=auc_best, pr_best=pr_best)
                auc_best=eval_dict['auc_best']
                pr_best=eval_dict['pr_best']
                mlp.train()

        scheduler.step()
        print('Epoch-{0} lr: {1}'.format(e, optimizer.param_groups[0]['lr']))

        mlp.eval()
        eval_dict = evaluate(args, mlp, val_X, val_Y, "val", auc_best=auc_best, pr_best=pr_best)
        auc_best=eval_dict['auc_best']
        pr_best=eval_dict['pr_best']
        tensorboard_writer.add_scalar('auc', eval_dict['auc'], global_step=n_iter)
        tensorboard_writer.add_scalar('pr', eval_dict['pr'], global_step=n_iter)

        all_losses.append(l)

    eval_dict = evaluate(args, mlp, val_X, val_Y, "val", auc_best=auc_best, pr_best=pr_best)
    file = open(FINISH_FLAG_FILE, '+a')
    file.close()
    return eval_dict


if __name__ == '__main__':
    main()
