#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import json
import argparse
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import torch
import torch.optim as optim
from utility import Datasets
import models


def setup_seed(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", default="0",
                        type=str, help="which gpu to use")
    parser.add_argument("-d", "--dataset", default="spotify",
                        type=str, help="which dataset to use")
    parser.add_argument("-m", "--model", default="",
                        type=str, help="which model to use")
    parser.add_argument("-i", "--info", default="", type=str,
                        help="any auxilary info that will be appended to the log file name")
    parser.add_argument("-l", "--lr", default=1e-3,
                        type=float, help="Learning rate")
    parser.add_argument("-r", "--reg", default=1e-5,
                        type=float, help="weight decay")

    parser.add_argument("--item_augment", default="NA", type=str,
                        help="NA (No Augmentation), FD (Factor-wise Dropout), FN (Factor-wise Noise), MD (Modality-wise Noise)")
    parser.add_argument("--bundle_ratio", default=0.5, type=float,
                        help="the ratio of reserved items in a bundle, [0, 0.25, 0,5, 0.75, 1, 1.25, 1.5, 1.75, 2]")
    parser.add_argument("--bundle_augment", default="ID",
                        type=str, help="ID (Item Dropout), IR (Item Replacement)")
    parser.add_argument("--dropout_rate", default=0.2,
                        type=float, help="item-level dropout")
    parser.add_argument("--noise_weight", default=0.02,
                        type=float, help="item-level noise")
    parser.add_argument("--cl_temp", default=0.2, type=float,
                        help="tau for item-level contrastive learning")
    parser.add_argument("--cl_alpha", default=0, type=float,
                        help="alpha for item-level contrastive learning")
    parser.add_argument("--bundle_cl_temp", default=0.2, type=float,
                        help="tau for bundle-level contrastive learning")
    parser.add_argument("--bundle_cl_alpha", default=0.1, type=float,
                        help="alpha for bundle-level contrastive learning")
    parser.add_argument("--attention", default='', type=str,
                        help="wether to use layernorm or w_v")
    parser.add_argument("--trans_layer", default=1, type=int,
                        help="the number of layers for layernorm")
    parser.add_argument("--num_token", default=200, type=int,
                        help="the number of tokens (items in the bundle)")
    
    parser.add_argument("--seed", default=2023, type=int, help="")
    parser.add_argument("--epoch", default=-1, type=int, help="")

    args = parser.parse_args()
    return args


def main():
    conf = yaml.safe_load(open("./config.yaml"))
    print("load config file done!")

    paras = get_cmd().__dict__
    dataset_name = paras["dataset"]
    conf = conf[dataset_name]
    for p in paras:
        conf[p] = paras[p]

    os.environ['CUDA_VISIBLE_DEVICES'] = conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device

    setup_seed(conf["seed"])

    dataset = Datasets(conf)
    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items

    lr = paras['lr'] if "lr" in paras else conf['lrs'][0]
    l2_reg = paras['reg'] if "reg" in paras else conf['l2_regs'][0]
    embedding_size = paras['embedding_size'] if "embedding_size" in paras else conf['embedding_sizes'][0]
    num_layers = paras['num_layers'] if "num_layers" in paras else conf['num_layerss'][0]

    log_path = "./log/%s/%s" % (conf["dataset"], conf["model"])
    run_path = "./runs/%s/%s" % (conf["dataset"], conf["model"])
    checkpoint_model_path = "./checkpoints/%s/%s/model" % (
        conf["dataset"], conf["model"])
    checkpoint_conf_path = "./checkpoints/%s/%s/conf" % (
        conf["dataset"], conf["model"])
    if not os.path.isdir(run_path):
        os.makedirs(run_path)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir(checkpoint_model_path):
        os.makedirs(checkpoint_model_path)
    if not os.path.isdir(checkpoint_conf_path):
        os.makedirs(checkpoint_conf_path)

    conf["l2_reg"] = l2_reg
    conf["embedding_size"] = embedding_size

    settings = []
    if conf["info"] != "":
        settings += [conf["info"]]

    settings += ["Epoch%d" % (conf['epochs']), str(conf["batch_size_train"]),
                 str(lr), str(l2_reg), str(embedding_size)]

    conf["num_layers"] = num_layers

    setting = "_".join(settings)
    log_path = log_path + "/" + setting
    run_path = run_path + "/" + setting
    checkpoint_model_path = checkpoint_model_path + "/" + setting
    checkpoint_conf_path = checkpoint_conf_path + "/" + setting
    run = SummaryWriter(run_path)
    try:
        model = getattr(models, conf['model'])(
            conf, dataset.graphs, dataset.features).to(device)
    except:
        raise ValueError("Unimplemented model %s" % (conf["model"]))

    with open(log_path, "a") as log:
        log.write(f"{conf}\n")
        print(conf)

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=conf["l2_reg"])
    batch_cnt = len(dataset.train_loader)
    test_interval_bs = int(batch_cnt * conf["test_interval"])

    best_metrics, best_perform = init_best_metrics(conf)
    best_epoch = 0
    setup_seed(conf["seed"])
    num_epoch = conf['epochs'] if conf['epoch'] == -1 else conf["epoch"]
    for epoch in range(num_epoch):
        epoch_anchor = epoch * batch_cnt
        model.train(True)
        pbar = tqdm(enumerate(dataset.train_loader),
                    total=len(dataset.train_loader))
        avg_losses = {}
        for batch_i, batch in pbar:
            model.train(True)
            optimizer.zero_grad()
            batch = [x.to(device) for x in batch]
            batch_anchor = epoch_anchor + batch_i

            losses = model(batch)

            losses['loss'].backward(retain_graph=False)
            optimizer.step()

            for l in losses:
                if l not in avg_losses:
                    avg_losses[l] = [losses[l].detach().cpu().item()]
                else:
                    avg_losses[l].append(losses[l].detach().cpu().item())

            pbar.set_description("epoch: %d, " % (epoch) +
                                 ", ".join([
                                     "%s: %.5f" % (l, losses[l].detach()) for l in losses
                                 ]))

            if (batch_anchor+1) % test_interval_bs == 0:
                metrics = {}
                metrics["val"] = test(model, dataset.val_loader, conf)
                metrics["test"] = test(model, dataset.test_loader, conf)
                best_metrics, best_perform, best_epoch, is_better = log_metrics(
                    conf, model, metrics, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch)

        for l in avg_losses:
            run.add_scalar(l, np.mean(avg_losses[l]), epoch)
        avg_losses = {}


def init_best_metrics(conf):
    best_metrics = {}
    best_metrics["val"] = {}
    best_metrics["test"] = {}
    for key in best_metrics:
        best_metrics[key]["recall"] = {}
        best_metrics[key]["ndcg"] = {}
    for topk in conf['topk']:
        for key, res in best_metrics.items():
            for metric in res:
                best_metrics[key][metric][topk] = 0
    best_perform = {}
    best_perform["val"] = {}
    best_perform["test"] = {}

    return best_metrics, best_perform


def write_log(run, log_path, topk, step, metrics):
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    val_scores = metrics["val"]
    test_scores = metrics["test"]

    for m, val_score in val_scores.items():
        test_score = test_scores[m]
        run.add_scalar("%s_%d/Val" % (m, topk), val_score[topk], step)
        run.add_scalar("%s_%d/Test" % (m, topk), test_score[topk], step)

    val_str = "%s, Top_%d, Val:  recall: %f, ndcg: %f" % (
        curr_time, topk, val_scores["recall"][topk], val_scores["ndcg"][topk])
    test_str = "%s, Top_%d, Test: recall: %f, ndcg: %f" % (
        curr_time, topk, test_scores["recall"][topk], test_scores["ndcg"][topk])

    log = open(log_path, "a")
    log.write("%s\n" % (val_str))
    log.write("%s\n" % (test_str))
    log.close()

    print(val_str)
    print(test_str)


def log_metrics(conf, model, metrics, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch):
    for topk in conf["topk"]:
        write_log(run, log_path, topk, batch_anchor, metrics)

    log = open(log_path, "a")

    topk_ = 20
    print("top%d as the final evaluation standard" % (topk_))
    is_better = False
    if metrics["val"]["recall"][topk_] > best_metrics["val"]["recall"][topk_] and metrics["val"]["ndcg"][topk_] > best_metrics["val"]["ndcg"][topk_]:
        torch.save(model.state_dict(), checkpoint_model_path)
        is_better = True
        dump_conf = dict(conf)
        del dump_conf["device"]
        json.dump(dump_conf, open(checkpoint_conf_path, "w"))
        best_epoch = epoch
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for topk in conf['topk']:
            for key, res in best_metrics.items():
                for metric in res:
                    best_metrics[key][metric][topk] = metrics[key][metric][topk]

            best_perform["test"][topk] = "%s, Best in epoch %d, TOP %d: REC_T=%.5f, NDCG_T=%.5f" % (
                curr_time, best_epoch, topk, best_metrics["test"]["recall"][topk], best_metrics["test"]["ndcg"][topk])
            best_perform["val"][topk] = "%s, Best in epoch %d, TOP %d: REC_V=%.5f, NDCG_V=%.5f" % (
                curr_time, best_epoch, topk, best_metrics["val"]["recall"][topk], best_metrics["val"]["ndcg"][topk])
            print(best_perform["val"][topk])
            print(best_perform["test"][topk])
            log.write(best_perform["val"][topk] + "\n")
            log.write(best_perform["test"][topk] + "\n")

    log.close()

    return best_metrics, best_perform, best_epoch, is_better


def test(model, dataloader, conf):
    tmp_metrics = {}
    for m in ["recall", "ndcg"]:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]

    device = conf["device"]
    model.eval()
    rs = model.propagate()
    pbar = tqdm(dataloader, total=len(dataloader))
    for index, b_i_input, seq_b_i_input, b_i_gt in pbar:
        pred_i = model.evaluate(
            rs, (index.to(device), b_i_input.to(device), seq_b_i_input.to(device)))
        pred_i = pred_i - 1e8 * b_i_input.to(device)  # mask
        tmp_metrics = get_metrics(
            tmp_metrics, b_i_gt.to(device), pred_i, conf["topk"])

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]

    return metrics


def get_metrics(metrics, grd, pred, topks):
    tmp = {"recall": {}, "ndcg": {}}
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(
            pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)

        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        tmp["ndcg"][topk] = get_ndcg(pred, grd, is_hit, topk)

    for m, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for i, x in enumerate(res):
                metrics[m][topk][i] += x

    return metrics


def get_recall(pred, grd, is_hit, topk):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt/(num_pos+epsilon)).sum().item()

    return [nomina, denorm]


def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        hit = hit/torch.log2(torch.arange(2, topk+2,
                             device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, dtype=torch.float).to(device)
        hit[:num_pos] = 1
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1+topk, dtype=torch.float).to(device)
    IDCGs[0] = 1 
    for i in range(1, topk+1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)

    idcg = IDCGs[num_pos]
    ndcg = dcg/idcg.to(device)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()

    return [nomina, denorm]


if __name__ == "__main__":
    main()
