import torch
import numpy as np

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def print_acc(class_type, top1, top5, n):
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    print(f'{class_type} : top1 Accuracy = {top1}%, top5 Accuracy = {top5}%')
