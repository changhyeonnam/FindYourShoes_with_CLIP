import numpy as np
import torch
import clip


def print_clip_info(model):
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

def update_dict_string(dt, key, value):
    if key in dt:
        dt[key] += ','
        dt[key] += value
    else:
        dt[key] = ''
        dt[key] += value

def update_dict(dict: dict, key, value=None):
    if key not in dict:
        if value is None:
            dict[key] = len(dict) + 1
        else:
            dict[key] = value


def update_dict_list(dt, key, value):
    if key in dt:
        dt[key].append(value)
    else:
        dt[key] = []
        dt[key].append(value)


def update_dict_num(dt, key):
    if key in dt:
        dt[key] += 1
    else:
        dt[key] = 1
