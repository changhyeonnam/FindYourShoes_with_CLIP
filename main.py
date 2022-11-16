import numpy as np
import torch
import clip
from tqdm.notebook import tqdm
from pkg_resources import packaging
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def print_clip_info(model):
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)


if __name__=='__main__':
    print(clip.available_models())
    model, preprocess = clip.load("ViT-B/32")

    print_clip_info(model)

    dataset_classes = [
        'converse high black',
        'converse high deep bordeaux',
        'converse high midnight clover',
        'converse high parchment',
        'converse high rush blue',
        'converse high sunflower',
        'converse high white',
        'converse low black',
        'converse low deep bordeaux',
        'converse low midnight clover',
        'converse low parchment',
        'converse low rush blue',
        'converse low sunflower',
        'converse low white',
        'converse onestar black',
        'converse onestar white',
        'converse run star hike high white',
        'converse run star hike low black',
        'converse run star hike low rush blue',
        'converse run star hike low white'
    ]

    dataset_templates = [
        'a photo of {} shoes.',
        'a photo of shoes that is {}.'
        'a photo of shoes and the name is {}.'
    ]
    print(f"{len(dataset_classes)} classes, {len(dataset_templates)} templates")
    dataset_brand_classes = ['converse']

