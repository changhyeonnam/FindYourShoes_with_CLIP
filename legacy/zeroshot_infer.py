import numpy as np
import torch
import clip
from tqdm import tqdm
from pkg_resources import packaging
from PIL import Image
import matplotlib.pyplot as plt
import skimage
import os
from torchvision.datasets import ImageFolder
import torchvision
import shutil
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


def zeroshot_classifier(classnames, templates,deivce):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, total=len(classnames)):
            texts = [template.format(classname) for template in templates] #format with class

            texts = clip.tokenize(texts).to(device) #tokenize
            # print(texts.shape)

            class_embeddings = model.encode_text(texts) #embed with text encoder
            # print(class_embeddings.shape)

            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # print(class_embeddings.shape)

            class_embedding = class_embeddings.mean(dim=0)
            # print(class_embedding.shape)
            #
            class_embedding /= class_embedding.norm()
            # print(class_embedding.shape)
            # print(class_embedding.shape)
            # exit(1)
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


if __name__ =="__main__":
    print("Available Models: ", clip.available_models())
    model, preprocess = clip.load("ViT-B/32")
    print_clip_info(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    brand_classes = ["converse", "nike", "adidas", "vans", "fila", "newbalance", "reebok"]

    brand_templates = [
        # 'shoes\'s brand is {}.',
        'shoes made by the {}.',
        '{} is brand of this shoes.',
        'a photo of the {} shoes.',
        # 'a photo of a {} sneakers.',
    ]

    color_classes = ["black", "blue", "deep bordeaux", "green","white","yellow"]

    color_templates = [
        'a photo of the {} color shoes.',
        'a photo of a {} shoes.',
        'Color of shoes is {}'
    ]

    height_classes = ["high", "low"]

    height_templates = [
        'a photo of the {} shoes.',
        'a photo of a {} sneakers.',
        'a bright photo of the {} sneakers.'
    ]

    ROOT_PATH = "../dataset"
    DIR_CONVERSE = os.path.join(ROOT_PATH,'converse')
    FILE_LIST = os.listdir(DIR_CONVERSE)
    print('list of files: ', FILE_LIST)

    # for idx,file_name in enumerate(os.listdir(DIR_CONVERSE),1):
    #     source = os.path.join(DIR_CONVERSE, file_name)
    #     destination = os.path.join(DIR_CONVERSE, str(idx)+".jpg")
    #     print(source, destination)
    #     os.rename(source, destination)

    # exit(1)
    NUM_CLASS = len(FILE_LIST)
    print('NUM_CLASS: ', NUM_CLASS)
    images = torchvision.datasets.ImageFolder(root=DIR_CONVERSE, transform=preprocess)
    loader = torch.utils.data.DataLoader(dataset=images, batch_size=64, num_workers=1)

    zeroshot_weights = zeroshot_classifier(classnames=color_classes, templates=color_templates,deivce=device)
    # color_weights = zeroshot_classifier(classnames=color_classes, templates=color_templates,deivce=device)
    # height_weights = zeroshot_classifier(classnames=height_classes, templates=height_templates,deivce=device)

    # print('brand weights is:', zeroshot_weights)


    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(tqdm(loader, total=len(loader))):
            images = images.to(device)
            target = target.to(device)
            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ zeroshot_weights)
            # print(logits.shape)
            # print(image_features.shape)
            # print(zeroshot_weights.shape)
            # exit(1)

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")
