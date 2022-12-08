import sys
sys.path.append('..')

import dataclasses

import pandas as pd
import torch
import clip
from tqdm import tqdm
from pkg_resources import packaging
from utils import print_clip_info, find_filtered_prod, invert_dict
from CoOp_dataset import ShoesImageDataset
from CoOp import CoOp
from metric import accuracy, print_acc
from torch.utils.data import DataLoader
import ssl


def main():
    root_path = "../dataset"
    meta_info_path = "../meta_info.csv"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setting optimizer and model
    @dataclasses.dataclass
    class Config:
        device: str

    cfg = Config
    cfg.device = device
    cfg.batch_size = 128
    cfg.epochs = 200
    cfg.learning_rate = 2e-3
    # print available models
    print('This is available models: ', clip.available_models())

    # load model
    model, preprocess = clip.load('ViT-B/32')

    # load meta info dataframe
    meta_info_df = pd.read_csv(meta_info_path)
    dataset = ShoesImageDataset(root=root_path,
                                preprocess=preprocess,
                                meta_info_df=meta_info_df,
                                verbose=True)
    name_dict = dataset.name_dict
    dataloader = DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=1)

    # brand
    classnames = list(name_dict.keys())

    # setting hyper parameter
    learning_rate = cfg.learning_rate
    epochs = cfg.epochs



    model = CoOp(cfg,classnames)
    optimizer = torch.optim.SGD(params=model.parameters(),lr=learning_rate)
    Loss = torch.nn.CrossEntropyLoss()
    for epoch in range(0, epochs):
        for i, (preproc_image, name, name_label) in enumerate(tqdm(dataloader, total=len(dataloader))):
            preproc_image,name_label  = preproc_image.to(device),name_label.to(device)
            pred = model(preproc_image)
            loss = Loss(pred,name_label)
            print(f'{i+1} epochs : loss value: {loss}')

            loss.backward()
            optimizer.zero_grad()
            optimizer.step()


if __name__ == '__main__':
    main()