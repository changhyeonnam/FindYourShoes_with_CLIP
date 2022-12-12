import os.path
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
from CoOp import PromptLearner,CustomCLIP,load_clip_to_cpu

def main():
    train_path = "../final_dataset"
    val_path = "../final_dataset"
    meta_info_path = "../meta_info_final.csv"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setting optimizer and model
    @dataclasses.dataclass
    class Config:
        device: str

    cfg = Config
    cfg.device = device
    cfg.batch_size = 128
    cfg.epochs = 200
    cfg.learning_rate = 1e-4
    cfg.choose_feature = 'brand'
    cfg.backbone = 'ViT-B/32'
    # print available models
    print('This is available models: ', clip.available_models())

    # load model
    clip_model, preprocess = clip.load(cfg.backbone)
    clip_model = load_clip_to_cpu(cfg.backbone)

    # load meta info dataframe
    meta_info_df = pd.read_csv(meta_info_path)
    train_dataset = ShoesImageDataset(root=train_path,
                                preprocess=preprocess,
                                meta_info_df=meta_info_df,
                                verbose=True,
                                feautre = cfg.choose_feature)
    train_set = train_dataset.file_set

    val_dataset = ShoesImageDataset(root=val_path,
                                preprocess=preprocess,
                                meta_info_df=meta_info_df,
                                verbose=False,
                                filter_file_set=train_set,
                                mode='validation',
                                feautre = cfg.choose_feature)
    val_set = val_dataset.file_set
    print(len(train_set&val_set))


    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=1)

    feature_dict = f'train_dataset.{cfg.choose_feature}_dict'
    feature_dict = eval(feature_dict)

    # classnames for prompt
    classnames = list(feature_dict.keys())

    # setting hyper parameter
    learning_rate = cfg.learning_rate
    epochs = cfg.epochs

    # CLIP's default precision is fp16
    clip_model.float()
    prompt_learner = PromptLearner(cfg, classnames, clip_model)
    # model = CoOp(cfg,classnames)
    optimizer = torch.optim.Adam(params=prompt_learner.parameters(),lr=learning_rate)
    model = CustomCLIP(cfg, clip_model, prompt_learner).to(device)
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    Loss = torch.nn.CrossEntropyLoss()

    # stop training when fail n times.
    cfg.fail_new_record = 5
    cfg.save_dir = 'saved'
    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)
    new_record = None

    for epoch in range(0, epochs):
        avg_loss = 0
        total_len = len(train_dataloader)*cfg.batch_size
        prompt_learner.train()
        for i, (preproc_image, name, target) in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
            preproc_image,target  = preproc_image.to(device),target.to(device)
            pred = model(preproc_image)
            loss = Loss(pred,target)
            optimizer.zero_grad()
            avg_loss +=loss
            loss.backward()
            optimizer.step()
        print(f'{epoch + 1} epochs : loss value: {avg_loss/total_len}')

        # run validation
        model.eval()
        total_val_len = len(val_dataloader) * cfg.batch_size
        avg_acc = 0
        with torch.no_grad():
            for i, (preproc_image, name, target) in enumerate(tqdm(val_dataloader, total=len(val_dataloader))):
                preproc_image, target = preproc_image.to(device), target.to(device)
                pred = model(preproc_image)
                acc = accuracy(pred, target, topk=(1,))
                avg_acc += acc[0]
        avg_acc = avg_acc/total_val_len
        if new_record is None or avg_acc > new_record:
            new_record = avg_acc
            torch.save(prompt_learner.state_dict(), os.path.join(cfg.save_dir, f'{cfg.choose_feature}_prompt_learner({epoch}).pth'))
        print(f'{epoch + 1} epochs : accuracy : {avg_acc/total_val_len}')


if __name__ == '__main__':
    main()