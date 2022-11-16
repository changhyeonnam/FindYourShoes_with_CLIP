import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict,List,Optional,Tuple
from torch.utils.data import Dataset
from PIL import Image
import clip
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils import update_dict

@dataclass
class ImageAnnotation:
    brand:str
    color:str
    hightop:str
    name:str


class ShoesImageDataset(Dataset):
    def __init__(self,
                root:str,
                meta_info_path:str,
                model,
                preprocess,
                device,
                verbose: bool=True
                ) ->None:
        super(ShoesImageDataset).__init__()
        self._root = root
        self._preprocess = preprocess
        self.verbose = verbose
        self.name_dict, self.brand_dict, self.color_dict, self.hightop_dict, self.meta_dict = self._load_meta_info(meta_info_path)
        if self.verbose:
            print(f'Length of name_dict: {len(self.name_dict)}\n'
                  f'Length of brand_dict: {len(self.brand_dict)}\n'
                  f'Length of color_dict: {len(self.color_dict)}\n'
                  f'Length of hightop_dict: {len(self.hightop_dict)}\n'
                  f'Length of meta_dict: {len(self.meta_dict)}')

        self.precomp_image_list = self._parse_image_files(root=self._root, name_dict=self.name_dict, model=model,device=device)

    def __len__(self):
        return len(precomp_image_list)

    def _line_mapper(self, line):
        prod_id, precomp_image = line
        meta_info = self.meta_dict[prod_id]
        brand,color,hightop,name = meta_info.brand, meta_info.color, meta_info.hightop, meta_info.name
        bid,cid,hid,nid = self.brand_dict[brand], self.color_dict[color], self.hightop_dict[hightop], self.name_dict[name]
        return prod_id, precomp_image, bid, cid, hid, nid

    def __getitem__(self, idx):
        return self._line_mapper(self.precomp_image_list[idx])

    def _load_meta_info(self,path):
        df = pd.read_csv(path)
        df_dicts = df.to_dict(orient='records')
        name_dict, brand_dict, color_dict, hightop_dict, meta_dict={},{},{},{},{}
        for dt in tqdm(df_dicts):
            name,brand,color,hightop = dt['name'], dt['brand'], dt['color'], dt['hightop']
            meta_info = ImageAnnotation(
                brand=brand,
                color=color,
                hightop=hightop,
                name=name
            )
            update_dict(dict=name_dict,key=name)
            update_dict(dict=brand_dict,key=brand)
            update_dict(dict=color_dict,key=color)
            update_dict(dict=hightop_dict,key=hightop)
            update_dict(dict=meta_dict,key=name_dict[name],value=meta_info)
        return name_dict, brand_dict, color_dict, hightop_dict, meta_dict

    def get_dict(self):
        return self.name_dict, self.brand_dict, self.color_dict, self.hightop_dict, self.meta_dict

    def _parse_image_files(self,root:str, name_dict:dict, model,device):
        dir_list = os.listdir(path=root)
        precomp_image_list = []
        for prod_name in tqdm(dir_list):
            if prod_name not in name_dict:
                continue
            prod_id = name_dict[prod_name]
            path = os.path.join(root, prod_name)
            file_list = os.listdir(path)
            for file_name in file_list:
                file_path = os.path.join(path,file_name)
                preproc_image = self._preprocess(Image.open(file_path)).unsqueeze(0).to(device)
                precomp_image = model.encode_image(preproc_image)
                precomp_image /= precomp_image.norm(dim=-1, keepdim=True)
                precomp_image_list.append([prod_id,precomp_image])
            print(f' Preprocessing images of {prod_name} is Completed, #{prod_name} = {len(file_list)} ')
        return precomp_image_list

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocess = clip.load("ViT-B/32")
    ROOT_PATH = "converse dataset"
    meta_info_path = "meta_info.csv"
    dataset = ShoesImageDataset(root=ROOT_PATH,
                                model=model,
                                preprocess=preprocess,
                                meta_info_path=meta_info_path,
                                device=device,
                                verbose=True)
