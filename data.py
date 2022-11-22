import os
from dataclasses import dataclass, field
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
    sole:str
    name:str


class ShoesImageDataset(Dataset):
    def __init__(self,
                root:str,
                meta_info_df:pd.DataFrame,
                preprocess,
                verbose: bool=True,
                ) ->None:
        super(ShoesImageDataset).__init__()
        self._root = root
        self._preprocess = preprocess
        self.verbose = verbose
        self.name_dict, self.brand_dict, self.color_dict, self.hightop_dict, self.sole_dict, self.meta_dict = self._load_meta_info(meta_info_df)
        if self.verbose:
            print(f'\n{"*" * 10} Preprocessing about Images is Started. {"*" * 10}\n')
            print(f'# class of name: {len(self.name_dict)}\n'
                  f'# class of brand: {len(self.brand_dict)}\n'
                  f'# class of color: {len(self.color_dict)}\n'
                  f'# class of hightop: {len(self.hightop_dict)}\n'
                  f'# class of sole: {len(self.sole_dict)}\n'
                  f'# class of meta: {len(self.meta_dict)}')

        self.preproc_image_list = self._parse_image_files(root=self._root, name_dict=self.name_dict)

    def __len__(self):
        return len(self.preproc_image_list)

    def _line_mapper(self, line):

        product_id, preproc_image = line

        meta_info = self.meta_dict[product_id]
        brand,color,hightop,sole = meta_info.brand, meta_info.color, meta_info.hightop, meta_info.sole
        bid,cid,hid,sid = self.brand_dict[brand], self.color_dict[color], self.hightop_dict[hightop], self.sole_dict[sole]

        return product_id, preproc_image, bid, cid, hid,sid

    def __getitem__(self, idx):
        return self._line_mapper(self.preproc_image_list[idx])

    def _load_meta_info(self,df):
        df_dicts = df.to_dict(orient='records')
        name_dict, brand_dict, color_dict, hightop_dict, sole_dict, meta_dict={},{},{},{},{},{}
        for dt in df_dicts:
            name,brand,color,hightop,sole = dt['name'], dt['brand'], dt['color'], dt['hightop'], dt['sole']
            meta_info = ImageAnnotation(
                brand=brand,
                color=color,
                hightop=hightop,
                sole=sole,
                name=name
            )
            update_dict(dict=name_dict,key=name)
            update_dict(dict=brand_dict,key=brand)
            update_dict(dict=color_dict,key=color)
            update_dict(dict=hightop_dict,key=hightop)
            update_dict(dict=sole_dict,key=sole)
            update_dict(dict=meta_dict,key=name_dict[name],value=meta_info)
        return name_dict, brand_dict, color_dict, hightop_dict, sole_dict, meta_dict

    def get_dict(self):
        return self.name_dict, self.brand_dict, self.color_dict, self.hightop_dict, self.sole_dict,self.meta_dict

    def _parse_image_files(self,root:str, name_dict:dict):
        validate_format = ['jpg','png','jpeg']
        dir_list = os.listdir(path=root)
        preproc_image_list = []
        preproc_image_dict = {}
        for prod_dir in tqdm(dir_list):
            # get prod_name
            prod_split_underscore = prod_dir.split('_')
            if len(prod_split_underscore)<5:
                print(f"Wrong product name: {prod_dir}")
                continue
            prod_name = prod_split_underscore[4].strip()
            # check keyerror
            if prod_name not in name_dict:
                continue
            prod_id = name_dict[prod_name]
            path = os.path.join(root, prod_dir)
            file_list = os.listdir(path)
            preproc_image_dict[prod_name]=len(file_list)
            for file_name in file_list:
                file_path = os.path.join(path,file_name)
                file_path_check = file_path.split('.')
                if file_path_check[-1] not in validate_format:
                    print(file_path)
                    continue
                preproc_image = self._preprocess(Image.open(file_path))
                preproc_image_list.append([prod_id,preproc_image])
        print(f'Number of total preprocessed items: {len(preproc_image_list)}')
        if self.verbose:
            for k,v in preproc_image_dict.items():
                print(f'# {k} : {v}')

            print(f'\n{"*"*10} Preprocessing about images is Completed. {"*"*10}\n')
        return preproc_image_list

if __name__ == "__main__":
    model, preprocess = clip.load("ViT-B/32")
    ROOT_PATH = "converse dataset"
    meta_info_path = "meta_info.csv"
    dataset = ShoesImageDataset(root=ROOT_PATH,
                                preprocess=preprocess,
                                meta_info_path=meta_info_path,
                                verbose=True)
