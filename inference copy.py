import torch
import pandas as pd
from PIL import Image
import clip
from tqdm import tqdm
import os
from prompt_compute import TextPreCompute
from utils import invert_dict,load_meta_info,build_feat_inv_dict
import numpy as np
from collections import Counter, defaultdict

class ImageCandidate:
    def __init__(self, infer_path, preprocess):
        self.infer_path = infer_path
        self.preproc_image_dict = self._parse_image_files(infer_path,preprocess)

    def _parse_image_files(self, infer_path: str, preprocess):
        validate_format = ['jpg', 'png', 'jpeg']
        dir_list = os.listdir(path=infer_path)
        # preproc_image_dict = {}
        preproc_image_dict = defaultdict(list)
        for prod_dir in tqdm(dir_list):
            # get prod_name
            prod_split_underscore = prod_dir.split('_')
            if len(prod_split_underscore) < 5:
                print(f"Wrong product name: {prod_dir}")
                continue
            product_name = prod_split_underscore[4].strip() # extract name of shoes
            # if product_name not in preproc_image_dict:
            preproc_image_dict[product_name]=[]
            # else:
            path = os.path.join(infer_path, prod_dir)
            file_list = os.listdir(path)
            for file_name in file_list:
                file_path = os.path.join(path,file_name)
                file_path_check = file_path.split('.')
                if file_path_check[-1] not in validate_format:
                    print(file_path)
                    continue
                preproc_image = preprocess(Image.open(file_path))
                """
                preproc_list = [(product_name, preproc_image)]
                for k, v in preproc_list:
                    preproc_image_dict[k].append(v)
                """
                preproc_image_dict[product_name].append(preproc_image)

        return preproc_image_dict

    def get_preproc_image_dict(self):
        return self.preproc_image_dict


class Recommend:
    def __init__(self,
                 image_path,
                 model,
                 preprocess,
                 text_precompute:TextPreCompute,
                 preproc_image_dict,
                 text_inv_dicts,
                 meta_dict,
                 meta_df,
                 verbose = True
                 ):
        self.model = model
        self.image = self.parse_image(image_path)
        self.preproc_image_dict = preproc_image_dict
        self.verbose =verbose
        self.classified_product = self.classify(self.image, model,preprocess, text_precompute, text_inv_dicts,meta_df)
        self.product_meta = meta_dict[self.classified_product]
        print('Given Shoes class is ', self.classified_product)
        print(f'This is information about given Shoes:\nbrand: {self.product_meta.brand}\n'
              f'color: {self.product_meta.color}\nhightop: {self.product_meta.hightop}\n'
              f'sole: {self.product_meta.sole}')
        print('You can use these info for generating prompt')
        print(f'{"-"*50}')

    def get_text(self):
        text = input('What Feature do you want change?(ex, I want same brand but color is red/gray/bown)\n : ')
        return text

    def parse_image(self, image_path):
        validate_format = ['jpg', 'png', 'jpeg']
        image_path_check = image_path.split('.')
        if image_path_check[-1] not in validate_format:
            print('Image format is not valid!.')
            exit(1)
        image = Image.open(image_path)
        return image

    def find_filtered_prod(self,df, brand, color, hightop, sole):
        prod_list = df.loc[((df['brand']==brand) & (df['color'] == color) & (df['hightop'] == hightop)), 'name'].values.tolist()
        return prod_list


    def classify(self, image, model, preprocess, text_precompute, text_inv_dicts,meta_df):
        # precomputed text features with prompt.
        name_weights, brand_weights, color_weights, hightop_weights, sole_weights=text_precompute.get_precomputed_text()

        # id to name dictionary
        name_inv_dict, brand_inv_dict, color_inv_dict, hightop_inv_dict, sole_inv_dict = text_inv_dicts['name'],\
                                                                                         text_inv_dicts['brand'],\
                                                                            text_inv_dicts['color'],\
                                                                                         text_inv_dicts['hightop'],\
                                                                                         text_inv_dicts['sole']
        # preprocessed image
        preproc_image = preprocess(image)
        preproc_image = torch.unsqueeze(preproc_image,dim=0)
        with torch.no_grad():
            image_feature = model.encode_image(preproc_image)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)

            # First zeroshot with brand
            logits = (100.0 * image_feature @ brand_weights)
            logits = torch.squeeze(logits)
            top1k_brand_idx = logits.topk(1, 0, True, True)[1].t().flatten().item()
            top1k_brand = brand_inv_dict[top1k_brand_idx]

            # Second zeroshot with color
            logits = (100.0 * image_feature @ color_weights)
            logits = torch.squeeze(logits)
            top1k_color_idx = logits.topk(1, 0, True, True)[1].t().flatten().item()
            top1k_color = color_inv_dict[top1k_color_idx]

            # Third zeroshot with hightop
            logits = (100.0 * image_feature @ hightop_weights)
            logits = torch.squeeze(logits)
            top1k_hightop_idx = logits.topk(1, 0, True, True)[1].t().flatten().item()
            top1k_hightop = hightop_inv_dict[top1k_hightop_idx]

            #Forth zeroshot with sole
            logits = (100.0 * image_feature @ sole_weights)
            logits = torch.squeeze(logits)
            top1k_sole_idx = logits.topk(1, 0, True, True)[1].t().flatten().item()
            top1k_sole = sole_inv_dict[top1k_sole_idx]

            # zeroshot with name
            logits = (100.0 * image_feature @ name_weights)
            logits = torch.squeeze(logits)
            top1k_name_idx = logits.topk(1, 0, True, True)[1].t().flatten().item()
            top1k_name = name_inv_dict[top1k_name_idx]
            name_logit = logits[top1k_name_idx]

            product_list = self.find_filtered_prod(meta_df, top1k_brand,
                                               top1k_color, top1k_hightop, top1k_sole)
            classified_product = None

            # if product_list is not empty, do zeroshot with filtered images.
            # else, just zeroshot with name
            if len(product_list) != 0:
                zeroshot_weight = text_precompute.compute_prompt_name(product_list, top1k_brand, top1k_color, top1k_hightop, top1k_sole, False)
                logits = (100.0 * image_feature @ zeroshot_weight)
                logits = torch.squeeze(logits)
                top1k_zeroshot_idx = logits.topk(1, 0, True, True)[1].t().flatten().item()

                # if logit is one element, then it can't be accessed by index after squeezing.
                if len(product_list)==1:
                    zeroshot_logit = logits
                else:
                    zeroshot_logit = logits[top1k_zeroshot_idx]

                zeroshot_product = product_list[top1k_zeroshot_idx]
                if self.verbose:
                    print(f'zeroshot logit: {zeroshot_logit}, name logit : {name_logit}')
                    print(f'zeroshot name: {zeroshot_product}, name top1k : {top1k_name} ')
                    print(f'{"-" * 50}')
                if zeroshot_logit>name_logit:
                    classified_product = zeroshot_product
                else:
                    classified_product = top1k_name
            else:
                if self.verbose:
                    print(f'name logit:{name_logit}')
                    print(f'name top1k:{top1k_name}')
                    print(f'{"-" * 50}')
                classified_product = top1k_name

            return classified_product

    def recommend(self):
        model = self.model
        input_text = self.get_text()
        preproc_image_dict = self.preproc_image_dict
        classified_product = self.classified_product
        Additional_text = f"Similar to {classified_product}, I want photo of "
        input_text = Additional_text+input_text

        text = clip.tokenize(input_text)
        text_feature = model.encode_text(text)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)

        products = []
        preproc_images = []
        rec_list = []
        for product, preproc_image_list in preproc_image_dict.items():
            if product == classified_product:
                continue
            products.append(product)
            # 1-shot
            preproc_images.append(preproc_image_list[0])
        preproc_images = torch.stack(preproc_images, dim=0)
        image_features = model.encode_image(preproc_images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = (100.0 * text_feature @ image_features.T)
        logits = torch.squeeze(logits)
        top1k_idx = logits.topk(1, 0, True, True)[1].t().flatten().item()
        rec_item = products[top1k_idx]
        print('rec_items: ', rec_item)
        rec_list.append(rec_item)
        print('rec_list: ', rec_list)
        count = Counter(rec_list)
        print('count: ', count)
        max_count = (count.most_common(1))[0][0]
        # top3k_idx = logits.topk(3, 0, True, True)[1].t().flat ten().tolist()
        print(f"Recommended item is {max_count}")

        """
        if self.verbose:
            for rank, idx in enumerate((top3k_idx)):
                print(f"rank#{rank+1} : {products[idx]}")
        print(f'{"-" * 50}')
        """


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print available models
    print('This is available models: ', clip.available_models())
    model, preprocess = clip.load('ViT-B/32')

    meta_info_path = "meta_info.csv"
    prompt_path = "config/prompt_template.yaml"
    infer_path = "dataset_infer"

    # preprocessing candidate images for inference
    image_candidate = ImageCandidate(infer_path, preprocess)
    preproc_image_dict = image_candidate.get_preproc_image_dict()

    # loading meta dataframe and preprocessing.
    meta_df = pd.read_csv(meta_info_path)
    name_dict, brand_dict, color_dict, hightop_dict, sole_dict, meta_dict = load_meta_info(meta_df)
    text_inv_dicts = build_feat_inv_dict(name_dict, brand_dict,color_dict, hightop_dict, sole_dict)

    text_precompute = TextPreCompute(model,
                                     device,
                                     prompt_path,
                                     name_dict,
                                     brand_dict,
                                     color_dict,
                                     hightop_dict,
                                     sole_dict,
                                     verbose=False)



    # image_path
    PATH_IMG = 'recommend_image/img.png'
    inference = Recommend(image_path=PATH_IMG,model=model,preprocess=preprocess,
                          text_precompute=text_precompute,preproc_image_dict = preproc_image_dict,
                          text_inv_dicts=text_inv_dicts,meta_dict=meta_dict,meta_df=meta_df)
    for i in range(10):
        inference.recommend()
if __name__ == '__main__':
    main()


