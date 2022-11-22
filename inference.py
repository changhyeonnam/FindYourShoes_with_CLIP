import torch
import pandas as pd
from PIL import Image
import clip
from tqdm import tqdm
import os
from prompt_compute import TextPreCompute
from utils import invert_dict,load_meta_info,build_feat_dict

class Recommend:
    def __init__(self,
                 image_path,
                 model,
                 preprocess,
                 text_precompute:TextPreCompute,
                 text_inv_dicts,
                 meta_df
                 ):
        self.image = self.parse_image(image_path)
        self.classified_image = self.classify(self.image, model,preprocess, text_precompute, text_inv_dicts,meta_df)

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
                zeroshot_logit = logits[top1k_zeroshot_idx]
                zeroshot_product = product_list[top1k_zeroshot_idx]
                print(f'zeroshot logit: {zeroshot_logit}, name logit : {name_logit}')
                print(f'zeroshot name: {zeroshot_product}, name top1k : {top1k_name} ')
                if zeroshot_logit>name_logit:
                    classified_product = zeroshot_product
                else:
                    classified_product = top1k_name
            else:
                print(f'name logit:{name_logit}')
                print(f'name top1k:{top1k_name}')
                classified_product = top1k_name

            return classified_product

    def recommend_with_text(text, preprocessed_images):
        pass

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print available models
    print('This is available models: ', clip.available_models())
    model, preprocess = clip.load('ViT-B/32')

    meta_info_path = "meta_info.csv"
    prompt_path = "config/prompt_template.yaml"

    meta_df = pd.read_csv(meta_info_path)
    name_dict, brand_dict, color_dict, hightop_dict, sole_dict = load_meta_info(meta_df)
    name_inv_dict, brand_inv_dict, color_inv_dict, hightop_inv_dict, sole_inv_dict = invert_dict(name_dict), \
                                                                                     invert_dict(brand_dict), \
                                                                                     invert_dict(color_dict), \
                                                                                     invert_dict(hightop_dict), \
                                                                                     invert_dict(sole_dict)

    text_inv_dicts = build_feat_dict(name_inv_dict, brand_inv_dict,color_inv_dict, hightop_inv_dict, sole_inv_dict)

    text_precompute = TextPreCompute(model,
                                     device,
                                     prompt_path,
                                     name_dict,
                                     brand_dict,
                                     color_dict,
                                     hightop_dict,
                                     sole_dict)

    # image_path
    PATH_IMG = 'recommend_image/img.png'
    inference = Recommend(image_path=PATH_IMG,model=model,preprocess=preprocess,
                          text_precompute=text_precompute,text_inv_dicts=text_inv_dicts,meta_df=meta_df)

if __name__ == '__main__':
    main()


