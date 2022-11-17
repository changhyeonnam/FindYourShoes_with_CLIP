import numpy as np
import pandas as pd
import torch
import clip
from tqdm import tqdm
from pkg_resources import packaging
from utils import print_clip_info, find_filtered_prod, invert_dict
from data import ShoesImageDataset
from prompt_compute import TextPreCompute
from metric import accuracy, print_acc
from torch.utils.data import DataLoader


def main(root_path, meta_info_path, prompt_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print available models
    print('This is available models: ', clip.available_models())

    # load model
    model, preprocess = clip.load("ViT-B/32")

    # print model information
    print_clip_info(model)

    # load meta info dataframe
    meta_info_df = pd.read_csv(meta_info_path)

    dataset = ShoesImageDataset(root=root_path,
                                preprocess=preprocess,
                                meta_info_df=meta_info_df,
                                verbose=True)

    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=1)

    # get dictionary about shoes.
    name_dict, brand_dict, color_dict, hightop_dict, meta_dict = dataset.get_dict()
    name_inv_dict, brand_inv_dict, color_inv_dict, hightop_inv_dict = invert_dict(name_dict), invert_dict(brand_dict), \
                                                                      invert_dict(color_dict), invert_dict(hightop_dict)
    # precompute text and prompt template with clip moodel.
    text_precompute = TextPreCompute(model,
                                     device,
                                     prompt_path,
                                     name_dict,
                                     brand_dict,
                                     color_dict,
                                     hightop_dict)

    # get precomputed embeddings from TextPreCompute
    name_weights, brand_weights, color_weights, hightop_weights = text_precompute.get_precomputed_text()

    with torch.no_grad():
        brand_top1, brand_top5, name_top1, name_top5, color_top1, color_top5, hightop_top1, hightop_top5, zeroshot_correct_count, total_num = \
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.

        for i, (prod_id, preproc_image, bid, cid, hid) in enumerate(tqdm(dataloader, total=len(dataloader))):
            preproc_image = preproc_image.to(device)
            target_brand = bid.to(device)
            target_color = cid.to(device)
            target_hightop = hid.to(device)
            target_name = prod_id.to(device).tolist()

            image_features = model.encode_image(preproc_image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # First zeroshot with brand
            logits = (100.0 * image_features @ brand_weights)
            top1k_brand_idx = logits.topk(1, 1, True, True)[1].t().flatten().tolist()
            top1k_brand_name = [brand_inv_dict[idx] for idx in top1k_brand_idx]
            acc1, acc5 = accuracy(logits, target_brand, topk=(1, 1))
            brand_top1 += acc1
            brand_top5 += acc5

            # Second zeroshot with color
            logits = (100.0 * image_features @ color_weights)
            top1k_color_idx = logits.topk(1, 1, True, True)[1].t().flatten().tolist()
            top1k_color_name = [color_inv_dict[idx] for idx in top1k_color_idx]
            acc1, acc5 = accuracy(logits, target_color, topk=(1, 5))
            color_top1 += acc1
            color_top5 += acc5

            # Third zeroshot with hightop
            logits = (100.0 * image_features @ hightop_weights)
            top1k_hightop_idx = logits.topk(1, 1, True, True)[1].t().flatten().tolist()
            top1k_hightop_name = [hightop_inv_dict[idx] for idx in top1k_hightop_idx]
            acc1, acc5 = accuracy(logits, target_hightop, topk=(1, 2))
            hightop_top1 += acc1
            hightop_top5 += acc5

            # We want to find name using brand, color, hightop info which is obtained from above.
            # We can list of name using multiple filter in pandas.
            # product_lists shape : batch_size x N_i (N_i is number of filtered products for each sample)
            product_lists = find_filtered_prod(meta_info_df, top1k_brand_name,
                                               top1k_color_name, top1k_hightop_name)

            # Last zeroshot with filtered name
            correct_count = 0
            for img_idx, prod_list, target in zip(range(image_features.size(0)),product_lists, target_name):
                if name_inv_dict[target] not in prod_list:
                    continue
                else:
                    tar_name = name_inv_dict[target]
                target_idx = torch.LongTensor([prod_list.index(tar_name)]).to(device)
                zeroshot_weight = text_precompute.compute_prompt_name(prod_list)
                image_feature = image_features[img_idx]
                logits = (100.0 * image_feature @ zeroshot_weight)
                pred = logits.topk(1, 0, True, True)[1].t().flatten().item()
                if pred == target_idx:
                    correct_count += 1
            acc1 = correct_count/preproc_image.size(0)
            print(f'{i+1}th batch Zeroshot performance: {acc1*100:.2f}')
            zeroshot_correct_count+=correct_count

            total_num += preproc_image.size(0)

        print_acc('brand', total_num, brand_top1, brand_top5)
        print_acc('color', total_num, color_top1, color_top5)
        print_acc('hightop', total_num, hightop_top1, hightop_top5)
        print_acc('zeroshot', total_num, zeroshot_correct_count)


if __name__ == '__main__':
    root_path = "converse dataset"
    meta_info_path = "meta_info.csv"
    prompt_path = "config/prompt_template.yaml"

    main(root_path, meta_info_path, prompt_path)
