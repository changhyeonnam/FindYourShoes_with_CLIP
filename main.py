import numpy as np
import torch
import clip
from tqdm import tqdm
from pkg_resources import packaging
from utils import print_clip_info
from data import ShoesImageDataset
from prompt_compute import TextPreCompute
from metric import accuracy,print_acc
from torch.utils.data import DataLoader

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print available models
    print('This is available models: ', clip.available_models())

    # load model
    model, preprocess = clip.load("ViT-B/32")

    # print model information
    print_clip_info(model)

    ROOT_PATH = "converse dataset"
    meta_info_path = "meta_info.csv"
    prompt_path = "config/prompt_template.yaml"

    dataset = ShoesImageDataset(root=ROOT_PATH,
                                preprocess=preprocess,
                                meta_info_path=meta_info_path,
                                verbose=True)

    dataloader = DataLoader(dataset=dataset,  batch_size=16, num_workers=1)

    # get dictionary about shoes.
    name_dict, brand_dict, color_dict, hightop_dict, meta_dict = dataset.get_dict()

    # precompute text and prompt template with clip moodel.
    encoded_text = TextPreCompute(model,
                                  device,
                                  prompt_path,
                                  name_dict,
                                  brand_dict,
                                  color_dict,
                                  hightop_dict)

    name_weights, brand_weights, color_weights, hightop_weights = encoded_text.get_precomputed_text()

    with torch.no_grad():
        brand_top1, brand_top5, name_top1, name_top5, color_top1, color_top5, hightop_top1, hightop_top5, n = \
            0., 0., 0.,0., 0., 0.,0., 0., 0.

        for i, (prod_id, preproc_image, bid, cid, hid, nid) in enumerate(tqdm(dataloader, total=len(dataloader))):
            preproc_image = preproc_image.to(device)
            target_brand = bid.to(device)
            target_color = cid.to(device)
            target_hightop = hid.to(device)
            target_name = nid.to(device)

            image_features = model.encode_image(preproc_image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # First zeroshot with brand
            logits = (100.0 * image_features @ brand_weights)
            acc1, acc5 = accuracy(logits, target_brand, topk=(1, 1))
            brand_top1 += acc1
            brand_top5 += acc5

            # Secondly zeroshot with color
            logits = (100.0 * image_features @ color_weights)
            exit(1)
            acc1, acc5 = accuracy(logits, target_color, topk=(1, 5))
            color_top1 += acc1
            color_top5 += acc5

            # Thirdly zeroshot with hightop
            logits = (100.0 * image_features @ hightop_weights)
            acc1, acc5 = accuracy(logits, target_hightop, topk=(1, 2))
            hightop_top1 += acc1
            hightop_top5 += acc5

            # Lastly zeroshot with name --> This is temporary code. We will fix with using filtering table.
            logits = (100.0 * image_features @ name_weights)
            acc1, acc5 = accuracy(logits, target_name, topk=(1, 5))
            name_top1 += acc1
            name_top5 += acc5

            n += preproc_image.size(0)

        print_acc('brand', brand_top1, brand_top5,n)
        print_acc('color', color_top1, color_top5,n)
        print_acc('hightop', hightop_top1, hightop_top5,n)
        print_acc('name', name_top1, name_top5,n)



