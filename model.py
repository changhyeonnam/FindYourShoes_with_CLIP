import os.path as osp

import torch
import torch.nn as nn
import pandas as pd
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from pkg_resources import packaging
from utils import print_clip_info, find_filtered_prod, invert_dict
from data import ShoesImageDataset
from prompt_compute import TextPreCompute
from metric import accuracy, print_acc
from torch.utils.data import DataLoader
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import dassl

_tokenizer = _Tokenizer()

class TextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.transformer = model.transformer
        self.positional_embedding = model.positional_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.dtype = model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        ctx_init = None # ???
        dtype = model.dtype
        ctx_dim = model.ln_final.weight.shape[0]
        clip_imsize = model.visual.input_resolution
        cfg_imsize = None # shape of input image ~size[0]

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, classnames, model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = model.visual
        self.text_encoder = TextEncoder(model)
        self.logit_scale = model.logit_scale
        self.dtype = model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


def main(root_path, meta_info_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocess = clip.load('ViT-B/32')
    print()
    print("Here")

    meta_info_df = pd.read_csv(meta_info_path)

    dataset = ShoesImageDataset(root=root_path,
                                preprocess=preprocess,
                                meta_info_df=meta_info_df,
                                verbose=True)

    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=1)

    name_dict, brand_dict, color_dict, hightop_dict, sole_dict, meta_dict = dataset.get_dict()
    name_inv_dict, brand_inv_dict, color_inv_dict, hightop_inv_dict, sole_inv_dict = invert_dict(name_dict), invert_dict(brand_dict), \
                                                                      invert_dict(color_dict), invert_dict(hightop_dict), invert_dict(sole_dict)


    print()
    print("****")

    prompt = PromptLearner(name_dict, model)
    print(prompt)
    print()
    print("!!!!!!!!!!!!!!!")

    # text_precompute = TextPreCompute(model,
    #                                  device,
    #                                  prompt,
    #                                  name_dict,
    #                                  brand_dict,
    #                                  color_dict,
    #                                  hightop_dict,
    #                                  sole_dict)
    # name_weights, brand_weights, color_weights, hightop_weights, sole_weights = text_precompute.get_precomputed_text()
    # with torch.no_grad():
    #     brand_top1, brand_top5, name_top1, name_top5, color_top1, color_top5, \
    #     hightop_top1, hightop_top5, sole_top1, sole_top5, zeroshot_correct_count, total_num = \
    #         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    #
    #     for i, (prod_id, preproc_image, bid, cid, hid, sid) in enumerate(tqdm(dataloader, total=len(dataloader))):
    #         preproc_image = preproc_image.to(device)
    #         target_brand = bid.to(device)
    #         target_color = cid.to(device)
    #         target_hightop = hid.to(device)
    #         target_sole = sid.to(device)
    #         target_name = prod_id.to(device)
    #
    #         image_features = model.encode_image(preproc_image)
    #         image_features /= image_features.norm(dim=-1, keepdim=True)
    #
    #         # First zeroshot with brand
    #         logits = (100.0 * image_features @ brand_weights)
    #         top1k_brand_idx = logits.topk(1, 1, True, True)[1].t().flatten().tolist()
    #         top1k_brand_name = [brand_inv_dict[idx] for idx in top1k_brand_idx]
    #         acc1, acc5 = accuracy(logits, target_brand, topk=(1, 1))
    #         brand_top1 += acc1
    #         brand_top5 += acc5
    #
    #         # Second zeroshot with color
    #         logits = (100.0 * image_features @ color_weights)
    #         top1k_color_idx = logits.topk(1, 1, True, True)[1].t().flatten().tolist()
    #         top1k_color_name = [color_inv_dict[idx] for idx in top1k_color_idx]
    #         acc1, acc5 = accuracy(logits, target_color, topk=(1, 5))
    #         color_top1 += acc1
    #         color_top5 += acc5
    #
    #         # Third zeroshot with hightop
    #         logits = (100.0 * image_features @ hightop_weights)
    #         top1k_hightop_idx = logits.topk(1, 1, True, True)[1].t().flatten().tolist()
    #         top1k_hightop_name = [hightop_inv_dict[idx] for idx in top1k_hightop_idx]
    #         acc1, acc5 = accuracy(logits, target_hightop, topk=(1, 2))
    #         hightop_top1 += acc1
    #         hightop_top5 += acc5
    #
    #         # Forth zeroshot with sole
    #         logits = (100.0 * image_features @ sole_weights)
    #         top1k_sole_idx = logits.topk(1, 1, True, True)[1].t().flatten().tolist()
    #         top1k_sole_name = [sole_inv_dict[idx] for idx in top1k_sole_idx]
    #         acc1 = accuracy(logits, target_sole, topk=(1,))
    #         # sole_top1 += acc1
    #         # sole_top5 += acc5
    #
    #         # Lastly zeroshot with name --> This is temporary code. We will fix with using filtering table.
    #         logits = (100.0 * image_features @ name_weights)
    #         acc1, acc5 = accuracy(logits, target_name, topk=(1, 5))
    #         name_top1 += acc1
    #         name_top5 += acc5
    #
    #         target_name = target_name.tolist()
    #         # We want to find name using brand, color, hightop info which is obtained from above.
    #         # We can list of name using multiple filter in pandas.
    #         # product_lists shape : batch_size x N_i (N_i is number of filtered products for each sample)
    #         product_lists = find_filtered_prod(meta_info_df, top1k_brand_name,
    #                                            top1k_color_name, top1k_hightop_name, top1k_sole_name)
    #
    #         # Last zeroshot with filtered name
    #         correct_count = 0
    #         prod_not_classified_list = []
    #         prod_filtered_wrong = []
    #         prod_failed_list = []
    #         # adidas = ['Adidas Samba Vegan Black', 'Adidas Superstar 82 Black']
    #
    #         for img_idx, prod_list_meta, target in zip(range(image_features.size(0)), product_lists, target_name):
    #
    #             prod_list, brand, color, hightop, sole = prod_list_meta['prod_list'], \
    #                                                      prod_list_meta['brand'], \
    #                                                      prod_list_meta['color'], \
    #                                                      prod_list_meta['hightop'], \
    #                                                      prod_list_meta['sole']
    #             # get name of target
    #             tar_name = name_inv_dict[target]
    #
    #             # brand, color, hightop condition are wrong.
    #             if tar_name not in prod_list:
    #                 prod_failed_list.append((img_idx, target))
    #                 prod_filtered_wrong.append(tar_name)
    #                 continue
    #
    #             target_idx = torch.LongTensor([prod_list.index(tar_name)]).to(device)
    #             zeroshot_weight = text_precompute.compute_prompt_name(prod_list, brand, color, hightop, sole, False)
    #             # if tar_name in adidas:
    #             #     zeroshot_weight = text_precompute.compute_prompt_name(prod_list, brand, color, hightop, sole, True)
    #             # else:
    #             #     zeroshot_weight = text_precompute.compute_prompt_name(prod_list,brand,color,hightop,sole, False)
    #             image_feature = image_features[img_idx]
    #             logits = (100.0 * image_feature @ zeroshot_weight)
    #             pred = logits.topk(1, 0, True, True)[1].t().flatten().item()
    #             if pred == target_idx:
    #                 correct_count += 1
    #             else:
    #                 prod_failed_list.append((img_idx, target))
    #                 prod_not_classified_list.append(tar_name)
    #
    #         prod_classify_again_list = []
    #         for img_idx, target in prod_failed_list:
    #             image_feature = image_features[img_idx]
    #             logits = (100.0 * image_feature @ name_weights)
    #             pred = logits.topk(1, 0, True, True)[1].t().flatten().item()
    #             if pred == target:
    #                 correct_count += 1
    #             else:
    #                 prod_classify_again_list.append(name_inv_dict[target])
    #
    #         acc1 = correct_count / preproc_image.size(0)
    #         print(f'{i + 1}th batch Zeroshot performance: {acc1 * 100:.2f}')
    #         print(f'1. FAIL : Filter items with brand, color, hightop: {prod_filtered_wrong}')
    #         print(f'2. FAIL : Classifier with Filtered items: {prod_not_classified_list}')
    #         print(f'3. FAIL : Classify with name again about failed items: {prod_classify_again_list}')
    #
    #         zeroshot_correct_count += correct_count
    #
    #         total_num += preproc_image.size(0)
    #
    #     print_acc('brand', total_num, brand_top1, brand_top5)
    #     print_acc('color', total_num, color_top1, color_top5)
    #     print_acc('hightop', total_num, hightop_top1, hightop_top5)
    #     print_acc('sole', total_num, sole_top1, sole_top5)
    #     print_acc('name_zeroshot', total_num, name_top1, name_top5)
    #     print_acc('zeroshot', total_num, zeroshot_correct_count)


    logits = CustomCLIP(name_dict, model)
    print()
    print("@@@@@@@@@@@")
    # print(logits)

if __name__ == '__main__':
    root_path = "dataset"
    meta_info_path = "meta_info.csv"
    main(root_path, meta_info_path)
    print()
    print("I reached here!")