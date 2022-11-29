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




def main(root_path, meta_info_path, prompt_path):
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

    print()
    print("!!!!!!!!!!!!!!!")
if __name__ == '__main__':
    root_path = "dataset"
    meta_info_path = "meta_info.csv"
    main(root_path, meta_info_path, prompt_path=None)
    print()
    print("I reached here!")

