import yaml
import torch
from utils import update_dict
import clip
from data import ShoesImageDataset
from tqdm import tqdm

class TextPreCompute:
    def __init__(self,
                 model,
                 device,
                 prompt_path:str,
                 name_dict:dict,
                 brand_dict:dict,
                 color_dict:dict,
                 hightop_dict:dict,
                 sole_dict:dict,
                 verbose = True
                 ):

        self.model = model
        self.device = device
        self.name_classes = name_dict.keys()
        self.brand_classes = brand_dict.keys()
        self.color_classes = color_dict.keys()
        self.hightop_classes = hightop_dict.keys()
        self.sole_classes = sole_dict.keys()
        self.prompt_dict = self._load_prompt_template(prompt_path)

        if verbose:
            print(f'\n{"*" * 10} Preprocessing about prompt template is Started. {"*" * 10}\n')
            self._print_prompt_template(self.prompt_dict)

        self.name_weights = self._precompute_prompt_text(self.name_classes, self.prompt_dict['name'], model, device)
        self.brand_weights = self._precompute_prompt_text(self.brand_classes, self.prompt_dict['brand'], model, device)
        self.color_weights = self._precompute_prompt_text(self.color_classes, self.prompt_dict['color'], model, device)
        self.hightop_weights = self._precompute_prompt_text(self.hightop_classes, self.prompt_dict['hightop'], model, device)
        self.sole_weights = self._precompute_prompt_text(self.sole_classes, self.prompt_dict['sole'], model, device)
        if verbose:
            print(f'\n{"*"*10} Preprocessing about prompt template is Completed. {"*"*10}\n')

    def get_precomputed_text(self):
        return self.name_weights, self.brand_weights, self.color_weights, self.hightop_weights, self.sole_weights

    def compute_prompt_name(self, classnames,brand,color,hightop):
        templates = self.prompt_dict['zeroshot']
        device = self.device
        model = self.model
        with torch.no_grad():
            encoded_text_weights = []
            for classname in classnames:
                texts = []
                for i, template in enumerate(templates):
                    if i == 0:
                        texts.append(template.format(classname,brand,color,hightop))
                    # elif i==1:
                    #     texts.append(template.format(brand, classname,color,hightop,sole))
                    elif i==1:
                        texts.append(template.format(brand,color,classname,hightop))
                    # elif i==3:
                    #     texts.append(template.format(brand,color,hightop,classname,sole))
                    elif i==2:
                        texts.append(template.format(brand,color,hightop,classname))


                texts = clip.tokenize(texts).to(device)  # tokenize
                class_embeddings = model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                encoded_text_weights.append(class_embedding)
            encoded_text_weights = torch.stack(encoded_text_weights, dim=1).to(device)
        return encoded_text_weights

    def _print_prompt_template(self, prompt_dict):
        for k,v in prompt_dict.items():
            print(f'This is prompt templates of {k}')
            print(v)


    def _load_prompt_template(self, prompt_path):
        with open(prompt_path) as f:
            prompt_dict = yaml.load(f, Loader=yaml.FullLoader)
        return prompt_dict

    def _precompute_prompt_text(self, classnames, templates, model, device):
        with torch.no_grad():
            encoded_text_weights = []
            for classname in tqdm(classnames, total=len(classnames)):
                texts = [template.format(classname) for template in templates]  # format with class
                texts = clip.tokenize(texts).to(device)  # tokenize
                class_embeddings = model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                encoded_text_weights.append(class_embedding)
            encoded_text_weights = torch.stack(encoded_text_weights, dim=1).to(device)
        return encoded_text_weights


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model, preprocess = clip.load("ViT-B/32")

    ROOT_PATH = "converse dataset"
    meta_info_path = "legacy/example.csv"
    prompt_path = "config/prompt_template.yaml"

    dataset = ShoesImageDataset(root=ROOT_PATH,
                                preprocess=preprocess,
                                meta_info_path=meta_info_path,
                                verbose=True)


    # get dictionary about shoes.
    name_dict, brand_dict, color_dict, hightop_dict, meta_dict = dataset.get_dict()

    # precompute text and prompt template with clip moodel.
    encoded_text = TextPreCompute(model, device, prompt_path,
                                  name_dict,
                                  brand_dict,
                                  color_dict,
                                  hightop_dict)

    name_weights, brand_weights, color_weights, hightop_weights = encoded_text.get_precomputed_text()
