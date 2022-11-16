import yaml
import torch
from utils import update_dict

class TextPreCompute:
    def __init__(self,
                 model,
                 device,
                 prompt_path:str,
                 name_dict:dict,
                 brand_dict:dict,
                 color_dict:dict,
                 hightop_dict:dict,
                 verbose = True
                 ):

        self.model = model
        self.device = device
        self.name_classes = name_dict.keys()
        self.brand_classes = brand_dict.keys()
        self.color_classes = color_dict.keys()
        self.hightop_classes = hightop_dict.keys()
        self.prompt_dict = self._load_prompt_template(prompt_path,model,device)

        if verbose:
            self._print_classes_info()

        self.name_weights = self._precompute_prompt_text(self.name_classes, prompt_dict['name'])
        self.brand_weights = self._precompute_prompt_text(self.brand_classes, prompt_dict['brand'])
        self.color_weights = self._precompute_prompt_text(self.color_classes, prompt_dict['color'])
        self.hightop_weights = self._precompute_prompt_text(self.hightop_classes, prompt_dict['hightop'])

    def get_precomputed_text(self):
        return self.name_weights, self.brand_weights, self.color_weights, self.hightop_weights

    def _print_classes_info(self):
        print(f'# name_classes: {len(self.name_classes)}.\n'
              f'# brand_classes: {len(self.brand_classes)}.\n'
              f'# color_classes: {len(self.color_classes)}\n'
              f'# hightop_classes: {len(self.hightop_classes)}\n')

    def _load_prompt_template(self, prompt_path):
        with open(prompt_path) as f:
            prompt_dict = yaml.load(f, Loader=yaml.FullLoader)
        return prompt_dict

    def _precompute_prompt_text(self, classnames, templates, model, deivce):
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
