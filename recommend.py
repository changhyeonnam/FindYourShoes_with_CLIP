import yaml
import torch
from utils import update_dict
import clip
import pandas as pd
from data import ShoesImageDataset
from tqdm import tqdm
# from prompt_compute import TextPreCompute
from torch.nn import CosineSimilarity as cos
from PIL import Image

class ComputeSimilarity:
    def __init__(self,
                 prompt_path:str
                 ):
        
        self.prompt_dict = self._load_prompt_template(prompt_path)

        
    def features(model, ):
        texts = clip.tokenize(texts).to(device)
        embeddings = model.encode_text(texts)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        embedding = embeddings.mean(dim=0)
        embedding /= embedding.norm()

    def second_similar(name_list, prompt):    
        name_list = name_list.remove(prompt)    # exclude the correct label
        
    def image_feature(name_list, image):
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        
    def sim_images(model, image):
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_2_features /= image_2_features.norm(dim=-1, keepdim=True)
        similarity = image_2_features.cpu().numpy() @ image_features.cpu().numpy().T
        g
        return similarity
        
        

if __name__ == "__main__":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, _ = clip.load("ViT-B/32")
        
        meta_info_path = "legacy/example.csv"
        prompt_path = "config/prompt_template.yaml"
        rep_data_path = "representative dataset"
        input_img_path = " "    # input image directory
        
        csv = pd.read_csv(meta_info_path)
        names = csv['name'] # get shoes name list from 'meta_info.csv'
        input_img = Image.open(input_img_path)
        
        try_(model, input_img)
        
        
        