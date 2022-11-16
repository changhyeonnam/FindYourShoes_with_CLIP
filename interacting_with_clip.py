import numpy as np
import torch
import clip
from tqdm.notebook import tqdm
from pkg_resources import packaging
from PIL import Image
import matplotlib.pyplot as plt
import skimage
import os

# for urllib error
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def print_clip_info(model):
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)


if __name__=='__main__':
    print(clip.available_models())
    model, preprocess = clip.load("ViT-B/32")
    print_clip_info(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # images in skimage to use and their textual descriptions
    descriptions = {
        "converse_high_black": "a photo of converse high and color is black",
        "converse_high_white": "a photo of converse high and color is white",
        "converse_high_blue": "a photo of converse high and color is blue",
        "converse_low_green": "a photo of converse low and color is green",
        "converse_low_deep_bordeaux": "a photo of converse low and color is ruby red",
        "nike_dunk_low_retro_gym_red": "a photo of nike dunk low and color is red",
        "nike_air_force_1_low_white": "a photo of nike air force 1 low and color is white",
        "nike_sacai_blazer_low_green": "a photo of nike sacai blazer low and color is green",
    }
    labels =["converse_high_black", "converse_high_white", "converse_high_blue", "converse_low_green","converse_low_deep_bordeaux",
             "nike_dunk_low_retro_gym_red","nike_air_force_1_low_white","nike_sacai_blazer_low_green"]
    original_images = []
    images = []
    texts = []
    plt.figure(figsize=(16, 5))
    PATH_IMG = 'img'
    for filename in [filename for filename in os.listdir(PATH_IMG) if
                     filename.endswith(".png") or filename.endswith(".jpeg")]:
        name = os.path.splitext(filename)[0]
        if name not in descriptions:
            continue

        image = Image.open(os.path.join(PATH_IMG,filename)).convert("RGB")

        plt.subplot(2, 4, len(images) + 1)
        plt.imshow(image)
        plt.title(f"{filename}\n{descriptions[name]}")
        plt.xticks([])
        plt.yticks([])

        original_images.append(image)
        images.append(preprocess(image))
        texts.append(descriptions[name])

    plt.tight_layout()


    image_input = torch.tensor(np.stack(images)).to(device)
    text_tokens = clip.tokenize(["This is " + desc for desc in texts]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

    count = len(descriptions)

    plt.figure(figsize=(20, 14))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    # plt.colorbar()
    plt.yticks(range(count), texts, fontsize=18)
    plt.xticks([])
    for i, image in enumerate(original_images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])

    plt.title("Cosine similarity between text and image features", size=20)

    # plt.show()

    # zero-shot image classification
    text_descriptions = [v for k,v in descriptions.items()]
    text_tokens = clip.tokenize(text_descriptions).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)

    plt.figure(figsize=(16, 16))

    for i, image in enumerate(original_images):
        plt.subplot(4, 4, 2 * i + 1)
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(4, 4, 2 * i + 2)
        y = np.arange(top_probs.shape[-1])
        plt.grid()
        plt.barh(y, top_probs[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [labels[index] for index in top_labels[i].numpy()])
        plt.xlabel("probability")

    plt.subplots_adjust(wspace=0.5)
    plt.show()

