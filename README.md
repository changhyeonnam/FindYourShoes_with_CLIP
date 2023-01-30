# Find Your Shoes with CLIP (On-going project)

This repository is about Find Your Shoes using [CLIP(Contrastive Language-Image PreTraining)](https://github.com/openai/CLIP) model which is from OpenAI. 

We thought it would be a meaningful service if we could search for the shoes by image when we don't know the name of the shoes, and search again by changing some features in the shoes. 
So we developed the service by limiting the dataset to shoes.

For example, when text for different color info from original color and a user's shoes image were given as an input, the model finds the same kind of shoes in the given text color.

We developed this service inspired by [Google's image search](https://images.google.com/). 
We found that our service is very similar to [NAVER OmniSearch](https://www.youtube.com/watch?v=jfGpplvNFFs) but we developed this service because it could be challenging and fun to implement.

This Project is ongoing which is completed by 2023.1. This is to-do-list about our development.
Our final presentation slide is completed!. This is link : [slide link](https://docs.google.com/presentation/d/1wV-ke1FDVbulnFdVXPnU6Sb2U8KFbmP5AiH0pkGLwSY/edit?usp=sharing)

(I need to refactorize codes. Current codes are messy.)

### **To do list for our project**

- [x]  Develop crawler for collecting full size image from goolge.
- [x]  Constructing small size of shoes dataset. (4 brands(Converse, Nike, Adidas, Vans),  28 shoes (7 shoes for each brand), total number of shoes is 1216)
- [x]  Applying prompt ensemble and run experiment with small dataset.
- [x]  Applying prompt ensemble for inference.
- [x]  Make demo in Command Line Interface.
- [x]  Constructing Large size of shoes dataset (ongoing, 7 brands(Nike, Converse, ASCIS, FILA, New Balance), 105 shoes (15 shose for each brands), at least 5k images.) (ongoing).
- [x]  Applying prompt ensemble and run experiment with Large dataset.
- [x]  Applying CoOp(Context Optimization) for prompt learning to improve performance.
- [x]  Use Few shot method for inference to improve performance.
- [ ]  Implement demo program in web interface using [streamlit](https://streamlit.io/) (ongoing).

## Demo
1. Demo with Jordan 1 Retro High og Chicago (image : Jordan 1 Retro High og Chicago, text : I want same brand but color is different)
    ![demo_1](demo_1.gif)
2. Demo with New Balance 574 (image : New Balance 574, text : I want similar shoes but brand is different)
    ![demo_2](demo_2.gif)


## Dataset

There were not existing shoes labeled dataset which include various features (e.g, brand, color, hightop, sole). 
And also, there were not stable crawler for crawling full size image from google. 
So we made our own [Crawler](https://github.com/changhyeonnam/Google-Full-size-image-crawler) using python, selenium.
For filtering crawled dataset, we made crawling Rule for our dataset. We followed this rule for crawling and filtering images. 

### Crawling Rule

1. The number of shoes (one or two) doesn't matter.
2. Exclude photos which do not show the entire shoe appearance.
3. Exclude photos that cannot be identified whether they are high-top or low.
4. Exclude photos of shoes with heels only.
5. Crawling only for human-recognizable photos about the brand, color, high top, and sole features.
6. It doesn't matter if a person is wearing it.
7.  The background doesn't matter.
8. It's good to have as many angles as possible.

**We will not use datasets for commercial purposes and we are going to share dataset when it is collected.**

## Method

### 1. Prompt Ensemble

We basically experimented with the model over five steps. Let's assume that Shoes image is given.
1. Put the given shoe image and promptenseble together in the clip model to obtain each simularity score for brand (nike, adidas,...), color (red, blue, yellow,...), and heightop (low or high). For each type, a feature with a top 1 score is extracted and classified.
2. Filter the shoes that does not match the selected feature in the shoes table.
3. Calculate the similarity score by applying the prompt ensemble  to the classified features (brand, color, hightop) and filtered shoe names.
4. Calculate the simiarity score by applying prompt ensemble to the entire shoe name.
5. Among the similarity scores calculated in 3 and 4, select the shoe type with the highest score.

- **Experiment with Large Dataset.**
- Inference without prompt learner

|  | Brand  | Color | Hightop | Sole | Name |
| --- | --- | --- | --- | --- | --- |
| Top 1 | 89.75 | 59.23 | 94.69 | 14.71 | 44.48 |
| Top 5 | 99.87 | 93.65 | 100 | 99.76 | 78.77 |
- Inference with prompt learner

|  | Brand  | Color | Hightop | Sole | Name |
| --- | --- | --- | --- | --- | --- |
| Top 1 | 97.47 | 95 | 98.43 | 99.25 | 93.19 |
| Top 5 | 99.96 | 99.65 | 100 | 100 | 99.52 |

## Quick Start

### Download Dataset
(not implemented yet. it will be added.)
```bash
sh dataset.sh
```

### Before run the model, You should generate table for shoes information.
(This require dataset. )
```bash
python3 csv_generator.py
```

### Experiment for evaluating model
(This require dataset. )

**using only prompt ensemble version(not CoOp)**
```bash
python3 main.py
```
**using CoOp**
```bash
cd CoOp_trainer
python3 exp_CoOp.py
```

### Inference
For inference, you should locate your image in img folder and run below command.

(This require dataset. )

**using only prompt ensemble version(not CoOp)**
```bash
python3 infer.py
```
**using CoOp**
```bash
cd CoOp_trainer
python3 CoOp_infer.py
```

## Members
- [Changhyeon Nam](https://github.com/changhyeonnam)
- [Heeje Kim](https://github.com/beaglemong)
- [Haejun Bae](https://github.com/hj1132)

## Reference
1. [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
1. [CLIP openai blog post](https://openai.com/blog/clip/)
2. [Multi-prompt](http://pretrain.nlpedia.ai/data/pdf/multi-prompt.pdf)
3. [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/abs/2110.07602)
4. [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134)
