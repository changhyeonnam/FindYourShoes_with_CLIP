# Find Your Shoes with CLIP (On-going project)

This repository is about Find Your Shoes using [CLIP(Contrastive Language-Image PreTraining)](https://github.com/openai/CLIP) model which is from OpenAI. 

We thought it would be a meaningful service if we could search for the shoes by image when we don't know the name of the shoes, and search again by changing some features in the shoes. 
So we developed the service by limiting the dataset to shoes.

For example, when text for different color info from original color and a user's shoes image were given as an input, the model finds the same kind of shoes in the given text color.

We developed this service inspired by [Google's image search](https://images.google.com/). 
We found that our service is very similar to [NAVER OmniSearch](https://www.youtube.com/watch?v=jfGpplvNFFs) but we developed this service because it could be challenging and fun to implement.

This Project is ongoing which is completed by 2022.12. This is to-do-list about our development.

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

Small size dataset에 대한 실험에서 96.3%의 정확도를 얻었고, 각 feature에 대한 정확도는 다음과 같다.

```bash
brand : top1 Accuracy = 96.95723684210526%, top5 Accuracy = 96.95723684210526%
color : top1 Accuracy = 88.56907894736842%, top5 Accuracy = 100.0%
hightop : top1 Accuracy = 83.05921052631578%, top5 Accuracy = 100.0%
name_zeroshot : top1 Accuracy = 88.32236842105263%, top5 Accuracy = 99.91776315789474%
zeroshot : top1 Accuracy = 96.38157894736842%
```

- **Experiment with Large Dataset.**

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
For inference you should locate your image in img folder and run below command.

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


## Reference
1. [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
1. [CLIP openai blog post](https://openai.com/blog/clip/)
2. [Multi-prompt](http://pretrain.nlpedia.ai/data/pdf/multi-prompt.pdf)
3. [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/abs/2110.07602)
4. [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134)