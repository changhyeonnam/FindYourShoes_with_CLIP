# Find Your Shoes with CLIP

This repository is about Find Your Shoes using [CLIP(Contrastive Language-Image PreTraining)](https://github.com/openai/CLIP) model which is from openai. We developed this service which is inspired by Google's image search and NAVER OmniSearch. 

We thought it would be a meaningful service if we could search for the shoes by image when we don't know the name of the shoes, and search again by changing some features in the shoes. So we developed the service by limiting the dataset to shoes.

For example, when text for different color info from original color and a user's shoes image were given as an input, the model finds the same kind of shoes in the given text color.

This Project is ongoing which is completed by 2022.12. This is to-do-list about our development.

### **To do list for our project**

- [x]  Develop crawler for collecting full size image from goolge.
- [x]  Constructing small size of shoes dataset. (4 brands(Converse, Nike, Adidas, Vans),  28 shoes (7 shoes for each brand), total number of shoes is 1216)
- [x]  Applying prompt ensemble and run experiment with small dataset.
- [x]  Applying prompt ensemble for inference.
- [x]  Make demo in Command Line Interface.
- [ ]  Constructing Large size of shoes dataset (ongoing, 7 brands(Nike, Converse, ASCIS, FILA, New Balance), 105 shoes (15 shose for each brands), at least 5k images.) (ongoing).
- [ ]  Applying prompt ensemble and run experiment with Large dataset (ongoing)
- [ ]  Applying CoOp, CoOpOp for prompt learning to improve performace (ongoing).
- [ ]  Use Few shot method for inference to improve performance (ongoing).
- [ ]  Implement demo program in web interface using [streamlit](https://streamlit.io/).

## 1. Dataset

There were not existing shoes labeled dataset which include various features (e.g, brand, color, hightop, sole). And also, there were not stable crawler for crawling full size image from google. So we made our own [Crawler](https://github.com/changhyeonnam/Google-Full-size-image-crawler) using python, selenium. 

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

---

## 3.Method

### 3.1 Prompt Ensemble

우리는 기본적으로  5단계에 걸쳐서 모델을 실험하고, 성능 향상시키고 있다. 

1. 신발 이미지가 주어져있다고 하자. 주어진 신발 이미지와 text prompt ensemble을 통해 brand(nike, adidas,..), color(red, blue, yellow, ..) , hightop(low or high)  각각에 대해 similiarity score를 계산한 뒤, top 1 score를 가진 feature를 뽑아내어 분류한다.
2. 미리 만들어진 신발 아이템별로 Feature를 적어놓은 테이블(csv file)에서 3개의 Feature에 부합하는 신발들만을 filtering을한다.  
3. 분류된 Feature(brand, color, hightop)와 filtering된 신발 이름에 prompt ensemble을 적용하여, 첫번째 similarity score를 계산한다. 
4. 전체 아이템의 신발 이름을 prompte ensemble을 적용하여 두번째 similiarity score를 계산한다.
5. 3번과 4번에서 계산된 similarity score 중 높은 score를 가진 신발 종류를 선택한다. 

Small size dataset에 대한 실험에서 96.3%의 정확도를 얻었고, 각 feature에 대한 정확도는 다음과 같다.

```jsx
brand : top1 Accuracy = 96.95723684210526%, top5 Accuracy = 96.95723684210526%
color : top1 Accuracy = 88.56907894736842%, top5 Accuracy = 100.0%
hightop : top1 Accuracy = 83.05921052631578%, top5 Accuracy = 100.0%
name_zeroshot : top1 Accuracy = 88.32236842105263%, top5 Accuracy = 99.91776315789474%
zeroshot : top1 Accuracy = 96.38157894736842%
```

- **Experiment about Few shot method for inference, Prompt learning (CoOp, CoOpOp) is not completed.**
- **Experiment with Large Dataset is not completed.**

## Quick Start

### Download Dataset

```jsx
sh dataset.sh
```

### Before run the model, You should generate table for shoes table.

```jsx
python3 csv_generator.py
```

### Experiment for evaluing model

```jsx
sh run_experiment.sh
```

### Inference

For inference you should locate your image in img folder and run below command.

```jsx
sh inference.sh
```

## Demo (Command Line Interface)

## Reference

1. Clip
2. Prompt ensemble
3. CoOp
4. CoOpOp