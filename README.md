# VAP-Former

This is the official implementation of [Visual-Attribute Prompt Learning for Progressive Mild Cognitive Impairment Prediction](https://arxiv.org/abs/2310.14158
) at MICCAI-2023

## Table of Contents

- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
- [Pretrained Model](#pretrained-model)
- [How to train](#how-to-train)
- [How to transfer](#how-to-transfer)
- [Acknowledgement](#acknowledgement)

## Requirements
```bash
conda create -n VAPFormer python=3.7
conda activate VAPFormer
pip install -r requirements.txt
```

## Dataset Preparation
The data can be accessed at http://adni.loni.usc.edu/data-samples/access-data/ 
You can use SPM12,CAT12 to preprocess data

## Pre-trained Model
| Task | Link |
|------|------|
| AD v.s. NC | [OneDrive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222010514_link_cuhk_edu_cn/ESvGnEWjvylGgzMiFYhZdo0BVwQqx37KJEWBFvfZ45NosA?e=fOx6rr)|
| sMCI v.s. pMCI | [OneDrive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222010514_link_cuhk_edu_cn/EQwKgE9I_pVDkguPeA1GTQoBUUmV2ocYwxIqop5oFoLdYw?e=LxFUh0)|



## How to train
Modify the code in trainclinical.py

with the following two lines of code

`ld_helper = LoaderHelper(task=Task.NC_v_AD)` # which defineds the task

`model_uuid = train_camull(ld_helper, epochs=50)` # function to train the model

The weights are stored in the weights file

## How to transfer

Modify the code in trainclinical.py as follows:


`ld_helper = LoaderHelper(task=Task.sMCI_v_pMCI)`

`model = load_model()` # Modify the pretrained weight

`model_uuid  = train_camull(ld_helper, model=model, epochs=50)`

`evaluate_model(DEVICE, model_uuid, ld_helper)`

## Acknowledgement
This project is built upon the foundations of several open-source codebases, including [camull-net](https://github.com/McSpooder/camull-net) and [VPT](https://github.com/kmnp/vpt).

