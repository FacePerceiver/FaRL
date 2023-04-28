# *FaRL* for *Fa*cial *R*epresentation *L*earning
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/general-facial-representation-learning-in-a/face-alignment-on-300w)](https://paperswithcode.com/sota/face-alignment-on-300w?p=general-facial-representation-learning-in-a)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/general-facial-representation-learning-in-a/face-alignment-on-aflw-19)](https://paperswithcode.com/sota/face-alignment-on-aflw-19?p=general-facial-representation-learning-in-a)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/general-facial-representation-learning-in-a/face-alignment-on-wflw)](https://paperswithcode.com/sota/face-alignment-on-wflw?p=general-facial-representation-learning-in-a)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/general-facial-representation-learning-in-a/face-parsing-on-celebamask-hq)](https://paperswithcode.com/sota/face-parsing-on-celebamask-hq?p=general-facial-representation-learning-in-a)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/general-facial-representation-learning-in-a/face-parsing-on-lapa)](https://paperswithcode.com/sota/face-parsing-on-lapa?p=general-facial-representation-learning-in-a)

This repo hosts official implementation of our CVPR2022 paper "[**General Facial Representation Learning in a Visual-Linguistic Manner**](https://arxiv.org/abs/2112.03109)".
## Updates

- [21/06/2022] [LAION-Face](https://github.com/FacePerceiver/LAION-Face) dataset was released.
- [10/03/2022] FaRL was accepted by CVPR 2022 as Oral presentation.
- [02/03/2022] [facer](https://github.com/FacePerceiver/facer) was released. It is a face related toolkit build upon FaRL.

## Introduction

**FaRL** offers powerful pre-training transformer backbones for face analysis tasks. Its pre-training combines both the image-text contrastive learning and the masked image modeling.

<img src="./figures/framework2.jpg" alt="framework" width="800"/>

After the pre-training, the image encoder can be utilized for various downstream face tasks. 

## Pre-trained Backbones

We offer different pre-trained transformer backbones as below.

| Model Name  |  Data | Epoch | Link |
| ----------- | -------------- | ----- | ---- |
| FaRL-Base-Patch16-LAIONFace20M-ep16 (used in paper) | [LAION Face 20M](https://github.com/FacePerceiver/LAION-Face)  | 16 | [OneDrive](https://1drv.ms/u/s!AperexS2nqQomyPsG2M4uPXay7Au); [Baidu](https://pan.baidu.com/s/162I6cfIYvyz7tUz4zSSuFA) Key: wu7p |
| FaRL-Base-Patch16-LAIONFace20M-ep64 | [LAION Face 20M](https://github.com/FacePerceiver/LAION-Face) | 64 | [OneDrive](https://1drv.ms/u/s!AperexS2nqQom0Zu3lsuM28UbEgP); [Baidu](https://pan.baidu.com/s/1fCjKPpwhqz7gF-GjA3O0vA) Key: mgau |

## Use FaRL as FaceCLIP

We provied both the pretrained text encoder and the image encoder. As FaRL shares the same network structure as CLIP, you can load the weights of FaRL using exactly the same network structure as CLIP VIT-B16, and use it exactly like CLIP. Here are the code sample modified from CLIP.
```
import torch
import clip
from PIL import Image

device ="cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device="cpu")
model = model.to(device)
farl_state=torch.load("FaRL-Base-Patch16-LAIONFace20M-ep16.pth") # you can download from https://github.com/FacePerceiver/FaRL#pre-trained-backbones
model.load_state_dict(farl_state["state_dict"],strict=False)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  
```

## Setup Downstream Training

We run all downstream trainings on 8 NVIDIA GPUs (32G). Our code supports other GPU configurations, but we do not guarantee the resulting performances on them.
Before setting up, install these packages:
* [PyTorch](https://pytorch.org/get-started/previous-versions/) 1.7.0
* [MMCV-Full](https://github.com/open-mmlab/mmcv) 1.3.14

Then, install the rest dependencies with `pip install -r ./requirement.txt`.

Please refer to [./DS_DATA.md](./DS_DATA.md) to prepare the training and testing data for downstream tasks.

Download [the pre-trained backbones](#pre-trained-backbones) into `./blob/checkpoint/`.
Now you can launch the downstream trainings & evaluations with following command template.

```
python -m blueprint.run \
  farl/experiments/{task}/{train_config_file}.yaml \
  --exp_name farl --blob_root ./blob
```

The repo has included some config files under `./farl/experiments/` that perform finetuning for face parsing and face alignment.
For example, if you would like to launch a face parsing training on LaPa by finetuning our `FaRL-Base-Patch16-LAIONFace20M-ep16` pre-training, simply run with:

```
python -m blueprint.run \
  farl/experiments/face_parsing/train_lapa_farl-b-ep16_448_refinebb.yaml \
  --exp_name farl --blob_root ./blob
```

Or if you would like to launch a face alignment training on 300W by finetuning our `FaRL-Base-Patch16-LAIONFace20M-ep16` pre-training, you can simply run with:

```
python -m blueprint.run \
  farl/experiments/face_alignment/train_ibug300w_farl-b-ep16_448_refinebb.yaml \
  --exp_name farl --blob_root ./blob
```

It is also easy to create new config files for training and evaluation on your own. For example, you can customize your own face parsing task on CelebAMask-HQ by editing the values below (remember to remove the comments before running).

```yaml
package: farl.experiments.face_parsing

class: blueprint.ml.DistributedGPURun
local_run:
  $PARSE('./trainers/celebm_farl.yaml', 
    cfg_file=FILE,
    train_data_ratio=None, # The data ratio used for training. None means using 100% training data; 0.1 means using only 10% training data.
    batch_size=5, # The local batch size on each GPU.
    model_type='base', # The size of the pre-trained backbone. Supports 'base', 'large' or 'huge'.
    model_path=BLOB('checkpoint/FaRL-Base-Patch16-LAIONFace20M-ep16.pth'), # The path to the pre-trained backbone.
    input_resolution=448, # The input image resolution, e.g 224, 448. 
    head_channel=768, # The channels of the head.
    optimizer_name='refine_backbone', # The optimization method. Should be 'refine_backbone' or 'freeze_backbone'.
    enable_amp=False) # Whether to enable float16 in downstream training.
```

## Performance

The following table illustrates the performances of our `FaRL-Base-Patch16-LAIONFace20M-ep16` pre-training, which is pre-trained with 16 epoches, both reported in the paper (Paper) and reproduced using this repo (Rep). There are small differences between their performances due to code refactorization.

| Name | Task | Benchmark | Metric | Score (Paper/Rep) | Logs (Paper/Rep) |
| ---- | ---- | ---- | --- | --- | --- |
| [face_parsing/<br/>train_celebm_farl-b-ep16-448_refinebb.yaml](./farl/experiments/face_parsing/train_celebm_farl-b-ep16_448_refinebb.yaml) | Face Parsing  | CelebAMask-HQ | F1-mean ⇑ | 89.56/89.65 | [Paper](./logs/paper/face_parsing.train_celebm_farl-b-ep16-448_refinebb), [Rep](./logs/reproduce/face_parsing.train_celebm_farl-b-ep16_448_refinebb) |
| [face_parsing/<br/>train_lapa_farl-b-ep16_448_refinebb.yaml](./farl/experiments/face_parsing/train_lapa_farl-b-ep16_448_refinebb.yaml) | Face Parsing | LaPa | F1-mean ⇑ | 93.88/93.86 | [Paper](./logs/paper/face_parsing.train_lapa_farl-b-ep16_448_refinebb), [Rep](./logs/reproduce/face_parsing.train_lapa_farl-b-ep16_448_refinebb) |
| [face_alignment/<br/>train_aflw19_farl-b-ep16_448_refinebb.yaml](./farl/experiments/face_alignment/train_aflw19_farl-b-ep16_448_refinebb.yaml) | Face Alignment | AFLW-19 (Full) | NME_diag ⇓ | 0.943/0.943 | [Paper](./logs/paper/face_alignment.train_aflw19_farl-b-ep16_448_refinebb), [Rep](./logs/reproduce/face_alignment.train_aflw19_farl-b-ep16_448_refinebb) |
| [face_alignment/<br/>train_ibug300w_farl-b-ep16_448_refinebb.yaml](./farl/experiments/face_alignment/train_ibug300w_farl-b-ep16_448_refinebb.yaml) | Face Alignment | 300W (Full) | NME_inter-ocular ⇓ | 2.93/2.92 | [Paper](./logs/paper/face_alignment.train_ibug300w_farl-b-ep16_448_refinebb), [Rep](./logs/reproduce/face_alignment.train_ibug300w_farl-b-ep16_448_refinebb) |
| [face_alignment/<br/>train_wflw_farl-b-ep16_448_refinebb.yaml](./farl/experiments/face_alignment/train_wflw_farl-b-ep16_448_refinebb.yaml) | Face Alignment | WFLW (Full) | NME_inter-ocular ⇓ | 3.96/3.98 | [Paper](./logs/paper/face_alignment.train_wflw_farl-b-ep16_448_refinebb), [Rep](./logs/reproduce/face_alignment.train_wflw_farl-b-ep16_448_refinebb) |


Below we also report results of our new `FaRL-Base-Patch16-LAIONFace20M-ep64`, which is pre-trained with 64 epoches instead of 16 epoches as above, showing further improvements on most tasks.

| Name | Task | Benchmark | Metric | Score | Logs |
| ---- | ---- | ---- | --- | --- | --- |
| [face_parsing/<br/>train_celebm_farl-b-ep64-448_refinebb.yaml](./farl/experiments/face_parsing/train_celebm_farl-b-ep64_448_refinebb.yaml) | Face Parsing  | CelebAMask-HQ | F1-mean ⇑ | 89.57 | [Rep](./logs/reproduce/face_parsing.train_celebm_farl-b-ep64_448_refinebb) |
| [face_parsing/<br/>train_lapa_farl-b-ep64_448_refinebb.yaml](./farl/experiments/face_parsing/train_lapa_farl-b-ep64_448_refinebb.yaml) | Face Parsing | LaPa | F1-mean ⇑ | 94.04 | [Rep](./logs/reproduce/face_parsing.train_lapa_farl-b-ep64_448_refinebb) |
| [face_alignment/<br/>train_aflw19_farl-b-ep64_448_refinebb.yaml](./farl/experiments/face_alignment/train_aflw19_farl-b-ep64_448_refinebb.yaml) | Face Alignment | AFLW-19 (Full) | NME_diag ⇓ | 0.938 | [Rep](./logs/reproduce/face_alignment.train_aflw19_farl-b-ep64_448_refinebb) |
| [face_alignment/<br/>train_ibug300w_farl-b-ep64_448_refinebb.yaml](./farl/experiments/face_alignment/train_ibug300w_farl-b-ep64_448_refinebb.yaml) | Face Alignment | 300W (Full) | NME_inter-ocular ⇓ | 2.88 | [Rep](./logs/reproduce/face_alignment.train_ibug300w_farl-b-ep64_448_refinebb) |
| [face_alignment/<br/>train_wflw_farl-b-ep64_448_refinebb.yaml](./farl/experiments/face_alignment/train_wflw_farl-b-ep64_448_refinebb.yaml) | Face Alignment | WFLW (Full) | NME_inter-ocular ⇓ | 3.88 | [Rep](./logs/reproduce/face_alignment.train_wflw_farl-b-ep64_448_refinebb) |


<!-- We also report results using the 50M pre-trained backbone, showing further enhancement on LaPa and AFLW-19.

| Config | Task | Benchmark | Metric | Score | Logs |
| ---- | ---- | ---- | --- | --- | --- |
| [face_parsing/<br/>train_celebm_farl-b-50m-ep16-448_refinebb.yaml](./farl/experiments/face_parsing/train_celebm_farl-b-50m-ep16_448_refinebb.yaml) | Face Parsing  | CelebAMask-HQ | F1-mean ⇑ | 89.68 | [Rep](./logs/reproduce/face_parsing.train_celebm_farl-b-50m-ep16_448_refinebb) |
| [face_parsing/<br/>train_lapa_farl-b-50m-ep16_448_refinebb.yaml](./farl/experiments/face_parsing/train_lapa_farl-b-50m-ep16_448_refinebb.yaml) | Face Parsing | LaPa | F1-mean ⇑ | 94.01 | [Rep](./logs/reproduce/face_parsing.train_lapa_farl-b-50m-ep16_448_refinebb) |
| [face_alignment/<br/>train_aflw19_farl-b-50m-ep16_448_refinebb.yaml](./farl/experiments/face_alignment/train_aflw19_farl-b-50m-ep16_448_refinebb.yaml) | Face Alignment | AFLW-19 (Full) | NME_diag ⇓ | 0.937 | [Rep](./logs/reproduce/face_alignment.train_aflw19_farl-b-50m-ep16_448_refinebb) |
| [face_alignment/<br/>train_ibug300w_farl-b-50m-ep16_448_refinebb.yaml](./farl/experiments/face_alignment/train_ibug300w_farl-b-50m-ep16_448_refinebb.yaml) | Face Alignment | 300W (Full) | NME_inter-ocular ⇓ | 2.92 | [Rep](./logs/reproduce/face_alignment.train_ibug300w_farl-b-50m-ep16_448_refinebb) |
| [face_alignment/<br/>train_wflw_farl-b-50m-ep16_448_refinebb.yaml](./farl/experiments/face_alignment/train_wflw_farl-b-50m-ep16_448_refinebb.yaml) | Face Alignment | WFLW (Full) | NME_inter-ocular ⇓ | 3.99 | [Rep](./logs/reproduce/face_alignment.train_wflw_farl-b-50m-ep16_448_refinebb) | -->

## Pre-trained Downstream Models

We will continuously update the pre-trained downstream face models in our [facer](https://github.com/FacePerceiver/facer) package.

## LAION-Face Dataset

We use the [LAION-Face](https://github.com/FacePerceiver/LAION-Face) dataset for training the FaRL model, [LAION-Face](https://github.com/FacePerceiver/LAION-Face) is the human face subset of LAION-400M, it consists of 50 million image-text pairs, we use the 20M subset for fast verification. 

## Contact

For help or issues concerning the code and the released models, feel free to submit a GitHub issue, or contact [Hao Yang](https://haya.pro) ([haya@microsoft.com](mailto:haya@microsoft.com)).


## Citation

If you find our work helpful, please consider citing 
```
@article{zheng2021farl,
  title={General Facial Representation Learning in a Visual-Linguistic Manner},
  author={Zheng, Yinglin and Yang, Hao and Zhang, Ting and Bao, Jianmin and Chen, Dongdong and Huang, Yangyu and Yuan, Lu and Chen, Dong and Zeng, Ming and Wen, Fang},
  journal={arXiv preprint arXiv:2112.03109},
  year={2021}
}
```


## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
