# *FaRL* for *Fa*cial *R*epresentation *L*earning

	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/general-facial-representation-learning-in-a/face-alignment-on-300w)](https://paperswithcode.com/sota/face-alignment-on-300w?p=general-facial-representation-learning-in-a)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/general-facial-representation-learning-in-a/face-alignment-on-aflw-19)](https://paperswithcode.com/sota/face-alignment-on-aflw-19?p=general-facial-representation-learning-in-a)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/general-facial-representation-learning-in-a/face-alignment-on-wflw)](https://paperswithcode.com/sota/face-alignment-on-wflw?p=general-facial-representation-learning-in-a)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/general-facial-representation-learning-in-a/face-parsing-on-celebamask-hq)](https://paperswithcode.com/sota/face-parsing-on-celebamask-hq?p=general-facial-representation-learning-in-a)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/general-facial-representation-learning-in-a/face-parsing-on-lapa)](https://paperswithcode.com/sota/face-parsing-on-lapa?p=general-facial-representation-learning-in-a)

This repo hosts official implementation of our paper "[**General Facial Representation Learning in a Visual-Linguistic Manner**](https://arxiv.org/abs/2112.03109)".


## Introduction

**FaRL** offers powerful pre-training transformer backbones for face analysis tasks. Its pre-training combines both the image-text contrastive learning and the masked image modeling.

<img src="./figures/framework.jpg" alt="framework" width="400"/>

After the pre-training, the image encoder can be utilized for various downstream face tasks. 


## Setup Downstream Training

Different pre-trained transformer backbones can be downloaded as below.

| Model Name  |  Pre-training Data | Link |
| ----------- | -------------- | ----- |
| FaRL-Base-Patch16-LAIONFace20M-ep16 (used in paper) | LAION Face 20M  | [OneDrive](https://1drv.ms/u/s!AperexS2nqQomyPsG2M4uPXay7Au?e=Ocvk1T), [BLOB](https://facevcstandard.blob.core.windows.net/haya/releases/farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth?sv=2020-08-04&st=2021-12-17T13%3A00%3A07Z&se=2025-01-18T13%3A00%3A00Z&sr=b&sp=r&sig=D0ZPJgp8BrAgHIdACfZzqPnyOcX1ivGdHnF8qgtWdoI%3D) |
| FaRL-Base-Patch16-LAIONFace20M-ep64 | LAION Face 20M  | [BLOB](https://facevcstandard.blob.core.windows.net/haya/releases/farl/FaRL-Base-Patch16-LAIONFace20M-ep64.pth?sv=2020-08-04&st=2021-12-27T05%3A22%3A56Z&se=2025-12-21T05%3A22%3A00Z&sr=b&sp=r&sig=til1J9u%2FQqf6qRc6cPx9nPyOGl%2F9ahTyvQ3VBPePs6A%3D) |
| FaRL-Base-Patch16-LAIONFace50M-ep16 | LAION Face 50M | [OneDrive](https://1drv.ms/u/s!AperexS2nqQomyZp2z2DdUNoqTVp?e=T7C1QA), [BLOB](https://facevcstandard.blob.core.windows.net/haya/releases/farl/FaRL-Base-Patch16-LAIONFace50M-ep16.pth?sv=2020-08-04&st=2021-12-17T13%3A01%3A48Z&se=2025-01-17T13%3A01%3A00Z&sr=b&sp=r&sig=6g1B3f4vEmFc1tmz8QWSH6lRoK%2BABA%2FWfmqXLGS61MM%3D) |

Download these models to `./blob/checkpoint/`.

All downstream trainings require 8 NVIDIA V100 GPUs (32G).
Before setting up, install these packages:

* [PyTorch](https://pytorch.org/get-started/previous-versions/) 1.7.0
* [MMCV-Full](https://github.com/open-mmlab/mmcv) 1.3.14

Then, install the rest dependencies with `pip install -r ./requirement.txt`.

Please refer to [./DS_DATA.md](./DS_DATA.md) to prepare the training and testing data for downstream tasks.

Now you can launch the trainings with following command template.

```
python -m blueprint.run farl/experiments/{task}/{train_config_file}.yaml --exp_name farl --blob_root ./blob
```

The repo has included some config files under `./farl/experiments/` that perform finetuning for face parsing and face alignment.

## Performance

The following table illustrates their performances reported in the paper (Paper) or reproduced using this repo (Rep). There are small differences between their performances due to code refactorization.

| File Name | Task | Benchmark | Metric | Score (Paper/Rep) | Logs (Paper/Rep) |
| ---- | ---- | ---- | --- | --- | --- |
| [face_parsing/<br/>train_celebm_farl-b-ep16-448_refinebb.yaml](./farl/experiments/face_parsing/train_celebm_farl-b-ep16_448_refinebb.yaml) | Face Parsing  | CelebAMask-HQ | F1-mean ⇑ | 89.56/89.65 | [Paper](./logs/paper/face_parsing.train_celebm_farl-b-ep16-448_refinebb), [Rep](./logs/reproduce/face_parsing.train_celebm_farl-b-ep16_448_refinebb) |
| [face_parsing/<br/>train_lapa_farl-b-ep16_448_refinebb.yaml](./farl/experiments/face_parsing/train_lapa_farl-b-ep16_448_refinebb.yaml) | Face Parsing | LaPa | F1-mean ⇑ | 93.88/93.86 | [Paper](./logs/paper/face_parsing.train_lapa_farl-b-ep16_448_refinebb), [Rep](./logs/reproduce/face_parsing.train_lapa_farl-b-ep16_448_refinebb) |
| [face_alignment/<br/>train_aflw19_farl-b-ep16_448_refinebb.yaml](./farl/experiments/face_alignment/train_aflw19_farl-b-ep16_448_refinebb.yaml) | Face Alignment | AFLW-19 (Full) | NME_diag ⇓ | 0.943/0.943 | [Paper](./logs/paper/face_alignment.train_aflw19_farl-b-ep16_448_refinebb), [Rep](./logs/reproduce/face_alignment.train_aflw19_farl-b-ep16_448_refinebb) |
| [face_alignment/<br/>train_ibug300w_farl-b-ep16_448_refinebb.yaml](./farl/experiments/face_alignment/train_ibug300w_farl-b-ep16_448_refinebb.yaml) | Face Alignment | 300W (Full) | NME_inter-ocular ⇓ | 2.93/2.92 | [Paper](./logs/paper/face_alignment.train_ibug300w_farl-b-ep16_448_refinebb), [Rep](./logs/reproduce/face_alignment.train_ibug300w_farl-b-ep16_448_refinebb) |
| [face_alignment/<br/>train_wflw_farl-b-ep16_448_refinebb.yaml](./farl/experiments/face_alignment/train_wflw_farl-b-ep16_448_refinebb.yaml) | Face Alignment | WFLW (Full) | NME_inter-ocular ⇓ | 3.96/3.98 | [Paper](./logs/paper/face_alignment.train_wflw_farl-b-ep16_448_refinebb), [Rep](./logs/reproduce/face_alignment.train_wflw_farl-b-ep16_448_refinebb) |

We also report results using the 50M pre-trained backbone, showing further enhancement on LaPa and AFLW-19.

| File Name | Task | Benchmark | Metric | Score | Logs |
| ---- | ---- | ---- | --- | --- | --- |
| [face_parsing/<br/>train_celebm_farl-b-50m-ep16-448_refinebb.yaml](./farl/experiments/face_parsing/train_celebm_farl-b-50m-ep16_448_refinebb.yaml) | Face Parsing  | CelebAMask-HQ | F1-mean ⇑ | 89.68 | [Rep](./logs/reproduce/face_parsing.train_celebm_farl-b-50m-ep16_448_refinebb) |
| [face_parsing/<br/>train_lapa_farl-b-50m-ep16_448_refinebb.yaml](./farl/experiments/face_parsing/train_lapa_farl-b-50m-ep16_448_refinebb.yaml) | Face Parsing | LaPa | F1-mean ⇑ | 94.01 | [Rep](./logs/reproduce/face_parsing.train_lapa_farl-b-50m-ep16_448_refinebb) |
| [face_alignment/<br/>train_aflw19_farl-b-50m-ep16_448_refinebb.yaml](./farl/experiments/face_alignment/train_aflw19_farl-b-50m-ep16_448_refinebb.yaml) | Face Alignment | AFLW-19 (Full) | NME_diag ⇓ | 0.937 | [Rep](./logs/reproduce/face_alignment.train_aflw19_farl-b-50m-ep16_448_refinebb) |
| [face_alignment/<br/>train_ibug300w_farl-b-50m-ep16_448_refinebb.yaml](./farl/experiments/face_alignment/train_ibug300w_farl-b-50m-ep16_448_refinebb.yaml) | Face Alignment | 300W (Full) | NME_inter-ocular ⇓ | 2.92 | [Rep](./logs/reproduce/face_alignment.train_ibug300w_farl-b-50m-ep16_448_refinebb) |
| [face_alignment/<br/>train_wflw_farl-b-50m-ep16_448_refinebb.yaml](./farl/experiments/face_alignment/train_wflw_farl-b-50m-ep16_448_refinebb.yaml) | Face Alignment | WFLW (Full) | NME_inter-ocular ⇓ | 3.99 | [Rep](./logs/reproduce/face_alignment.train_wflw_farl-b-50m-ep16_448_refinebb) |


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

## Contact

For help or issues concerning the code and the released models, please submit a GitHub issue.
Otherwise, please contact [Hao Yang](https://haya.pro) (`haya@microsoft.com`).


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
