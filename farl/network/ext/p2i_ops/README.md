# P2I

`p2i` is a simple yet effective point rendering operator for PyTorch. It is fully differentiable. It supports gradients to be back-propagated to not only point colors/features, but also point coordinates. The current implementation of `p2i` requires CUDA.

## Citation

If you find this operator useful, please consider citing

```
@article{zheng2021farl,
  title={General Facial Representation Learning in a Visual-Linguistic Manner},
  author={Zheng, Yinglin and Yang, Hao and Zhang, Ting and Bao, Jianmin and Chen, Dongdong and Huang, Yangyu and Yuan, Lu and Chen, Dong and Zeng, Ming and Wen, Fang},
  journal={arXiv preprint arXiv:2112.03109},
  year={2021}
}

@inproceedings{xie2021style,
  title={Style-based Point Generator with Adversarial Rendering for Point Cloud Completion},
  author={Xie, Chulin and Wang, Chuxin and Zhang, Bo and Yang, Hao and Chen, Dong and Wen, Fang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4619--4628},
  year={2021}
}
```

## Contact

Please raise issues or contact [Hao Yang](https://haya.pro) (`haya@microsoft.com`) for any questions about this implementation.