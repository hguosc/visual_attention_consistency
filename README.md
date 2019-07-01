# Visual Attention Consistency
This repository is for the following paper:
```
@InProceedings{Guo_2019_CVPR,
author = {Guo, Hao and Zheng, Kang and Fan, Xiaochuan and Yu, Hongkai and Wang, Song},
title = {Visual Attention Consistency Under Image Transforms for Multi-Label Image Classification},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```


## Note:
*This is just a preliminary version for early access. I will complete it as soon as I can.*

To run this code, you need to specify the `WIDER_DATA_DIR` (WIDER dataset "Image"), `WIDER_ANNO_DIR` (WIDER dataset annotations) in "configs.py"and the argument of `model_dir` (path to save checkpoints). Then, run the command (with PyTorch installed):
`python main.py`.

Select the checkpoint with the best mAP.

## Datasets

1. WIDER Attribute Dataset: http://mmlab.ie.cuhk.edu.hk/projects/WIDERAttribute.html
2. PA-100K: (will be supported later)
3. MS-COCO: (will be supported later)

