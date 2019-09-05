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
***This is just a preliminary version for early access. I will complete it as soon as I can.***

### WIDER Attribute Dataset
To run this code, you need to specify the `WIDER_DATA_DIR` (WIDER dataset "Image"), `WIDER_ANNO_DIR` (WIDER dataset annotations) in "configs.py"and the argument of `model_dir` (path to save checkpoints). Then, run the command (with PyTorch installed):
`python main.py`.

Note that you may need to use the specific PyTorch version: 0.3.1.

### PA-100K Dataset
(To be integrated)

### MS-COCO
(To be integrated)

*For those who want to test the proposed method on MS-COCO dataset, the source codes which I used for experiments are temporarily uploaded in the `./tmp/` folder. I will do code cleanup later.*

**Select the checkpoint that produces the best mAP**.

You can also evaluate the predictions with code at: https://github.com/zhufengx/SRN_multilabel, which would produce a slightly better (~0.1\%) performance.

## Datasets

1. WIDER Attribute Dataset: http://mmlab.ie.cuhk.edu.hk/projects/WIDERAttribute.html
2. PA-100K: (will be supported later)
3. MS-COCO: (will be supported later)

