# Reco_WSSS
This is Yin Xu's work for WSSS


Before looking through the work, I recommend you to first read the following literature:
###

1. [Weakly Supervised Semantic Segmentation by Pixel-to-Prototype Contrast](https://arxiv.org/abs/2110.07110)

2. [Railroad is not a Train: Saliency as Pseudo-pxiel Supervision for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Railroad_Is_Not_a_Train_Saliency_As_Pseudo-Pixel_Supervision_for_CVPR_2021_paper.pdf)

3. [Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations](https://arxiv.org/abs/1904.05044)

###

Requirements:
* [Pascal Voc](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
* [Salience map](https://drive.google.com/file/d/19AjSmgdMlIZH4FXVZ5zjlUZcoZZCkwrI/view)
* [Pretrained model for ResNet38 (on ImageNet)] (https://drive.google.com/file/d/15F13LEL5aO45JU-j45PYjzv5KW5bn_Pn/view)
###

Steps of experiments:
* You can directly run the code by excuting the script script_contrast.sh on the script folder.
*  The steps consist of 
   *  Train the seeds
   * Post processing with CRF
   * Evaluation
###

Our main contribution is made on the region_utils.py that implements a memory bank to perform the foreground-background contrastive learning.

You can download the checkpoint with: 
1. [wseg](https://drive.google.com/file/d/1fSWXSmMZA09fh_NG-dg3slMi9MRBr6L9/view?usp=sharing)
2. [AMN_pascal](https://drive.google.com/file/d/1762JhjDd-ckcCxnmD3OepCt2xylFoZ3G/view?usp=sharing) w/ [IRN](https://drive.google.com/file/d/1VNpipFLRbVzi6qNLy1E0Pjd8oh_zsIjE/view?usp=sharing)
2. [AMN_coco](https://drive.google.com/file/d/1hFW_L-HXUcjy3ecSc0rxfNNlonifHGf0/view?usp=sharing) 
