# FBR
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fine-grained-background-representation-for/weakly-supervised-semantic-segmentation-on-14)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on-14?p=fine-grained-background-representation-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fine-grained-background-representation-for/weakly-supervised-semantic-segmentation-on-1)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on-1?p=fine-grained-background-representation-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fine-grained-background-representation-for/weakly-supervised-semantic-segmentation-on-4)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on-4?p=fine-grained-background-representation-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fine-grained-background-representation-for/weakly-supervised-semantic-segmentation-on)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on?p=fine-grained-background-representation-for)


This is the official implementation of [Fine-grained Background Representation for Weakly Supervised Semantic Segmentation](https://sgvr.kaist.ac.kr/~xuyin/FBR/) that was published in IEEE TCSVT 2024.
# Overall Framework
![overall framework](https://github.com/YininKorea/FBR/blob/4802ab1ee66f683d98deef9bb635c39b1988e621/architecture.PNG)
>
###

Requirements:
* [Pascal Voc](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
* [Salience map](https://drive.google.com/file/d/19AjSmgdMlIZH4FXVZ5zjlUZcoZZCkwrI/view)
* [Pretrained model for ResNet38 (on ImageNet)](https://drive.google.com/file/d/1LzMIhXGjdiml0KbgTUH2xglSQelqpx0H/view?usp=drive_link)
###

Steps of experiments:
* You can directly run the code by executing the script script_contrast.sh on the script folder.
*  The steps consist of 
   *  Train the seeds
   * Post-processing with CRF
   * Evaluation
   

#Results and trained models 

on Pascal Voc 2012

|Method|train set|val set|test set|
| ---- | ----    |  ---- |  ----  |
|AMN|72.2|70.7|70.6|
|+ours|73.1|71.8|[73.2](http://host.robots.ox.ac.uk:8080/anonymous/30LARO.html)|
|PPC|73.6|72.6|73.6|
|+ours|75.9|74.2|[74.9](http://host.robots.ox.ac.uk:8080/anonymous/BHSCOK.html)|

on MS COCO 2014

|Method|train set|val set|
| ---- | ----    |  ---- |
|AMN|-|44.7|
|+ours|46.7|45.6|


[Checkpoints](https://drive.google.com/file/d/1CKkKk72nowWnsVYFPxt9kNk50k3eWVu2/view?usp=sharing)

[Pseudo labels](https://drive.google.com/file/d/1ZFowMQkvFBWQyPnlQC3WtPZr6ykFFLqb/view?usp=sharing)


License:

This project is released under the MIT License.
