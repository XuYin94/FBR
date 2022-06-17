# NEED TO SET
DATASET_ROOT=/root/Dataset/VOC/VOCdevkit/VOC2012/
WEIGHT_ROOT=./
SALIENCY_ROOT=/root/Dataset/VOC/VOCdevkit/VOC2012/salience

GPU=0

# Default setting
IMG_ROOT=${DATASET_ROOT}/JPEGImages
SAL_ROOT=${SALIENCY_ROOT}
BACKBONE=resnet38_contrast
SESSION=EPS_contrast
BASE_WEIGHT=${WEIGHT_ROOT}/ilsvrc-cls_rna-a1_cls1000_ep-0001.params


# train classification network with Contrastive Learning
#CUDA_VISIBLE_DEVICES=${GPU} python3 contrast_train.py \
  #--session ${SESSION} \
  #--network network.${BACKBONE} \
  #--data_root ${IMG_ROOT} \
  #--saliency_root ${SAL_ROOT} \
  #--weights ${BASE_WEIGHT} \
  #--crop_size 448 \
  #--tau 0.4 \
  #--max_iters 10000 \
  #--iter_size 2 \
  #--batch_size 8


# 2. inference CAM
DATA=train # train / train_aug
TRAINED_WEIGHT=train_log/${SESSION}/checkpoint_contrast1.pth

CUDA_VISIBLE_DEVICES=${GPU} python3 contrast_infer.py \
    --infer_list data/voc12/${DATA}_id.txt \
    --img_root ${IMG_ROOT} \
    --network network.${BACKBONE} \
    --weights ${TRAINED_WEIGHT} \
    --thr 0.22 \
    --n_gpus 1 \
    --n_processes_per_gpu 4 \
    --cam_png train_log/${SESSION}/result/cam_png \
    --cam_npy train_log/${SESSION}/result/cam_npy \
    --crf train_log/${SESSION}/result/crf_png\
    --crf_t 5 \
    --crf_alpha 8

# 3. evaluate CAM
GT_ROOT=${DATASET_ROOT}/SegmentationClassAug/
DATA=train  # 记得改train

CUDA_VISIBLE_DEVICES=${GPU} python3 eval.py \
    --list data/voc12/${DATA}_id.txt \
    --gt_dir '/root/Dataset/VOC/VOCdevkit/VOC2012/SegmentationClassAug' \
    --logfile ./train_log/${DATA}_eval.txt \
    --predict_dir ./train_log/${SESSION}/result/cam_npy