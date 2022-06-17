import torch
import numpy as np
import os
import PIL
value_scale = 255
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def Check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
                    palette.append(0)
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask



class AverageMeter(object):
    "help computing and storing the metrics"
    def __init__(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0

    def update(self,value,count=1):
        self.val=value
        self.sum+=value*count
        self.count+=count
        self.avg=self.sum/self.count

    @property
    def value(self):
        return self.val

    def average(self):
        return np.round(self.avg,5)

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target




def get_voc_palette(num_classes):
    n = num_classes
    palette = [0]*(n*3)
    for j in range(0,n):
            lab = j
            palette[j*3+0] = 0
            palette[j*3+1] = 0
            palette[j*3+2] = 0
            i = 0
            while (lab > 0):
                    palette[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                    palette[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                    palette[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                    i = i + 1
                    lab >>= 3
    return palette

COCO_palette = [31, 119, 180, 255, 127, 14, 44, 160, 44, 214, 39, 40, 148, 103, 189, 140, 86, 75, 227,
    119, 194, 127, 127, 127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127, 14, 44, 160, 44,
    214, 39, 40, 148, 103, 189, 140, 86, 75, 227, 119, 194, 127, 127, 127, 188, 189, 34, 23, 190, 207,
    31, 119, 180, 255, 127, 14, 44, 160, 44, 214, 39, 40, 148, 103, 189, 140, 86, 75,
    227, 119, 194, 127, 127, 127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127, 14, 44, 160, 44,
    214, 39, 40, 148, 103, 189, 140, 86, 75, 227, 119, 194, 127, 127, 127, 188, 189,
    34, 23, 190, 207, 31, 119, 180, 255, 127, 14, 44, 160, 44, 214, 39, 40, 148, 103, 189, 140, 86, 75,
    227, 119, 194, 127, 127, 127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127,
    14, 44, 160, 44, 214, 39, 40, 148, 103, 189, 140, 86, 75, 227, 119, 194, 127, 127, 127, 188, 189,
    34, 23, 190, 207, 31, 119, 180, 255, 127, 14, 44, 160, 44, 214, 39, 40, 148, 103,
    189, 140, 86, 75, 227, 119, 194, 127, 127, 127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127,
    14, 44, 160, 44, 214, 39, 40, 148, 103, 189, 140, 86, 75, 227, 119, 194, 127, 127
    , 127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127, 14, 44, 160, 44, 214, 39, 40, 148, 103,
    189, 140, 86, 75, 227, 119, 194, 127, 127, 127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127, 14,
    44, 160, 44, 214, 39, 40, 148, 103, 189, 140, 86, 75, 227, 119, 194, 127, 127,
    127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127, 14, 44, 160, 44, 214, 39, 40, 148, 103, 189,
    140, 86, 75, 227, 119, 194, 127, 127, 127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127, 14, 44,
    160, 44, 214, 39, 40, 148, 103, 189, 140, 86, 75, 227, 119, 194, 127, 127, 127, 188, 189, 34, 23, 190,
    207, 31, 119, 180, 255, 127, 14, 44, 160, 44, 214, 39, 40, 148, 103, 189, 140, 86, 75, 227, 119, 194,
    127, 127, 127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127, 14, 44, 160, 44, 214, 39, 40, 148,
    103, 189, 140, 86, 75, 227, 119, 194, 127, 127, 127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127,
    14, 44, 160, 44, 214, 39, 40, 148, 103, 189, 140, 86, 75, 227, 119, 194, 127, 127, 127, 188, 189, 34,
    23, 190, 207, 31, 119, 180, 255, 127, 14, 44, 160, 44, 214, 39, 40, 148, 103, 189, 140, 86, 75, 227,
    119, 194, 127, 127, 127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127, 14, 44, 160, 44, 214, 39,
    40, 148, 103, 189, 140, 86, 75, 227, 119, 194, 127, 127, 127, 188, 189, 34, 23, 190, 207, 31, 119,
    180, 255, 127, 14, 44, 160, 44, 214, 39, 40, 148, 103, 189, 140, 86, 75, 227, 119, 194, 127, 127,
    127, 188, 189, 34, 23, 190, 207, 31, 119, 180, 255, 127, 14]