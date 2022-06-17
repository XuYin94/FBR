import PIL.Image
import random
import numpy as np
import torch.nn as nn
import torch

class RandomResizeLong:

    def __init__(self, min_long, max_long):
        self.min_long = min_long
        self.max_long = max_long

    def __call__(self, img, target_long=None, mode='image'):
        if target_long is None:
            target_long = random.randint(self.min_long, self.max_long)
        w, h = img.size

        if w < h:
            target_shape = (int(round(w * target_long / h)), target_long)
        else:
            target_shape = (target_long, int(round(h * target_long / w)))

        if mode == 'image':
            img = img.resize(target_shape, resample=PIL.Image.CUBIC)
        elif mode == 'mask':
            img = img.resize(target_shape, resample=PIL.Image.NEAREST)

        return img


class RandomCrop:

    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, imgarr):

        h, w, c = imgarr.shape

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        w_space = w - self.cropsize
        h_space = h - self.cropsize

        if w_space > 0:
            cont_left = 0
            img_left = random.randrange(w_space+1)
        else:
            cont_left = random.randrange(-w_space+1)
            img_left = 0

        if h_space > 0:
            cont_top = 0
            img_top = random.randrange(h_space+1)
        else:
            cont_top = random.randrange(-h_space+1)
            img_top = 0

        container = np.zeros((self.cropsize, self.cropsize, imgarr.shape[-1]), np.float32)
        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            imgarr[img_top:img_top+ch, img_left:img_left+cw]

        return container


def random_crop_with_saliency(imgarr, mask, crop_size):

    h, w, c = imgarr.shape

    ch = min(crop_size, h)
    cw = min(crop_size, w)

    w_space = w - crop_size
    h_space = h - crop_size

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space+1)
    else:
        cont_left = random.randrange(-w_space+1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space+1)
    else:
        cont_top = random.randrange(-h_space+1)
        img_top = 0

    container = np.zeros((crop_size, crop_size, imgarr.shape[-1]), np.float32)
    container_mask = np.zeros((crop_size, crop_size, imgarr.shape[-1]), np.float32)
    container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        imgarr[img_top:img_top+ch, img_left:img_left+cw]
    container_mask[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        mask[img_top:img_top+ch, img_left:img_left+cw]

    return container, container_mask


class RandomHorizontalFlip():
    def __init__(self):
        return

    def __call__(self, img):
        if bool(random.getrandbits(1)):
            img = np.fliplr(img).copy()
        return img


class CenterCrop():

    def __init__(self, cropsize, default_value=0):
        self.cropsize = cropsize
        self.default_value = default_value

    def __call__(self, npimg):

        h, w = npimg.shape[:2]

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        sh = h - self.cropsize
        sw = w - self.cropsize

        if sw > 0:
            cont_left = 0
            img_left = int(round(sw / 2))
        else:
            cont_left = int(round(-sw / 2))
            img_left = 0

        if sh > 0:
            cont_top = 0
            img_top = int(round(sh / 2))
        else:
            cont_top = int(round(-sh / 2))
            img_top = 0

        if len(npimg.shape) == 2:
            container = np.ones((self.cropsize, self.cropsize), npimg.dtype)*self.default_value
        else:
            container = np.ones((self.cropsize, self.cropsize, npimg.shape[2]), npimg.dtype)*self.default_value

        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            npimg[img_top:img_top+ch, img_left:img_left+cw]

        return container


def HWC_to_CHW(img):
    return np.transpose(img, (2, 0, 1))


def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img_arr = np.asarray(img)
        normalized_img = np.empty_like(img_arr, np.float32)

        normalized_img[..., 0] = (img_arr[..., 0] / 255. - self.mean[0]) / self.std[0]
        normalized_img[..., 1] = (img_arr[..., 1] / 255. - self.mean[1]) / self.std[1]
        normalized_img[..., 2] = (img_arr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return normalized_img
        

class ThresholdedAvgPool2d(nn.Module):
    def __init__(self, threshold=0.0):
        super(ThresholdedAvgPool2d, self).__init__()
        self.threshold = threshold

    def forward(self, feature_map, layer, truncate=False, shift=0.0, bias=False):
        # threshold feature map
        batch_size, channel, height, width = feature_map.shape
        max_vals, _ = torch.max(feature_map.view(batch_size, channel, -1), dim=2)
        thr_vals = (max_vals * self.threshold).view(batch_size, channel, 1, 1).expand_as(feature_map)
        thr_feature_map = torch.where(
            torch.gt(feature_map, thr_vals), feature_map, torch.zeros_like(feature_map))

        # divided by the number of positives
        num_positives = torch.sum(torch.gt(thr_feature_map, 0.), dim=(2,3))
        num_positives = torch.where(torch.eq(num_positives, 0),
                                    torch.ones_like(num_positives),
                                    num_positives).view(batch_size, channel, 1, 1).expand_as(feature_map)
        avg_feature_map = torch.div(thr_feature_map, num_positives.float())

        # convolve
        #weight = layer.weight + compute_shift(shift, layer.weight)
        weight = layer.weight

        if truncate:
            weight = torch.where(torch.gt(layer.weight, 0.),
                                 layer.weight, torch.zeros_like(layer.weight))

        if len(weight.shape) < 4:
            weight = weight.unsqueeze(-1).unsqueeze(-1)

        avgpooled_map = nn.functional.conv2d(
            avg_feature_map, weight=weight, bias=None)
        pred = torch.sum(avgpooled_map, dim=(2,3))
        score_map = nn.functional.conv2d(
            feature_map, weight=weight, bias=None)
        if bias:
            pred = pred + layer.bias.view(1, -1)

        return pred,score_map

class CustomAvgPool2d(nn.Module):
    def __init__(self):
        super(CustomAvgPool2d, self).__init__()

    def forward(self, feature_map, layer, truncate=False, shift=0.0, bias=True):
        _, _, height, width = feature_map.shape

        avg_feature_map = feature_map / (height * width)

        weight = layer.weight #+ compute_shift(shift, layer.weight)

        if truncate:
            weight = torch.where(torch.gt(layer.weight, 0.),
                                 layer.weight, torch.zeros_like(layer.weight))

        if len(weight.shape) < 4:
            weight = weight.unsqueeze(-1).unsqueeze(-1)

        score_map = nn.functional.conv2d(
            avg_feature_map, weight=weight, bias=None)
        pred = torch.sum(score_map, dim=(2, 3))

        if bias and layer.bias is not None:
            pred = pred + layer.bias.view(1, -1)

        return pred,score_map



def compute_shift(shift, weight):
    shift_type = list(shift.keys())[0]
    if shift_type == 'global_min':
        shift = torch.abs(torch.min(weight)) + float(shift[shift_type])
    elif shift_type == 'adaptive_min':
        shift = torch.abs(torch.min(weight, dim=1)[0].unsqueeze(-1)) + float(shift[shift_type])
    elif shift_type == 'global_multiple':
        gmin = torch.min(weight)
        gmax = torch.max(weight)
        k = shift[shift_type]
        shift = (gmax - k * gmin) / float(k - 1)
    elif shift_type == 'adaptive_multiple':
        amin = torch.abs(torch.min(weight, dim=1)[0].unsqueeze(-1))
        amax = torch.abs(torch.min(weight, dim=1)[0].unsqueeze(-1))
        k = shift[shift_type]
        shift = (amax - k * amin) / float(k - 1)

    return shift

