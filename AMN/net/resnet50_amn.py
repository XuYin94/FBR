import torch.nn as nn
import torch.nn.functional as F
from net import resnet50
import torch


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)
        self.proj_fg = nn.Conv2d(2048, 128, 1, bias=False)
        # self.proj_bg = nn.Conv2d(2048, 128, 1, bias=False)
        # self.bg_bn = nn.BatchNorm2d(128)
        # self.bg_classifier = nn.Conv2d(128, 1, 1, bias=False)
        # self.dropout7 = nn.Dropout(0.5)

        astrous_rates = [6, 12, 18, 24]

        self.label_enc = nn.Linear(20, 2048)

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            _ASPP(in_ch=2048, out_ch=21, rates=astrous_rates)
        )
        # torch.nn.init.xavier_uniform_(self.bg_classifier.weight)
        torch.nn.init.xavier_uniform_(self.proj_fg.weight)
        # torch.nn.init.xavier_uniform_(self.proj_bg.weight)
        # torch.nn.init.xavier_uniform_(self.bg_classifier.weight)
        # self.bg_bn.weight.data.fill_(1.)
        # self.bg_bn.bias.data.fill_(1e-4)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.proj_fg, self.classifier, self.label_enc])

    def forward(self, img, label_cls):

        y = self.label_enc(label_cls).unsqueeze(-1).unsqueeze(-1)

        x = self.stage1(img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x * y

        fg_feats = F.relu(self.proj_fg(x.clone()), inplace=True)
        logit = self.classifier(x)

        return logit, fg_feats

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))

    def cosine_loss(self, fore_feat, bg_feat):
        cosine_loss = torch.nn.CosineSimilarity(1, 1e-8)(fore_feat, bg_feat).mean()
        return cosine_loss


class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x, y):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        B, C, H, W = x.shape

        y = self.label_enc(y)
        y = y.unsqueeze(-1).unsqueeze(-1)

        x = x * y

        logit = self.classifier(x)

        logit = (logit[0] + logit[1].flip(-1)) / 2

        return logit
