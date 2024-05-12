import torch
import torch.nn as nn
from torch.nn import functional as F

def get_syncbn():
    # return nn.BatchNorm2d
    return nn.SyncBatchNorm


class dec_deeplabv3_plus(nn.Module):
    def __init__(
        self,
        in_planes,
        num_classes=9,
        inner_planes=256,
        sync_bn=False,
        dilations=(12, 24, 36),
        low_conv_planes=48,
    ):
        super(dec_deeplabv3_plus, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d

        self.low_conv = nn.Sequential(
            nn.Conv2d(256, low_conv_planes, kernel_size=1), 
            norm_layer(low_conv_planes), 
            nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(
            in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations
        )

        self.head = nn.Sequential(
            nn.Conv2d(self.aspp.get_outplanes(), 256, 1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(256+int(low_conv_planes), 256, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def mask_pooling(self, mask, x):
        #mask B,H,W
        mask = F.interpolate(mask, x.shape[-2:], mode="nearest", align_corners=True)
        s = mask.sum(dim=[1,2])
        mask_empty = (s==0)
        s[mask_empty] = 10 # any non-zero number
        pool_x = (x*mask.unsqueeze(1)).sum(dim=[2,3])/s.unsqueeze(1) #B,C
        return pool_x

    def forward(self, x, mask=None, return_feat=False):
        x1, x2, x3, x4 = x
        low_feat = self.low_conv(x1)
        h, w = low_feat.size()[-2:]

        aspp_out = self.aspp(x4)
        aspp_out = self.head(aspp_out)
        aspp_out = F.interpolate(
            aspp_out, size=(h, w), mode="bilinear", align_corners=True
        )

        aspp_out = torch.cat((low_feat, aspp_out), dim=1)
        if mask==None:
            if return_feat:
                return self.classifier(aspp_out), aspp_out
            return self.classifier(aspp_out)
        else:
            #mask B,H,W,3
            pool_list = [ ]
            for i in range(mask.shape[-1]):
                m = mask[:,:,:,i]
                pool_x = self.mask_pooling(m, aspp_out)
                pool_list.append(pool_x)
            return self.classifier(aspp_out), pool_list

class decoder_prototype(nn.Module):
    def __init__(
        self,
        in_planes,
        num_classes=9,
        inner_planes=256,
        sync_bn=False,
        dilations=(12, 24, 36),
        low_conv_planes=48,
    ):
        super(decoder_prototype, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d

        self.low_conv = nn.Sequential(
            nn.Conv2d(256, low_conv_planes, kernel_size=1), 
            norm_layer(low_conv_planes), 
            nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(
            in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations
        )

        self.head = nn.Sequential(
            nn.Conv2d(self.aspp.get_outplanes(), 256, 1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
        )
        
        self.pre = nn.Sequential(
            nn.Conv2d(256+int(low_conv_planes), 256, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.projector = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=1, bias=False)

    def mask_pooling(self, mask, x):
        #mask B,H,W
        mask = F.interpolate(mask, x.shape[-2:], mode="nearest", align_corners=True)
        s = mask.sum(dim=[1,2])
        mask_empty = (s==0)
        s[mask_empty] = 10 # any non-zero number
        pool_x = (x*mask.unsqueeze(1)).sum(dim=[2,3])/s.unsqueeze(1) #B,C
        return pool_x

    def forward(self, x, return_feat=False,project=False):
        x1, x2, x3, x4 = x
        low_feat = self.low_conv(x1)
        h, w = low_feat.size()[-2:]

        aspp_out = self.aspp(x4)
        aspp_out = self.head(aspp_out)
        aspp_out = F.interpolate(
            aspp_out, size=(h, w), mode="bilinear", align_corners=True
        )

        aspp_out = torch.cat((low_feat, aspp_out), dim=1)

        aspp_out = self.pre(aspp_out)
        if return_feat:
            return aspp_out
        if project:
            return self.classifier(aspp_out), self.projector(aspp_out)
        else:
            return self.classifier(aspp_out)
        

class dec_deeplabv3_plus_u2pl(nn.Module):
    def __init__(
        self,
        in_planes,
        num_classes=19,
        inner_planes=256,
        sync_bn=False,
        dilations=(12, 24, 36),
        rep_head=True,
        low_conv_planes=256,
    ):
        super(dec_deeplabv3_plus_u2pl, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.rep_head = rep_head

        self.low_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1), norm_layer(256), nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(
            in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations
        )

        self.head = nn.Sequential(
            nn.Conv2d(
                self.aspp.get_outplanes(),
                256,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

        if self.rep_head:

            self.representation = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            )

    def forward(self, x):
        x1, x2, x3, x4 = x
        aspp_out = self.aspp(x4)
        low_feat = self.low_conv(x1)
        aspp_out = self.head(aspp_out)
        h, w = low_feat.size()[-2:]
        aspp_out = F.interpolate(
            aspp_out, size=(h, w), mode="bilinear", align_corners=True
        )
        aspp_out = torch.cat((low_feat, aspp_out), dim=1)

        res = {"pred": self.classifier(aspp_out)}

        if self.rep_head:
            res["rep"] = self.representation(aspp_out)

        return res


class Aux_Module(nn.Module):
    def __init__(self, in_planes, num_classes=19, sync_bn=False):
        super(Aux_Module, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.aux = nn.Sequential(
            nn.Conv2d(in_planes, 256, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        res = self.aux(x)
        return res


class ASPP(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(
        self, in_planes, inner_planes=256, sync_bn=False, dilations=(12, 24, 36)
    ):
        super(ASPP, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[0],
                dilation=dilations[0],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[1],
                dilation=dilations[1],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[2],
                dilation=dilations[2],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )

        self.out_planes = (len(dilations) + 2) * inner_planes

    def get_outplanes(self):
        return self.out_planes

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(
            self.conv1(x), size=(h, w), mode="bilinear", align_corners=True
        )
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        aspp_out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        return aspp_out
