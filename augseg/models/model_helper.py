import importlib
import torch.nn as nn
from torch.nn import functional as F
from .decoder import Aux_Module
import copy
import torch

class ModelBuilder(nn.Module):
    def __init__(self, net_cfg, classifier=False):
        super(ModelBuilder, self).__init__()
        self._sync_bn = net_cfg["sync_bn"]
        self._num_classes = net_cfg["num_classes"]

        self.encoder = self._build_encoder(net_cfg["encoder"])
        self.decoder = self._build_decoder(net_cfg["decoder"])
        #self.register_parameter('mask_param', nn.Parameter(torch.randn((3,))) )
        if classifier:
            self.classifier = nn.Sequential(
                    nn.Linear(256+48, 256),
                    nn.SyncBatchNorm(256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, self._num_classes),
                )

        self._use_auxloss = True if net_cfg.get("aux_loss", False) else False
        if self._use_auxloss:
            cfg_aux = net_cfg["aux_loss"]
            self.loss_weight = cfg_aux["loss_weight"]
            self.auxor = Aux_Module(
                cfg_aux["aux_plane"], self._num_classes, self._sync_bn
            )

    def _build_encoder(self, enc_cfg):
        enc_cfg["kwargs"].update({"sync_bn": self._sync_bn})
        pretrained_model_url = enc_cfg["pretrain"]
        encoder = self._build_module(enc_cfg["type"], enc_cfg["kwargs"], pretrain_model_url=pretrained_model_url)
        return encoder

    def _build_decoder(self, dec_cfg):
        dec_cfg["kwargs"].update(
            {
                "in_planes": self.encoder.get_outplanes(),
                "sync_bn": self._sync_bn,
                "num_classes": self._num_classes,
            }
        )
        decoder = self._build_module(dec_cfg["type"], dec_cfg["kwargs"])
        return decoder

    def _build_module(self, mtype, kwargs, pretrain_model_url=None):
        module_name, class_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        if pretrain_model_url is None:
            return cls(**kwargs)
        else:
            return cls(pretrain_model_url=pretrain_model_url, **kwargs)

    def forward(self, x, flag_use_fdrop=False, mask=None, return_feat=False, patch_mask=None):
        h, w = x.shape[-2:]
        if patch_mask!=None:
            mask_region = (patch_mask==0).detach()
            mask_len = x.shape[0]//2
            x[mask_len:,0,:,:][mask_region] = self.mask_param[0]
            x[mask_len:,1,:,:][mask_region] = self.mask_param[1]
            x[mask_len:,2,:,:][mask_region] = self.mask_param[2]
        
        if self._use_auxloss:
            f1, f2, feat1, feat2 = self.encoder(x)
            outs = self.decoder([f1, f2, feat1, feat2])
            pred_aux = self.auxor(feat1)

            # upsampling
            outs = F.interpolate(outs, (h, w), mode="bilinear", align_corners=True)
            pred_aux = F.interpolate(pred_aux, (h, w), mode="bilinear", align_corners=True)
            
            return outs, pred_aux
        else:
            pool_list = None
            if flag_use_fdrop:
                aug_len = x.shape[0]//2
                f1, f2, feat1, feat2 = self.encoder(x)
                f1_, feat2_ = f1[0:aug_len].clone(), feat2[0:aug_len].clone()
                f1_2 = nn.Dropout2d(0.5)(f1[aug_len:])
                feat2_2 = nn.Dropout2d(0.5)(feat2[aug_len:])
                f1 = torch.cat((f1_, f1_2))
                feat2 = torch.cat((feat2_, feat2_2))
                outs = self.decoder([f1, f2, feat1, feat2])
            else:
                feat = self.encoder(x)
                if mask==None:
                    if return_feat:
                        outs, pool_list = self.decoder(feat, mask=mask, return_feat=return_feat)
                    else:
                        outs = self.decoder(feat)
                    
                else:
                    outs, pool_list = self.decoder(feat, mask=mask, return_feat=return_feat)

            outs = F.interpolate(outs, (h, w), mode="bilinear", align_corners=True)
            
            return outs, pool_list

class ModelBuilder_u2pl(nn.Module):
    def __init__(self, net_cfg):
        super(ModelBuilder_u2pl, self).__init__()
        self._sync_bn = net_cfg["sync_bn"]
        self._num_classes = net_cfg["num_classes"]

        self.encoder = self._build_encoder(net_cfg["encoder"])
        self.decoder = self._build_decoder(net_cfg["decoder"])

        self._use_auxloss = True if net_cfg.get("aux_loss", False) else False
        self.fpn = True
        if self._use_auxloss:
            cfg_aux = net_cfg["aux_loss"]
            self.loss_weight = cfg_aux["loss_weight"]
            self.auxor = Aux_Module(
                cfg_aux["aux_plane"], self._num_classes, self._sync_bn
            )

    def _build_encoder(self, enc_cfg):
        enc_cfg["kwargs"].update({"sync_bn": self._sync_bn})
        encoder = self._build_module(enc_cfg["type"], enc_cfg["kwargs"])
        return encoder

    def _build_decoder(self, dec_cfg):
        dec_cfg["kwargs"].update(
            {
                "in_planes": self.encoder.get_outplanes(),
                "sync_bn": self._sync_bn,
                "num_classes": self._num_classes,
            }
        )
        decoder = self._build_module(dec_cfg["type"], dec_cfg["kwargs"])
        return decoder

    def _build_module(self, mtype, kwargs):
        module_name, class_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**kwargs)

    def forward(self, x):
        h, w = x.shape[-2:]
        if self._use_auxloss:
            if self.fpn:
                # feat1 used as dsn loss as default, f1 is layer2's output as default
                f1, f2, feat1, feat2 = self.encoder(x)
                outs = self.decoder([f1, f2, feat1, feat2])
            else:
                feat1, feat2 = self.encoder(x)
                outs = self.decoder(feat2)

            pred_aux = self.auxor(feat1)

            outs.update({"aux": pred_aux})
            return outs["pred"], None
        else:
            feat = self.encoder(x)
            outs = self.decoder(feat)
            outs = F.interpolate(outs['pred'], (h, w), mode="bilinear", align_corners=True)
            return outs, None

class ModelBuilder_Proto(nn.Module):
    def __init__(self, net_cfg):
        super(ModelBuilder_Proto, self).__init__()
        self._sync_bn = net_cfg["sync_bn"]
        self._num_classes = net_cfg["num_classes"]

        self.encoder = self._build_encoder(net_cfg["encoder"])
        net_cfg["decoder"]["type"] = 'augseg.models.decoder.decoder_prototype'
        self.decoder = self._build_decoder(net_cfg["decoder"])
        self.register_buffer("prototype", torch.randn(128, self._num_classes))
        self.prototype = F.normalize(self.prototype, dim=0)
        self.register_buffer("coefficient", torch.full((self._num_classes,),net_cfg["coefficient"]))
        self.register_buffer("last_update", torch.ones(self._num_classes,))
        self.coefficient = self.coefficient.float()
        self.last_update = self.last_update.long()
        self._use_auxloss = True if net_cfg.get("aux_loss", False) else False
        if self._use_auxloss:
            cfg_aux = net_cfg["aux_loss"]
            self.loss_weight = cfg_aux["loss_weight"]
            self.auxor = Aux_Module(
                cfg_aux["aux_plane"], self._num_classes, self._sync_bn
            )

    def _build_encoder(self, enc_cfg):
        enc_cfg["kwargs"].update({"sync_bn": self._sync_bn})
        pretrained_model_url = enc_cfg["pretrain"]
        encoder = self._build_module(enc_cfg["type"], enc_cfg["kwargs"], pretrain_model_url=pretrained_model_url)
        return encoder

    def _build_decoder(self, dec_cfg):
        dec_cfg["kwargs"].update(
            {
                "in_planes": self.encoder.get_outplanes(),
                "sync_bn": self._sync_bn,
                "num_classes": self._num_classes,
            }
        )
        decoder = self._build_module(dec_cfg["type"], dec_cfg["kwargs"])
        return decoder

    def _build_module(self, mtype, kwargs, pretrain_model_url=None):
        module_name, class_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        if pretrain_model_url is None:
            return cls(**kwargs)
        else:
            return cls(pretrain_model_url=pretrain_model_url, **kwargs)
    
    @torch.no_grad()
    def update_prototype(self, feat, label):
        feat = feat.permute(0,2,3,1)
        feat = F.normalize(feat, dim=-1)
        for i in range(self._num_classes):
            if i in label:
                label_mask=(label==i)
                feat_now = feat[label_mask].mean(dim=0)
                #NOTE update the i th prototype using feat_now
                c = self.coefficient[i] ** self.last_update[i]
                self.prototype[:,i] = self.prototype[:,i]*c + feat_now*(1-c)
                self.last_update[i] = 1
                #self.prototype[i] = self.prototype[i]*(self.coefficient**iter) + feat_now*(1-c**iter)
                # c --> c**iter 
            else:
                #NOTE update the coefficients
                self.last_update[i] += 1
        self.prototype = F.normalize(self.prototype, dim=0)

    def forward(self, x, mask=None, return_proj=False, tem=1.):
        h, w = x.shape[-2:]
        if self._use_auxloss:
            f1, f2, feat1, feat2 = self.encoder(x)
            outs = self.decoder([f1, f2, feat1, feat2])
            pred_aux = self.auxor(feat1)

            # upsampling
            outs = F.interpolate(outs, (h, w), mode="bilinear", align_corners=True)
            pred_aux = F.interpolate(pred_aux, (h, w), mode="bilinear", align_corners=True)
            
            return outs, pred_aux
        else:
            pool_list = None
            feat = self.encoder(x)
            if mask==None:
                pred_outs, proto_outs = self.decoder(feat, project=True) # N, 128, H, W 

            pred_outs = F.interpolate(pred_outs, (h, w), mode="bilinear", align_corners=True)
            proto_outs_resize = F.interpolate(proto_outs, (h, w), mode="bilinear", align_corners=True)
            
            proto_outs = F.normalize(proto_outs, dim=1)
            prototype = F.normalize(self.prototype, dim=0).detach()

            relations = (proto_outs.permute(0,2,3,1) @ prototype).permute(0,3,1,2) #N num_class, H, W
            relations = F.softmax(relations/tem, dim=1)

            relations = F.interpolate(relations, (h, w), mode="bilinear", align_corners=True)
            if return_proj:
                return pred_outs, relations, proto_outs_resize
            
            return pred_outs, relations