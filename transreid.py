import copy

import torch
import torch.nn as nn

from vit_pytorch import SwinV2_B, PatchEmbed


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not False and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    def __init__(self, num_classes, args):
        super(build_transformer, self).__init__()
        model_path = args.pretrain
        self.l2norm = Normalize(2)

        print('using Transformer_type: {} as a backbone'.format(args.backbone))

        self.base = SwinV2_B(img_size=(args.height, args.width), stride=args.stride,
                             use_checkpoint=getattr(args, 'activation_checkpointing', False))
        self.in_planes = self.base.num_features

        self.base.load_param(model_path)
        print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.num_classes = num_classes
        if args.nonlinear:
            self.classifier = nn.Sequential(nn.Linear(self.in_planes, self.in_planes, bias=False),
                                            nn.ReLU(),
                                            nn.Linear(self.in_planes, self.num_classes, bias=False)
                                            )
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, cam_label=None, view_label=None):
        x = self.base.patch_embed(x)
        global_feat = self.base(x)

        feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)

            return global_feat, cls_score
        else:
            return self.l2norm(global_feat), self.l2norm(feat)


class modal_transformer(nn.Module):
    def __init__(self, num_classes, args):
        super(modal_transformer, self).__init__()
        model_path = args.pretrain
        self.l2norm = Normalize(2)

        print('using Transformer_type: {} as a backbone'.format(args.backbone))

        self.visible_module = PatchEmbed(img_size=(args.height, args.width), patch_stride=args.stride)
        self.thermal_module = PatchEmbed(img_size=(args.height, args.width), patch_stride=args.stride)
        self.base = SwinV2_B(img_size=(args.height, args.width), stride=args.stride,
                             use_checkpoint=getattr(args, 'activation_checkpointing', False))
        self.in_planes = self.base.num_features

        self.base.load_param(model_path)
        self.visible_module.load_param(model_path)
        self.thermal_module.load_param(model_path)
        print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.num_classes = num_classes
        if args.nonlinear:
            self.classifier = nn.Sequential(nn.Linear(self.in_planes, self.in_planes, bias=False),
                                            nn.ReLU(),
                                            nn.Linear(self.in_planes, self.num_classes, bias=False)
                                            )
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, visible, infrare, modal=0, cam_label=None, view_label=None):
        if modal == 0:
            x1 = self.visible_module(visible)
            x2 = self.thermal_module(infrare)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(visible)
        elif modal == 2:
            x = self.thermal_module(infrare)
        else:
            raise Exception(f"Excepted modal: [0, 1, 2], but got {modal}")

        global_feat = self.base(x)

        feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)

            return global_feat, cls_score
        else:
            return self.l2norm(global_feat), self.l2norm(feat)


def make_model(args, num_class):
    if args.two:
        model = modal_transformer(num_class, args)
        print("===========building transformer with modal stream ===========")
    else:
        model = build_transformer(num_class, args)
        print('===========building transformer===========')
    return model


# if __name__ == '__main__':
#     model = vit_base_patch16_224_TransReID(camera=6)
#     for key, value in model.named_parameters():
#         if not value.requires_grad:
#             continue
#         lr = 0.05
#         weight_decay = 0.0001
#         # if "bias" in key:
#         #     lr = lr * 2
#         #     weight_decay = 1e-4
#         # elif "bottleneck" or "classifier" or "sie_embed" in key:
#         if "bottleneck" in key or "classifier" in key or "sie_embed" in key:
#             pass
#         else:
#             lr /= 10
#         print(f"{key}: lr  {lr} \t weight_decay  {weight_decay}")
