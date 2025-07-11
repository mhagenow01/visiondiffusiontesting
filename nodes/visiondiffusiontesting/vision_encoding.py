
############################################################
# Helper Functions and Models for Anything Vision Related  #
############################################################s

import numpy as np
from typing import Callable
import torch
from PIL import Image

# resnet
import torchvision.transforms as T
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# dinov2
from transformers import AutoModel, AutoImageProcessor


###############################################
# Don't process images -- becomes state-based #
###############################################

class DummyVisionEncoder(nn.Module):
    def forward(self, images):
        B = images.shape[0]
        device = images.device
        return torch.empty((B, 0), device=device)


class NoImages:
    def __init__(self, device = 'cuda') -> None:
        self.model = DummyVisionEncoder()
        self.device = device

    def getModel(self):
        return self.model

    def transform(self, images_arr, distort=False):
    
        # just do imagenet transforms -- though encoder will throw away images
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225])
        ])

        # Preprocess all images
        tensors = [
            transform(Image.fromarray(img).convert("RGB"))
            for img in images_arr
        ]
        imgs_tensor = torch.stack(tensors, dim=0)  # (T, 3, 224, 224)

        return imgs_tensor

    def encode(self, images_arr, batch_size=None, device=None, distort=False):
        imgs_tensor = self.transform(images_arr, distort=distort).to(self.device)
        with torch.no_grad():
            output = self.model(imgs_tensor)
        return output  # shape (B, feat_dim)



###############################################
# Basic ResNet (18, but can be other)         #
###############################################

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


class SpatialSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # infer the feature and channel size
        N, C, H, W = x.shape
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, W, device=x.device),
            torch.linspace(-1.0, 1.0, H, device=x.device),
            indexing='xy'
        )
        pos_x = pos_x.reshape(H * W)
        pos_y = pos_y.reshape(H * W)

        x = x.view(N, C, H * W)
        softmax = F.softmax(x, dim=2)

        expected_x = torch.sum(pos_x * softmax, dim=2)
        expected_y = torch.sum(pos_y * softmax, dim=2)
        return torch.cat([expected_x, expected_y], dim=1)


class ResNet:
    def __init__(self, device = 'cuda') -> None:
        self.device = device
        self.use_groupnorm = True
        self.model = self.get_resnet('resnet18', weights="IMAGENET1K_V1").to(device)

    def get_resnet(self, name: str, weights=None, **kwargs) -> nn.Module:
        func = getattr(torchvision.models, name)
        resnet = func(weights=weights, **kwargs)

        # TODO: check this vs original paper or robomimic implementation

        # Remove avgpool and fc layers
        modules = list(resnet.children())[:-2]  # keep up to layer4
        self.feature_extractor = nn.Sequential(*modules)

        # Add spatial softmax in place of global avgpool
        self.spatial_softmax = SpatialSoftmax()

        # Compose into a final model
        model = nn.Sequential(
            self.feature_extractor,
            self.spatial_softmax
        )

        if self.use_groupnorm:
            model = replace_bn_with_gn(model)

        return model
    
    def getModel(self):
        return self.model
    
    def transform(self, images_arr, distort=False):
        if not distort:
            # just do imagenet transforms
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std =[0.229, 0.224, 0.225])
            ])
        else:
            # augment image & imagenet transforms
            transform = T.Compose([
                T.Resize((240, 240)),
                T.RandomCrop((224, 224)),
                T.ColorJitter(brightness=0.3, contrast=0.3),
                T.GaussianBlur(kernel_size=9,sigma=(0.1, 2.0)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.05)
            ])

        # Preprocess all images
        tensors = [
            transform(Image.fromarray(img).convert("RGB"))
            for img in images_arr
        ]
        imgs_tensor = torch.stack(tensors, dim=0)  # (T, 3, 224, 224)

        return imgs_tensor

    def encode(self, images_arr, batch_size=None, device=None, distort=False):
        imgs_tensor = self.transform(images_arr, distort=distort).to(self.device)
        with torch.no_grad():
            output = self.model(imgs_tensor)
        return output  # shape (B, feat_dim)


class DinoV2:
    def __init__(self, device = 'cuda') -> None:
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        self.device = device
    
    def getModel(self):
        return self.model

    def transform(self, images_arr, distort=False):
        if not distort:
            # just do imagenet transforms
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std =[0.229, 0.224, 0.225])
            ])
        else:
            # augment image & imagenet transforms
            transform = T.Compose([
                T.Resize((240, 240)),
                T.RandomCrop((224, 224)),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.GaussianBlur(kernel_size=3),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.05)
            ])

        # Preprocess all images
        tensors = [
            transform(Image.fromarray(img).convert("RGB"))
            for img in images_arr
        ]
        imgs_tensor = torch.stack(tensors, dim=0)  # (T, 3, 224, 224)

        return imgs_tensor

    def encode(self, images_arr, batch_size=None, device=None, distort=False):
        imgs_tensor = self.transform(images_arr, distort=distort).to(self.device)
        with torch.no_grad():
            output = self.model(imgs_tensor)
        return output  # shape (B, feat_dim)


def getVisionModel(modelname='none'):
    if modelname=='resnet18':
        return ResNet()
    elif modelname == 'dinov2':
         return DinoV2()
    elif modelname=='none': # don't use images
        return NoImages()
    else:
        print("Invalid Image Encoding Option. Exiting.")
        raise NotImplementedError() 
