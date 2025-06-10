
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

# dinov2
from transformers import AutoModel, AutoImageProcessor


###############################################
# Don't process images -- becomes state-based #
###############################################

class NoImages:
    def __init__(self, device = 'cuda') -> None:
        self.model = None
        self.device = device

    def encode(self, images_arr, batch_size, distort = False):
        return np.zeros((len(images_arr),0)) # shape (T, feat_dim = 0)

    def precompute_feats(self, images_arr: np.ndarray,
                            batch_size=32,
                            distort = False):
        """
        images_arr: np.ndarray of shape (T, H, W, 3)
        Returns: np.ndarray of shape (T, feat_dim)
        """
        assert images_arr.ndim == 4 and images_arr.shape[-1] == 3, "Expected (T, H, W, 3) array"

        return self.encode(images_arr, batch_size, distort=False)


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


class ResNet:
    def __init__(self, device = 'cuda') -> None:
        self.model = self.get_resnet('resnet18', weights="IMAGENET1K_V1")
        self.device = device
        self.use_groupnorm = True

    def get_resnet(self, name:str, weights=None, **kwargs) -> nn.Module:
        """
        name: resnet18, resnet34, resnet50
        weights: "IMAGENET1K_V1", None
        """
        # Use standard ResNet implementation from torchvision
        func = getattr(torchvision.models, name)
        resnet = func(weights=weights, **kwargs)

        # remove the final fully connected layer
        # for resnet18, the output dim should be 512
        resnet.fc = torch.nn.Identity()
        return resnet

    def encode(self, images_arr, batch_size, distort=False):
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

        feats = []
        with torch.no_grad():
            for i in range(0, imgs_tensor.shape[0], batch_size):
                batch = imgs_tensor[i:i+batch_size].to(self.device)
                output = self.model(batch).cpu()
                feats.append(output)

        return torch.cat(feats, dim=0).numpy()  # shape (T, feat_dim)

    def precompute_feats(self,
                         images_arr: np.ndarray,
                         batch_size=32,
                         distort = False):
        """
        images_arr: np.ndarray of shape (T, H, W, 3)
        Returns: np.ndarray of shape (T, feat_dim)
        """
        assert images_arr.ndim == 4 and images_arr.shape[-1] == 3, "Expected (T, H, W, 3) array"


        if self.use_groupnorm:
            replace_bn_with_gn(self.model)

        self.model.to(self.device).eval()

        return self.encode(images_arr, batch_size, distort=distort)

class DinoV2:
    def __init__(self, device = 'cuda') -> None:
        self.model = AutoModel.from_pretrained("facebook/dinov2-base")
        self.device = device

    def encode(self, images_arr,batch_size, distort=False):
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

        feats = []
        with torch.no_grad():
            for i in range(0, imgs_tensor.shape[0], batch_size):
                batch = imgs_tensor[i:i+batch_size].to(self.device)
                output = self.model(batch).cpu()
                feats.append(output)

        return torch.cat(feats, dim=0).numpy()  # shape (T, feat_dim)

    def precompute_feats(self,
                         images_arr: np.ndarray,
                         batch_size=32,
                         distort = False):
        """
        images_arr: np.ndarray of shape (T, H, W, 3)
        Returns: np.ndarray of shape (T, feat_dim)
        """
        assert images_arr.ndim == 4 and images_arr.shape[-1] == 3, "Expected (T, H, W, 3) array"

        self.model.to(self.device).eval()

        return self.encode(images_arr, batch_size, distort=distort)


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
