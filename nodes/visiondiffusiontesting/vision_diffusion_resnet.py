from base_policy import Policy  # LLMVISIONUPDATE: policy will now include image conditioning
import numpy as np
from tqdm.auto import tqdm
import pickle
import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
from typing import Union

from diffusers.training_utils import EMAModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.optimization import get_scheduler
import collections


import torchvision.transforms as T

from PIL import Image

# Credit -- Diffusion related work based on 
# https://github.com/real-stanford/diffusion_policy

########## Diffusion Data Processing Helper Functions #########


import torch
import numpy as np
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import pickle
import os

from torchvision import transforms as T
import torchvision
import torch.nn as nn

from typing import Callable


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

def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
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

def precompute_resnet_feats(images_arr: np.ndarray,
                            batch_size=32,
                            device='cuda',
                            model_name='resnet18',
                            use_groupnorm=True,
                            distort = False):
    """
    images_arr: np.ndarray of shape (T, H, W, 3)
    Returns: np.ndarray of shape (T, feat_dim)
    """
    assert images_arr.ndim == 4 and images_arr.shape[-1] == 3, "Expected (T, H, W, 3) array"

    model = get_resnet(model_name, weights="IMAGENET1K_V1")

    if use_groupnorm:
        replace_bn_with_gn(model)

    model = model.to(device).eval()

    return encode(model, images_arr, batch_size, device, distort=False)
   

def encode(model,images_arr,batch_size,device, distort=False):
    if not distort:
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225])
        ])
    else:
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
            batch = imgs_tensor[i:i+batch_size].to(device)
            output = model(batch).cpu()
            feats.append(output)

    return torch.cat(feats, dim=0).numpy()  # shape (T, feat_dim)


class TrajectoryDatasetDiffusion(Dataset):
    def __init__(self, states, actions, images, pred_horizon, obs_horizon, action_horizon, distort=False):
        # LLMVISIONUPDATE: using precomputed image features
        num_episodes = len(states)
        obs_dim = np.shape(states[0])[0]
        action_dim = np.shape(actions[0])[0]

        if distort:
            print("Creating new dataset with image distortions")

        # Process visual features
        image_feats = []
        for image_ep in images:
            image_feats.append(precompute_resnet_feats(image_ep,distort=distort))

        feat_dim = np.shape(image_feats[0])[1]  # assume (T_i, feat_dim)

        self.num_samples = 0 
        self.states_flat = np.zeros((0, obs_dim))
        self.actions_flat = np.zeros((0, action_dim))
        self.image_feats_flat = np.zeros((0, feat_dim))
        self.ends = []

        for ii in range(num_episodes):
            T_i = np.shape(states[ii])[1]
            self.num_samples += T_i
            self.states_flat = np.concatenate((self.states_flat, states[ii].T), axis=0)
            self.actions_flat = np.concatenate((self.actions_flat, actions[ii].T), axis=0)
            self.image_feats_flat = np.concatenate((self.image_feats_flat, image_feats[ii]), axis=0)
            self.ends.append(self.num_samples)

        train_data = {
            'action': self.actions_flat,
            'obs': self.states_flat
        }

        episode_ends = np.array(self.ends)

        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        stats = {}
        normalized_train_data = {}
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.image_feats_flat = torch.from_numpy(self.image_feats_flat).float()
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )
        nsample['obs'] = nsample['obs'][:self.obs_horizon, :]

        # LLMVISIONUPDATE: use precomputed image feature at obs_horizon-1
        feats_window = self.image_feats_flat[buffer_start_idx:buffer_end_idx]
        nsample['image_feat'] = feats_window[self.obs_horizon - 1].float()

        return nsample


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# helper function for creating subsets of state/action data
def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices

def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

########## END Diffusion Data Processing Helper Functions #########

######### Diffusion Model (u-net) Related ############

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv


    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x
    
    ######### END Diffusion Model (u-net) Related ############

class VisionDiffusionResnet(Policy):
    def __init__(self, num_epochs=100, pred_horizon=40, obs_horizon=1, action_horizon=4, diffusion_iterations=10, estimate_horizon=16):
        # existing initialization
        self.num_epochs = num_epochs
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.estimate_horizon = estimate_horizon
        self.diffusion_iterations = diffusion_iterations

        self.device = torch.device('cuda')
        # processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        # self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device).eval()
        self.model = get_resnet('resnet18', weights="IMAGENET1K_V1").to(self.device).eval()


        # LLMVISIONUPDATE: add image deque for inference
        self.obs_deque = None
        self.image_deque = None

        super().__init__()

    def setupData(self, states, actions, images, distort=False):
        dataset = TrajectoryDatasetDiffusion(
            states=states,
            actions=actions,
            images=images,
            pred_horizon=self.pred_horizon,
            obs_horizon=self.obs_horizon,
            action_horizon=self.action_horizon,
            distort=distort
        )
        dataloader = DataLoader(
            dataset,
            batch_size=256,
            num_workers=1,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True
        )
        return dataset, dataloader

    def train(self, states, actions, images, savelocation,save_epoch_freq=None,distort=False):
        

        obs_dim = np.shape(states[0])[0]
        action_dim = np.shape(actions[0])[0]
        # LLMVISIONUPDATE: determine image feature dimension
        dummy_img = torch.zeros((1, 3, 224, 224)).to(self.device)
        model = get_resnet('resnet18', weights="IMAGENET1K_V1").to(self.device)


        feat_dim = model(dummy_img).shape[-1]

        # LLMVISIONUPDATE: include image features in global conditioning dimension
        total_cond_dim = obs_dim * self.obs_horizon + feat_dim

        noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=total_cond_dim
        )

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.diffusion_iterations,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
            )

        device = torch.device('cuda')
        noise_pred_net.to(device)

        ema = EMAModel(parameters=noise_pred_net.parameters(), power=0.75)
        optimizer = torch.optim.AdamW(noise_pred_net.parameters(), lr=1e-4, weight_decay=1e-6)
        
        if not distort:
            dataset, dataloader = self.setupData(states, actions, images)

            # Cosine LR schedule with linear warmup
            lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=optimizer,
                num_warmup_steps=500,
                num_training_steps=len(dataloader) * self.num_epochs
            )

        # store diffusion_params including new feature dim
        self.diffusion_params = {
            'action_dim': action_dim,
            'obs_dim': obs_dim,
            'obs_horizon': self.obs_horizon,
            'pred_horizon': self.pred_horizon,
            'action_horizon': self.action_horizon,
            'diffusion_iterations': self.diffusion_iterations,
            'image_feat_dim': feat_dim,
        }
        all_actions = np.concatenate(actions, axis=1)  # list of (m, T_i) → (m, T_total)
        self.action_mins = np.min(all_actions, axis=1)
        self.action_maxs = np.max(all_actions, axis=1)
        # self.action_mins = np.min(actions, axis=(0,2))
        # self.action_maxs = np.max(actions, axis=(0,2))
        self.diffusion_params['action_mins'] = self.action_mins
        self.diffusion_params['action_maxs'] = self.action_maxs

        epoch_losses = []
        for epoch_idx in range(self.num_epochs):
            epoch_loss = []
            epoch_startt=time.time()
            if distort:
                dataset, dataloader = self.setupData(states, actions, images,distort=True)
                # Cosine LR schedule with linear warmup
                lr_scheduler = get_scheduler(
                    name='cosine',
                    optimizer=optimizer,
                    num_warmup_steps=500,
                    num_training_steps=len(dataloader) * self.num_epochs
                )
            for nbatch in dataloader:
                nobs = nbatch['obs'].to(device).float()
                naction = nbatch['action'].to(device).float()
                # LLMVISIONUPDATE: get image feature batch
                img_feat = nbatch['image_feat'].to(device).float()

                B = nobs.shape[0]
                obs_cond = nobs[:, :self.obs_horizon, :].flatten(start_dim=1)
                # LLMVISIONUPDATE: concatenate image features to conditioning
                obs_cond = torch.cat([obs_cond, img_feat], dim=1)

                noise = torch.randn(naction.shape, device=device)
                timesteps = torch.randint(0, self.diffusion_iterations, (B,), device=device)

                noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
                noise_pred = noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
                loss = nn.functional.mse_loss(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                ema.step(noise_pred_net.parameters())

                epoch_loss.append(loss.item())

            # try:
            if epoch_idx%save_epoch_freq==0 and epoch_idx>0:
                self.inference_model = copy.deepcopy(noise_pred_net)
                ema.copy_to(self.inference_model.parameters())
                self.dataset = dataset
                self.save(savelocation, epochid=epoch_idx)
            # except Exception as e:
            #     print(e)

            print("epoch ",epoch_idx," loss:",np.mean(epoch_loss), "time (s):",time.time()-epoch_startt)

            epoch_losses.append(np.mean(epoch_loss))
        ema.copy_to(noise_pred_net.parameters())
        self.ema_pred_net = ema
        self.noise_pred_net = noise_pred_net
        self.dataset = dataset


    def save(self, location, epochid=None):
        # location is a fully qualified path for the learned model
        # need to save a file with network and dataloader
        if epochid is None:
            torch.save(self.noise_pred_net.state_dict(),location)
        else:
            epoch_location = location.split(".pkl")[0] + "_epoch" +str(epochid)+".pkl"
            # haven't overwritten parameters
            torch.save(self.inference_model.state_dict(),epoch_location)

        location2 = location.split(".pkl")[0] + "_dataset.pkl"
        with open(location2, 'wb') as handle:
            pickle.dump((self.dataset,self.diffusion_params), handle)


    def load(self, location):
        # location is a fully qualified path for the learned model
        location2 = location.split(".pkl")[0] + "_dataset.pkl"
        with open(location2, 'rb') as handle:
            (dataset, self.diffusion_params) = pickle.load(handle)

        self.dataset = dataset # used for normalization stats
    
        self.action_dim = self.diffusion_params["action_dim"]
        self.obs_dim = self.diffusion_params["obs_dim"]
        self.obs_horizon = self.diffusion_params["obs_horizon"]
        self.pred_horizon = self.diffusion_params["pred_horizon"]
        self.action_horizon = self.diffusion_params["action_horizon"]
        self.diffusion_iterations = self.diffusion_params["diffusion_iterations"]
        self.feat_dim = self.diffusion_params['image_feat_dim']

        self.action_mins = self.diffusion_params["action_mins"]
        self.action_maxs = self.diffusion_params["action_maxs"]

        self.device = torch.device('cuda')
        
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim*self.obs_horizon + self.feat_dim
            )

        self.noise_pred_net.load_state_dict(torch.load(location, map_location='cpu'))
        self.noise_pred_net.eval()

        _ = self.noise_pred_net.to(self.device)

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.diffusion_iterations,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
            )


    def getAction(self, state, image, forecast=False):
        # LLMVISIONUPDATE: signature now accepts `image` alongside state
        if self.obs_deque is None:
            self.obs_deque = collections.deque([state] * self.obs_horizon, maxlen=self.obs_horizon)
            self.image_deque = collections.deque([image] * self.obs_horizon, maxlen=self.obs_horizon)
            self.noise_scheduler.set_timesteps(self.diffusion_iterations)
        self.obs_deque.append(state)
        print(type(image))
        self.image_deque.append(image)

        obs_seq = np.stack(self.obs_deque)
        nobs = normalize_data(obs_seq, stats=self.dataset.stats['obs'])
        nobs = torch.from_numpy(nobs).to(self.device)
        obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

        # LLMVISIONUPDATE: process latest image through DINOv2 for conditioning
        raw_img = self.image_deque[-1]


        img_feat = encode(self.model,[raw_img],1,self.device)
    
        # img_tensor = self.dataset.transform(raw_img).unsqueeze(0).to(self.device)
        # with torch.no_grad():
        #     img_feat = self.dataset.image_encoder(img_tensor).squeeze(0)
        print(np.shape(img_feat))
        obs_cond = obs_cond.float()  # Ensure float32
        obs_cond = torch.cat([obs_cond, torch.from_numpy(img_feat).float().to(self.device)], dim=1)

        noisy_action = torch.randn((1, self.pred_horizon, self.action_dim), device=self.device)
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(noisy_action, k, global_cond=obs_cond)
            noisy_action = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=noisy_action).prev_sample

        naction = noisy_action.cpu().detach().numpy()[0]
        generated = unnormalize_data(naction, stats=self.dataset.stats['action'])
        return generated if forecast else generated[0]

    def forecastAction(self, state, image, num_seeds, length):
            """
            Generate multiple action sequences (trajectories) starting from state and latest image.
            Returns an array of shape (num_seeds, length, action_dim).
            """
            return self.sampleActions(state, image, num_seeds, length)

    def sampleActions(self, state, image, num_samples, length):
        """
        Sample `num_samples` independent action sequences of given `length`.
        """
        if self.obs_deque is None:
            self.obs_deque = collections.deque([state] * self.obs_horizon, maxlen=self.obs_horizon)
            self.image_deque = collections.deque([image] * self.obs_horizon, maxlen=self.obs_horizon)
            self.noise_scheduler.set_timesteps(self.diffusion_iterations)
        self.obs_deque.append(state)
        self.image_deque.append(image)

        obs_seq = np.stack(self.obs_deque)
        nobs = normalize_data(obs_seq, stats=self.dataset.stats['obs'])
        nobs = torch.from_numpy(nobs).to(self.device, dtype=torch.float32)
        obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

        raw_img = self.image_deque[-1]
        img_tensor = self.dataset.transform(raw_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_feat = self.dataset.image_encoder(img_tensor).squeeze(0)

        # replicate conditioning for batch
        cond_state = obs_cond.repeat(num_samples, 1)
        cond_img = img_feat.unsqueeze(0).repeat(num_samples, 1)
        global_cond = torch.cat([cond_state, cond_img], dim=1)

        noisy_actions = torch.randn((num_samples, self.pred_horizon, self.action_dim), device=self.device)
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(noisy_actions, k, global_cond=global_cond)
            noisy_actions = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=noisy_actions).prev_sample

        naction = noisy_actions.cpu().numpy()
        generated = unnormalize_data(naction, stats=self.dataset.stats['action'])
        if length < generated.shape[1]:
            return generated[:, :length, :]
        return generated

    def reset(self):
        self.obs_deque = None
        self.image_deque = None