# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import os
import sys
from importlib import import_module
from pathlib import Path

import torch
import torch.nn as nn
import torchvision


_DINOV2_REPO_DIR = Path(
    os.environ.get("DINOV2_REPO_DIR", "/root/.cache/torch/hub/facebookresearch_dinov2_main")
)
_DINOV2_CHECKPOINT_DIR = Path(
    os.environ.get("DINOV2_CHECKPOINT_DIR", "/root/.cache/torch/hub/checkpoints")
)
_DINOV2_CHECKPOINTS = {
    "dinov2_vits14": "dinov2_vits14_pretrain.pth",
    "dinov2_vitb14": "dinov2_vitb14_pretrain.pth",
    "dinov2_vitl14": "dinov2_vitl14_pretrain.pth",
    "dinov2_vitg14": "dinov2_vitg14_pretrain.pth",
}


def _load_dinov2_from_local_cache(backbone_name: str):
    ckpt_name = _DINOV2_CHECKPOINTS.get(backbone_name)
    if ckpt_name is None:
        raise ValueError(f"Unsupported DINOv2 backbone: {backbone_name}")

    ckpt_path = _DINOV2_CHECKPOINT_DIR / ckpt_name
    if not _DINOV2_REPO_DIR.is_dir() or not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Local DINOv2 assets not found for {backbone_name}. "
            f"repo={_DINOV2_REPO_DIR}, checkpoint={ckpt_path}"
        )

    repo_dir_str = str(_DINOV2_REPO_DIR)
    added_to_syspath = False
    if repo_dir_str not in sys.path:
        sys.path.insert(0, repo_dir_str)
        added_to_syspath = True

    try:
        hub_backbones = import_module("dinov2.hub.backbones")
        builder = getattr(hub_backbones, backbone_name)
        model = builder(pretrained=False)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        return model
    finally:
        if added_to_syspath:
            sys.path.remove(repo_dir_str)


class DinoV2(torch.nn.Module):
    AVAILABLE_MODELS = [
        'dinov2_vits14',
        'dinov2_vitb14',
        'dinov2_vitl14',
        'dinov2_vitg14'
    ]
    
    def __init__(
        self,
        backbone_name="dinov2_vitb14",
        unfreeze_n_blocks=2,
        reshape_output=True,
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.reshape_output = reshape_output
        
        # make sure the backbone_name is in the available models
        if self.backbone_name not in self.AVAILABLE_MODELS:
            print(f"Backbone {self.backbone_name} is not recognized!, using dinov2_vitb14")
            self.backbone_name = "dinov2_vitb14"

        try:
            self.dino = _load_dinov2_from_local_cache(self.backbone_name)
            print(f"[INFO] Loaded {self.backbone_name} from local DINOv2 cache")
        except Exception as e:
            print(f"[WARN] Local DINOv2 cache load failed: {e}")
            print(f"[WARN] Falling back to torch.hub.load for {self.backbone_name}")
            self.dino = torch.hub.load('facebookresearch/dinov2', self.backbone_name)
        
        # freeze all parameters
        for param in self.dino.parameters():
            param.requires_grad = False
        
        # unfreeze the last few blocks
        for block in self.dino.blocks[ -unfreeze_n_blocks : ]:
            for param in block.parameters():
                param.requires_grad = True
        
        self.out_channels = self.dino.embed_dim
        
    @property
    def patch_size(self):
        return self.dino.patch_embed.patch_size[0]  # Assuming square patches
    
    def forward(self, x):
        B, _, H, W = x.shape  #[512, 3, 224, 224]
        # No need to compute gradients for frozen layers
        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(x)  #[512, 257, 768]
            for blk in self.dino.blocks[ : -self.unfreeze_n_blocks]:
                x = blk(x) #[512, 257, 768]
        # Last blocks are trained
        for blk in self.dino.blocks[-self.unfreeze_n_blocks : ]:
            x = blk(x)
        
        x = x[:, 1:] # remove the [CLS] token  这一块的输出x.shape 怎么还是[512, 257, 768]？
        # reshape the output tensor to B, C, H, W
        if self.reshape_output:
            _, _, C = x.shape # or C = self.embed_dim
            patch_size = self.patch_size #14
            x = x.permute(0, 2, 1).view(B, C, H // patch_size, W // patch_size) #在做什么？
        return x
    
    
class ResNet(nn.Module):
    AVAILABLE_MODELS = {
        "resnet18": torchvision.models.resnet18,
        "resnet34": torchvision.models.resnet34,
        "resnet50": torchvision.models.resnet50,
        "resnet101": torchvision.models.resnet101,
        "resnet152": torchvision.models.resnet152,
        "resnext50": torchvision.models.resnext50_32x4d,
    }

    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        unfreeze_n_blocks=1,
        crop_last_block=True,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.crop_last_block = crop_last_block

        if backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Backbone {backbone_name} is not recognized!" 
                             f"Supported backbones are: {list(self.AVAILABLE_MODELS.keys())}")

        # Load the model
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = self.AVAILABLE_MODELS[backbone_name](weights=weights)

        # Create backbone with only the necessary layers
        self.net = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            *([] if crop_last_block else [resnet.layer4]),
        )

        # Handle trainable/frozen layers
        nb_layers = len(self.net)
        assert (
            isinstance(unfreeze_n_blocks, int) and 0 <= unfreeze_n_blocks <= nb_layers
        ), f"unfreeze_n_blocks must be an integer between 0 and {nb_layers} (inclusive)"

        if pretrained:
            # Freeze required layers
            for layer in self.net[:nb_layers - unfreeze_n_blocks]:
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            if self.unfreeze_n_blocks > 0:
                print("Warning: unfreeze_n_blocks is ignored when pretrained=False. Setting it to 0.")
                self.unfreeze_n_blocks = 0

        # Output channels
        if backbone_name in ["resnet18", "resnet34"]:
            self.out_channels = resnet.layer3[-1].conv2.out_channels
        else:
            self.out_channels = resnet.layer3[-1].conv3.out_channels

    def forward(self, x):
        return self.net(x)
