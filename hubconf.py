dependencies = ['torch', 'torchvision']

import sys
import os
from typing import Optional

# Add BoQ's src directory directly to path
boq_root = os.path.dirname(__file__)  # Root of the cloned repo
sys.path.append(os.path.join(boq_root, "src"))  

import torch
from backbones import ResNet, DinoV2, DinoV3
from boq import BoQ

    

class VPRModel(torch.nn.Module):
    def __init__(self, 
                 backbone,
                 aggregator):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        
    def forward(self, x):
        x = self.backbone(x)
        x, attns = self.aggregator(x)
        return x, attns


AVAILABLE_BACKBONES = {
    # this list will be extended
    # "resnet18": [8192 , 4096],
    "resnet50": [16384],
    "dinov2": [12288],
    "dinov3": [12288],
}

MODEL_URLS = {
    "resnet50_16384": "https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/resnet50_16384.pth",
    "dinov2_12288": "https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/dinov2_12288.pth",
    # Optional: set by environment variable BOQ_DINOV3_12288_URL or pass model_url argument
    "dinov3_12288": os.environ.get("BOQ_DINOV3_12288_URL", ""),

    # "resnet50_4096": "",
}

def _resolve_model_url(backbone_name: str, output_dim: int, model_url: Optional[str] = None):
    if model_url:
        return model_url

    key = f"{backbone_name}_{output_dim}"
    url = MODEL_URLS.get(key, "")
    if url:
        return url

    raise ValueError(
        f"No checkpoint URL configured for {key}. "
        "Pass `model_url=...` or set env var `BOQ_DINOV3_12288_URL` (for dinov3)."
    )


def get_trained_boq(backbone_name="resnet50", output_dim=16384, model_url=None):
    if backbone_name not in AVAILABLE_BACKBONES:
        raise ValueError(f"backbone_name should be one of {list(AVAILABLE_BACKBONES.keys())}")
    try:
        output_dim = int(output_dim)
    except:
        raise ValueError(f"output_dim should be an integer, not a {type(output_dim)}")
    if output_dim not in AVAILABLE_BACKBONES[backbone_name]:
        raise ValueError(f"output_dim should be one of {AVAILABLE_BACKBONES[backbone_name]}")
    
    if "dinov2" in backbone_name:
        # load the backbone
        backbone = DinoV2()
        # load the aggregator
        aggregator = BoQ(
            in_channels=backbone.out_channels,  # make sure the backbone has out_channels attribute
            proj_channels=384,
            num_queries=64,
            num_layers=2,
            row_dim=output_dim//384, # 32 for dinov2
        )
    elif "dinov3" in backbone_name:
        # load the backbone
        backbone = DinoV3()
        # load the aggregator
        aggregator = BoQ(
            in_channels=backbone.out_channels,  # make sure the backbone has out_channels attribute
            proj_channels=384,
            num_queries=64,
            num_layers=2,
            row_dim=output_dim//384, # 32 for dinov3 at output_dim=12288
        )
        
    elif "resnet" in backbone_name:
        backbone = ResNet(
                backbone_name=backbone_name,
                crop_last_block=True,
            )
        aggregator = BoQ(
                in_channels=backbone.out_channels,  # make sure the backbone has out_channels attribute
                proj_channels=512,
                num_queries=64,
                num_layers=2,
                row_dim=output_dim//512, # 32 for resnet
            )

    vpr_model = VPRModel(
            backbone=backbone,
            aggregator=aggregator
        )
    
    checkpoint_url = _resolve_model_url(backbone_name, output_dim, model_url=model_url)
    vpr_model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            checkpoint_url,
            map_location=torch.device('cpu')
        )
    )
    return vpr_model
