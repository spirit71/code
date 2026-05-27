# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torchvision
from pathlib import Path


class SideAdapter(nn.Module):
    """LoPA side adapter used in EDTformer."""

    def __init__(
        self,
        d_features,
        d_hidden_features=4,
        act_layer=nn.GELU,
        skip_connect=True,
        alpha=0.5,
        init_zero=True,
    ):
        super().__init__()
        self.skip_connect = skip_connect
        self.act = act_layer()
        self.d_fc1 = nn.Linear(d_features, d_hidden_features)
        self.d_fc2 = nn.Linear(d_hidden_features, d_features)
        self.alpha = alpha
        self.init_zero = init_zero
        if self.init_zero:
            nn.init.constant_(self.d_fc2.weight, 0.0)
            if self.d_fc2.bias is not None:
                nn.init.constant_(self.d_fc2.bias, 0.0)

    def forward(self, x):
        xs = self.d_fc1(x)
        xs = self.act(xs)
        xs = self.d_fc2(xs)
        if self.skip_connect:
            return x + self.alpha * xs
        return xs


class _DinoBackbone(torch.nn.Module):
    """Shared loader/wrapper for DINO-family ViT backbones from torch.hub."""

    AVAILABLE_MODELS = []
    DEFAULT_MODEL = ""
    REPO_OR_DIR = ""

    def __init__(
        self,
        backbone_name,
        unfreeze_n_blocks=2,
        reshape_output=True,
        weights=None,
        use_lopa=False,
        lopa_rank=4,
        lopa_alpha=0.5,
        lopa_skip_connect=True,
        lopa_zero_init_last=True,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.reshape_output = reshape_output
        self.weights = weights
        self.use_lopa = use_lopa
        self.lopa_rank = lopa_rank
        self.lopa_alpha = lopa_alpha
        self.lopa_skip_connect = lopa_skip_connect
        self.lopa_zero_init_last = lopa_zero_init_last

        # Make sure the backbone_name is in the available models.
        if self.backbone_name not in self.AVAILABLE_MODELS:
            print(
                f"Backbone {self.backbone_name} is not recognized! "
                f"Using {self.DEFAULT_MODEL}."
            )
            self.backbone_name = self.DEFAULT_MODEL

        load_kwargs = {}
        if self.weights is not None:
            load_kwargs["weights"] = self.weights

        self.dino = self._safe_torchhub_load(
            self.REPO_OR_DIR,
            self.backbone_name,
            load_kwargs,
        )

        if not hasattr(self.dino, "blocks"):
            raise AttributeError(
                f"Loaded model '{self.backbone_name}' does not expose `blocks`, "
                "so staged freezing is not supported by this wrapper."
            )

        # Freeze all parameters first.
        for param in self.dino.parameters():
            param.requires_grad = False

        total_blocks = len(self.dino.blocks)
        if unfreeze_n_blocks < 0 or unfreeze_n_blocks > total_blocks:
            raise ValueError(
                f"unfreeze_n_blocks must be between 0 and {total_blocks}, got {unfreeze_n_blocks}."
            )

        self.adapters = None
        if self.use_lopa:
            if self.lopa_rank <= 0:
                raise ValueError(f"lopa_rank must be > 0 when use_lopa=True, got {self.lopa_rank}.")
            if self.unfreeze_n_blocks > 0:
                print(
                    f"[Backbone] use_lopa=True, overriding unfreeze_n_blocks={self.unfreeze_n_blocks} to 0 "
                    "to match LoPA fine-tuning (frozen backbone + trainable side adapters)."
                )
                self.unfreeze_n_blocks = 0
            self.adapters = nn.ModuleList(
                [
                    SideAdapter(
                        d_features=self.dino.embed_dim,
                        d_hidden_features=self.lopa_rank,
                        alpha=self.lopa_alpha,
                        skip_connect=self.lopa_skip_connect,
                        init_zero=self.lopa_zero_init_last,
                    )
                    for _ in range(total_blocks)
                ]
            )
        else:
            # Unfreeze the last few blocks.
            if unfreeze_n_blocks > 0:
                for block in self.dino.blocks[-unfreeze_n_blocks:]:
                    for param in block.parameters():
                        param.requires_grad = True

        self.out_channels = self.dino.embed_dim

    @staticmethod
    def _safe_torchhub_load(repo_or_dir: str, model_name: str, model_kwargs: dict):
        """
        Load a torch.hub model with rate-limit-safe behavior.

        1) Try GitHub source with `skip_validation=True` (avoids GitHub API 403 in many cases).
        2) If that fails, fallback to local cached repo in torch hub dir.
        """
        try:
            return torch.hub.load(
                repo_or_dir,
                model_name,
                trust_repo=True,
                skip_validation=True,
                **model_kwargs,
            )
        except Exception as github_error:
            hub_dir = Path(torch.hub.get_dir())
            if "/" not in repo_or_dir:
                raise RuntimeError(
                    f"Failed to load '{model_name}' from '{repo_or_dir}'."
                ) from github_error

            owner, repo = repo_or_dir.split("/", 1)
            local_candidates = [
                hub_dir / f"{owner}_{repo}_main",
                hub_dir / f"{owner}_{repo}_master",
            ]
            local_candidates.extend(sorted(hub_dir.glob(f"{owner}_{repo}_*")))

            checked_paths = []
            for local_repo_dir in local_candidates:
                if not local_repo_dir.is_dir():
                    continue
                checked_paths.append(str(local_repo_dir))
                try:
                    print(f"[Backbone] Falling back to local torch.hub cache: {local_repo_dir}")
                    return torch.hub.load(
                        str(local_repo_dir),
                        model_name,
                        source="local",
                        **model_kwargs,
                    )
                except Exception:
                    continue

            raise RuntimeError(
                f"Failed to load '{model_name}' from '{repo_or_dir}' due to remote access limits "
                f"and no usable local torch.hub cache was found.\n"
                f"Checked cache root: {hub_dir}\n"
                f"Candidate local repos: {checked_paths if checked_paths else 'none'}\n"
                "You can prefetch once with network access or set GITHUB_TOKEN to reduce rate-limit failures."
            ) from github_error

    @property
    def patch_size(self):
        patch_size = self.dino.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            return patch_size[0]
        return patch_size

    def _run_block(self, blk, x, rope_sincos):
        if rope_sincos is not None:
            return blk(x, rope_sincos)
        return blk(x)

    def forward(self, x):
        B, _, H, W = x.shape
        rope_sincos = None

        with torch.no_grad():
            tokens_out = self.dino.prepare_tokens_with_masks(x)
        if isinstance(tokens_out, tuple):
            x, hw_tuple = tokens_out
            if hasattr(self.dino, "rope_embed") and self.dino.rope_embed is not None:
                rope_sincos = self.dino.rope_embed(H=hw_tuple[0], W=hw_tuple[1])
        else:
            x = tokens_out

        if self.use_lopa:
            # LoPA (as in EDTformer):
            # y = adapter(y + blk(x)); x = y
            y = x.clone()
            for index, blk in enumerate(self.dino.blocks):
                with torch.no_grad():
                    x = self._run_block(blk, x, rope_sincos)
                y = self.adapters[index](y + x)
                x = y
        else:
            # Frozen blocks are run without grad.
            with torch.no_grad():
                frozen_until = len(self.dino.blocks) - self.unfreeze_n_blocks
                for blk in self.dino.blocks[:frozen_until]:
                    x = self._run_block(blk, x, rope_sincos)

            # Last blocks are trainable.
            if self.unfreeze_n_blocks > 0:
                for blk in self.dino.blocks[-self.unfreeze_n_blocks:]:
                    x = self._run_block(blk, x, rope_sincos)

        # Remove CLS + optional storage tokens before spatial reshape.
        num_prefix_tokens = 1 + int(getattr(self.dino, "n_storage_tokens", 0))
        x = x[:, num_prefix_tokens:]

        # reshape the output tensor to B, C, H, W
        if self.reshape_output:
            _, _, C = x.shape
            patch_size = self.patch_size
            x = x.permute(0, 2, 1).view(B, C, H // patch_size, W // patch_size)
        return x


class DinoV2(_DinoBackbone):
    AVAILABLE_MODELS = [
        "dinov2_vits14",
        "dinov2_vitb14",
        "dinov2_vitl14",
        "dinov2_vitg14",
        "dinov2_vits14_reg",
        "dinov2_vitb14_reg",
        "dinov2_vitl14_reg",
        "dinov2_vitg14_reg",
    ]
    DEFAULT_MODEL = "dinov2_vitb14"
    REPO_OR_DIR = "facebookresearch/dinov2"

    def __init__(
        self,
        backbone_name="dinov2_vitb14",
        unfreeze_n_blocks=2,
        reshape_output=True,
        weights=None,
        use_lopa=False,
        lopa_rank=4,
        lopa_alpha=0.5,
        lopa_skip_connect=True,
        lopa_zero_init_last=True,
    ):
        super().__init__(
            backbone_name=backbone_name,
            unfreeze_n_blocks=unfreeze_n_blocks,
            reshape_output=reshape_output,
            weights=weights,
            use_lopa=use_lopa,
            lopa_rank=lopa_rank,
            lopa_alpha=lopa_alpha,
            lopa_skip_connect=lopa_skip_connect,
            lopa_zero_init_last=lopa_zero_init_last,
        )


class DinoV3(_DinoBackbone):
    AVAILABLE_MODELS = [
        "dinov3_vits16",
        "dinov3_vits16plus",
        "dinov3_vitb16",
        "dinov3_vitl16",
        "dinov3_vitl16plus",
        "dinov3_vith16plus",
        "dinov3_vit7b16",
    ]
    DEFAULT_MODEL = "dinov3_vitb16"
    REPO_OR_DIR = "facebookresearch/dinov3"
    DEFAULT_LOCAL_WEIGHTS = "/home/code_qy_7_28/Bag-of-Queries/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

    def __init__(
        self,
        backbone_name="dinov3_vitb16",
        unfreeze_n_blocks=2,
        reshape_output=True,
        weights=None,
        use_lopa=False,
        lopa_rank=4,
        lopa_alpha=0.5,
        lopa_skip_connect=True,
        lopa_zero_init_last=True,
    ):
        if weights is None and Path(self.DEFAULT_LOCAL_WEIGHTS).is_file():
            weights = self.DEFAULT_LOCAL_WEIGHTS
            print(f"[Backbone] Using local DINOv3 weights: {weights}")

        super().__init__(
            backbone_name=backbone_name,
            unfreeze_n_blocks=unfreeze_n_blocks,
            reshape_output=reshape_output,
            weights=weights,
            use_lopa=use_lopa,
            lopa_rank=lopa_rank,
            lopa_alpha=lopa_alpha,
            lopa_skip_connect=lopa_skip_connect,
            lopa_zero_init_last=lopa_zero_init_last,
        )


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
