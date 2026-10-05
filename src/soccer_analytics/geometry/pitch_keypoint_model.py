"""HRNet-w48 pitch-keypoint model, ported from the reference project so a supplied checkpoint can be run.

The reference (`whisdev/soccer-video-detection-ai-agent`) detects 32 field markers with an HRNet heatmap model. Its
*weights are not distributed* - the 265 MB file in that repository is a Git-LFS pointer whose object was never
pushed, so a fresh clone cannot run it. This module therefore ships the architecture and the channel->template map
and nothing else: drop a checkpoint at ``data/models/keypoint_detect.pt`` (or pass ``--weights``) and it runs.

Everything here is deliberately isolated from the rest of the geometry package. :mod:`auto_register` is model-
agnostic and tested against the synthetic oracle; this is one possible source of the keypoints it consumes. The
architecture is exercised by a forward pass with random weights in the tests, which catches the shape and config
mistakes that are the realistic way a port like this goes wrong - but it cannot verify detection quality, because
that lives in weights we do not have.

The checkpoint must match the reference's layout: an HRNet-w48 built from the config below, ``NUM_JOINTS = 58``,
heatmaps at half the input resolution, and the last channel unused. :data:`TEMPLATE_CHANNEL_MAP` is the reference's
own `map_keypoints`, which pairs its heatmap channels with :mod:`soccer_analytics.geometry.pitch_template` indices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from soccer_analytics.geometry.auto_register import KeypointObservation
from soccer_analytics.geometry.pitch_keypoints import Keypoint, extract_heatmap_peaks

REPO_ROOT = Path(__file__).resolve().parents[3]  # src/soccer_analytics/geometry/x.py -> repo root
MODELS_DIR = REPO_ROOT / "data" / "models"
DEFAULT_WEIGHTS_NAME = "keypoint_detect.pt"
# The reference's own input size and peak threshold. It resizes to 960x540, the model returns heatmaps at half that,
# and `_extract_keypoints` maps them back with scale=2. Keeping the size fixes the scale arithmetic below.
MODEL_IMAGE_SIZE = (960, 540)
DEFAULT_PEAK_THRESHOLD = 0.2

# The reference's `map_keypoints`: heatmap channel (1-based, after the unused last channel is dropped) -> template
# index (1-based, into its `TEMPLATE_F0`). Re-express as 0-based on both sides in `TEMPLATE_CHANNEL_MAP`.
_REFERENCE_CHANNEL_MAP = {
    1: 1, 2: 14, 3: 25, 4: 2, 5: 10, 6: 18, 7: 26, 8: 3, 9: 7, 10: 23,
    11: 27, 20: 4, 21: 8, 22: 24, 23: 28, 24: 5, 25: 13, 26: 21, 27: 29,
    28: 6, 29: 17, 30: 30, 31: 11, 32: 15, 33: 19, 34: 12, 35: 16, 36: 20,
    45: 9, 50: 31, 52: 32, 57: 22,
}
# channel (0-based heatmap channel) -> template index (0-based)
TEMPLATE_CHANNEL_MAP = {channel - 1: template - 1 for channel, template in _REFERENCE_CHANNEL_MAP.items()}

# HRNet-w48 as the reference's `hrnetv2_w48.yaml` describes it, embedded rather than shipped as package data.
HRNET_W48_CONFIG = {
    "MODEL": {
        "IMAGE_SIZE": list(MODEL_IMAGE_SIZE),
        "NUM_JOINTS": 58,
        "EXTRA": {
            "FINAL_CONV_KERNEL": 1,
            "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": "BOTTLENECK", "NUM_BLOCKS": [4], "NUM_CHANNELS": [64], "FUSE_METHOD": "SUM"},
            "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC", "NUM_BLOCKS": [4, 4], "NUM_CHANNELS": [48, 96], "FUSE_METHOD": "SUM"},
            "STAGE3": {"NUM_MODULES": 4, "NUM_BRANCHES": 3, "BLOCK": "BASIC", "NUM_BLOCKS": [4, 4, 4], "NUM_CHANNELS": [48, 96, 192], "FUSE_METHOD": "SUM"},
            "STAGE4": {"NUM_MODULES": 3, "NUM_BRANCHES": 4, "BLOCK": "BASIC", "NUM_BLOCKS": [4, 4, 4, 4], "NUM_CHANNELS": [48, 96, 192, 384], "FUSE_METHOD": "SUM"},
        },
    }
}


# --------------------------------------------------------------------------------------------------------------
# Architecture (transcribed from the reference)
# --------------------------------------------------------------------------------------------------------------
_BN_MOMENTUM = 0.1


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=_BN_MOMENTUM)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=_BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=_BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=_BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=_BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


_BLOCKS = {"BASIC": _BasicBlock, "BOTTLENECK": _Bottleneck}


class _HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super().__init__()
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=_BN_MOMENTUM),
            )
        layers = [block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        return nn.ModuleList([self._make_one_branch(i, block, num_blocks, num_channels) for i in range(num_branches)])

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=_BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_inchannels[i], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_inchannels[i], momentum=_BN_MOMENTUM)))
                        else:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_inchannels[j], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_inchannels[j], momentum=_BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]), size=[x[i].shape[2], x[i].shape[3]], mode="bilinear")
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class HRNet(nn.Module):
    """HRNet-w48 keypoint head, output at half the input resolution (as the reference's)."""

    def __init__(self, config: dict):
        self.inplanes = 64
        extra = config["MODEL"]["EXTRA"]
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=_BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=_BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(_Bottleneck, 64, 64, 4)

        self.stage2_cfg = extra["STAGE2"]
        channels = [c * _BLOCKS[self.stage2_cfg["BLOCK"]].expansion for c in self.stage2_cfg["NUM_CHANNELS"]]
        self.transition1 = self._make_transition_layer([256], channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, channels)

        self.stage3_cfg = extra["STAGE3"]
        channels = [c * _BLOCKS[self.stage3_cfg["BLOCK"]].expansion for c in self.stage3_cfg["NUM_CHANNELS"]]
        self.transition2 = self._make_transition_layer(pre_stage_channels, channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, channels)

        self.stage4_cfg = extra["STAGE4"]
        channels = [c * _BLOCKS[self.stage4_cfg["BLOCK"]].expansion for c in self.stage4_cfg["NUM_CHANNELS"]]
        self.transition3 = self._make_transition_layer(pre_stage_channels, channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, channels, multi_scale_output=True)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        final_inp_channels = sum(pre_stage_channels) + self.inplanes
        self.head = nn.Sequential(nn.Sequential(
            nn.Conv2d(final_inp_channels, final_inp_channels, kernel_size=1),
            nn.BatchNorm2d(final_inp_channels, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_inp_channels, config["MODEL"]["NUM_JOINTS"], kernel_size=extra["FINAL_CONV_KERNEL"]),
            nn.Softmax(dim=1)))

    def _make_head(self, x, x_skip):
        x = self.upsample(x)
        x = torch.cat([x, x_skip], dim=1)
        return self.head(x)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i], momentum=_BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=_BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=_BN_MOMENTUM),
            )
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = _BLOCKS[layer_config["BLOCK"]]
        fuse_method = layer_config["FUSE_METHOD"]
        modules = []
        for i in range(num_modules):
            reset_multi_scale_output = True if multi_scale_output or i < num_modules - 1 else False
            modules.append(_HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x_skip = x.clone()
        x = self.relu(self.bn1(x))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        x_list = [self.transition1[i](x) if self.transition1[i] is not None else x for i in range(self.stage2_cfg["NUM_BRANCHES"])]
        y_list = self.stage2(x_list)

        x_list = [self.transition2[i](y_list[-1]) if self.transition2[i] is not None else y_list[i] for i in range(self.stage3_cfg["NUM_BRANCHES"])]
        y_list = self.stage3(x_list)

        x_list = [self.transition3[i](y_list[-1]) if self.transition3[i] is not None else y_list[i] for i in range(self.stage4_cfg["NUM_BRANCHES"])]
        x = self.stage4(x_list)

        height, width = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(height, width), mode="bilinear", align_corners=False)
        x2 = F.interpolate(x[2], size=(height, width), mode="bilinear", align_corners=False)
        x3 = F.interpolate(x[3], size=(height, width), mode="bilinear", align_corners=False)
        x = torch.cat([x[0], x1, x2, x3], 1)
        return self._make_head(x, x_skip)


def build_keypoint_model(config: dict | None = None) -> HRNet:
    """A randomly-initialised HRNet-w48; load a checkpoint onto it to detect anything."""
    return HRNet(config or HRNET_W48_CONFIG)


def resolve_keypoint_weights(weights: str | Path | None = None) -> str | None:
    """Locate a keypoint checkpoint, mirroring Stage A's resolution: an explicit path, then ``data/models/``.

    Unlike person detection there is no stock checkpoint Ultralytics can fetch, so nothing is downloaded: absent a
    file, this returns None and the caller reports that automatic registration needs one. The newest ``*.pt`` under
    ``data/models/`` wins, on the same reasoning as Stage A - that is where a checkpoint for this camera would land.
    """
    if weights is not None:
        return str(weights) if Path(weights).exists() else None
    directory = MODELS_DIR / DEFAULT_WEIGHTS_NAME
    if directory.exists():
        return str(directory)
    if not MODELS_DIR.exists():
        return None
    candidates = sorted(MODELS_DIR.glob("*.pt"), key=lambda path: path.stat().st_mtime)
    return str(candidates[-1]) if candidates else None


def load_keypoint_model(weights: str | Path | None = None, device: str | int | None = None) -> HRNet:
    """Build the model and load a supplied checkpoint, or raise with a message that says what to do about it."""
    path = resolve_keypoint_weights(weights)
    if path is None:
        raise FileNotFoundError(
            "no pitch-keypoint checkpoint found. The reference project's keypoint weights are not distributed, so "
            f"one must be supplied: place it at {MODELS_DIR / DEFAULT_WEIGHTS_NAME} (or pass a path)."
        )
    model = build_keypoint_model()
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    return model.to(device).eval()


# --------------------------------------------------------------------------------------------------------------
# Inference: frame -> template keypoints
# --------------------------------------------------------------------------------------------------------------


def _preprocess(frame: np.ndarray, image_size: tuple[int, int] = MODEL_IMAGE_SIZE) -> torch.Tensor:
    import cv2

    height, width = int(image_size[1]), int(image_size[0])
    resized = cv2.resize(frame, (width, height)).astype(np.float32) / 255.0
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float()[None]


def keypoint_peaks(model, frame: np.ndarray, threshold: float = DEFAULT_PEAK_THRESHOLD) -> dict[int, Keypoint]:
    """Heatmap peaks in *input-pixel* coordinates, keyed by heatmap channel (0-based).

    The model's last channel is dropped before the peaks are read, exactly as the reference does. Coordinates are
    scaled from the half-resolution heatmap back to the model's input pixels (the reference's `scale=2`), and are
    therefore independent of the original frame size.
    """
    device = next(model.parameters()).device if isinstance(model, nn.Module) else "cpu"
    batch = _preprocess(frame).to(device)
    with torch.no_grad():
        heatmaps = model(batch)
    planes = heatmaps[0, :-1]  # the reference discards the final channel
    _batch, channels, height, width = heatmaps.shape
    scale = MODEL_IMAGE_SIZE[0] / max(width, 1)  # heatmap pixel -> input pixel
    return extract_heatmap_peaks(planes, threshold=threshold, scale=scale)


def keypoint_observations(
    model,
    frame: np.ndarray,
    frame_index: int,
    *,
    threshold: float = DEFAULT_PEAK_THRESHOLD,
    image_size: tuple[int, int] = MODEL_IMAGE_SIZE,
) -> list[KeypointObservation]:
    """Detections for one frame as solver observations, mapped onto the pitch template.

    Both coordinates are divided by the model input *width*, the solver's convention, so a detection is independent
    of the original frame's resolution. Channels with no template entry are dropped.
    """
    peaks = keypoint_peaks(model, frame, threshold=threshold)
    out: list[KeypointObservation] = []
    width = float(image_size[0])
    for channel, peak in peaks.items():
        index = TEMPLATE_CHANNEL_MAP.get(channel)
        if index is None:
            continue
        out.append(KeypointObservation(
            frame=frame_index,
            index=index,
            u=peak["x"] / width,
            v=peak["y"] / width,
            confidence=peak["p"],
        ))
    return out