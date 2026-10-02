"""Appearance embedding backends. All return L2-normalised float32 vectors (cosine distance = 1 - dot).

colorhist : HSV strip histograms. CPU-only, no weights, ~instant. Default for the synthetic demo / CPU fallback.
resnet18  : torchvision ResNet-18 (ImageNet weights are downloaded by torchvision on first use). Generic features.
osnet     : OSNet via the `torchreid` package (https://github.com/KaiyangZhou/deep-person-reid). ReID weights
            (e.g. osnet_x0_25_msmt17.pth) must be downloaded separately and passed via data.embedder_weights.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

from ..utils import device_info, get_logger, resolve_path

log = get_logger()


class Embedder:
    name = "base"
    dim = 0

    def embed(self, crops: List[np.ndarray]) -> np.ndarray:  # BGR uint8 crops
        raise NotImplementedError


class ColorHistEmbedder(Embedder):
    name = "colorhist"

    def __init__(self, strips: int = 4, h_bins: int = 8, s_bins: int = 4, v_bins: int = 3):
        self.strips, self.h_bins, self.s_bins, self.v_bins = strips, h_bins, s_bins, v_bins
        self.dim = strips * (h_bins * s_bins + v_bins)

    def _one(self, crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            v = np.ones(self.dim, np.float32)
            return v / np.linalg.norm(v)
        h, w = crop.shape[:2]
        crop = crop[:, int(0.15 * w): max(int(0.85 * w), int(0.15 * w) + 1)]  # drop box edges (background)
        hsv = cv2.cvtColor(cv2.resize(crop, (16, 48), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        feats = []
        rows = hsv.shape[0] // self.strips
        for s in range(self.strips):
            part = hsv[s * rows:(s + 1) * rows].reshape(-1, 3).astype(np.float32)
            hh = np.clip((part[:, 0] / 180.0 * self.h_bins).astype(int), 0, self.h_bins - 1)
            ss = np.clip((part[:, 1] / 256.0 * self.s_bins).astype(int), 0, self.s_bins - 1)
            vv = np.clip((part[:, 2] / 256.0 * self.v_bins).astype(int), 0, self.v_bins - 1)
            hs = np.bincount(hh * self.s_bins + ss, minlength=self.h_bins * self.s_bins).astype(np.float32)
            vh = np.bincount(vv, minlength=self.v_bins).astype(np.float32)
            f = np.concatenate([hs, vh])
            feats.append(np.sqrt(f / max(f.sum(), 1.0)))  # Hellinger
        v = np.concatenate(feats)
        # centre: subtract the uniform-histogram mean so cosine distance spreads over a useful range
        v = v - v.mean()
        n = np.linalg.norm(v)
        return (v / n if n > 1e-9 else v).astype(np.float32)

    def embed(self, crops):
        return np.stack([self._one(c) for c in crops]) if crops else np.zeros((0, self.dim), np.float32)


class _TorchEmbedder(Embedder):
    def __init__(self, device: str):
        info = device_info(device)
        self.device = info["device"]
        log.info(f"Embedding backend device: {self.device} (CUDA available: {info['cuda_available']})")

    def _prep(self, crops):
        import torch  # type: ignore
        arr = []
        for c in crops:
            if c is None or c.size == 0:
                c = np.zeros((8, 4, 3), np.uint8)
            c = cv2.cvtColor(cv2.resize(c, (128, 256)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            arr.append((c - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32))
        return torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).to(self.device)


class ResNet18Embedder(_TorchEmbedder):
    name = "resnet18"
    dim = 512

    def __init__(self, device: str = "auto", pretrained: bool = True):
        super().__init__(device)
        import torch  # type: ignore
        import torchvision  # type: ignore
        m = torchvision.models.resnet18(weights="DEFAULT" if pretrained else None)
        m.fc = torch.nn.Identity()
        self.model = m.eval().to(self.device)

    def embed(self, crops):
        import torch  # type: ignore
        out = []
        with torch.no_grad():
            for i in range(0, len(crops), 128):
                f = self.model(self._prep(crops[i:i + 128]))
                out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, self.dim), np.float32)


class OSNetEmbedder(_TorchEmbedder):
    name = "osnet"

    def __init__(self, model_name: str = "osnet_x0_25", weights: Optional[str] = None, device: str = "auto"):
        super().__init__(device)
        try:
            from torchreid.utils import FeatureExtractor  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "The 'osnet' backend needs torchreid:\n  pip install git+https://github.com/KaiyangZhou/deep-person-reid.git\n"
                "(on Windows you may need the MSVC build tools). Alternatively use --embedder colorhist or resnet18.\n"
                f"Original error: {e}")
        if not weights or not resolve_path(weights).exists():
            raise FileNotFoundError(
                f"OSNet ReID weights not found at '{weights}'. Download e.g. osnet_x0_25_msmt17.pth from the torchreid "
                "model zoo (https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO) and set data.embedder_weights.")
        self.ext = FeatureExtractor(model_name=model_name, model_path=str(resolve_path(weights)),
                                    device=self.device, image_size=(256, 128))
        self.dim = int(self.ext(np.zeros((1, 256, 128, 3), np.uint8)).shape[1])

    def embed(self, crops):
        import torch  # type: ignore
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        rgb = [cv2.cvtColor(c if (c is not None and c.size) else np.zeros((8, 4, 3), np.uint8), cv2.COLOR_BGR2RGB) for c in crops]
        out = []
        for i in range(0, len(rgb), 128):
            f = self.ext(rgb[i:i + 128])
            out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def get_embedder(data_cfg: Dict, device: str = "auto") -> Embedder:
    name = data_cfg.get("embedder", "colorhist")
    if name == "colorhist":
        return ColorHistEmbedder()
    if name == "resnet18":
        return ResNet18Embedder(device)
    if name == "osnet":
        return OSNetEmbedder(data_cfg.get("embedder_model", "osnet_x0_25"), data_cfg.get("embedder_weights"), device)
    raise ValueError(f"Unknown embedder '{name}'. Choose colorhist | resnet18 | osnet.")
