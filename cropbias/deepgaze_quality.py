"""DeepGaze IIE (Linardos et al., 2021) as a second, independent judge of preview quality.

The crop rules never see this model. For each scored composite we keep the predicted
fixation density summed over rows, so the share inside any full-height crop is a
simple sum. A flat center bias is used, because the center of a side-by-side
composite is the seam between the two photos.
"""
from __future__ import annotations

import os
import pickle
from collections import OrderedDict
from unittest import mock

import numpy as np
from PIL import Image
from tqdm import tqdm

from . import config as C
from .composites import composite_path

WEIGHTS_URL = "https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/deepgaze2e.pth"
STORE = C.WORK / "deepgaze" / "test.pkl"


def load_model():
    """Build DeepGaze IIE without downloading ImageNet backbones or running hub code.
    deepgaze2e.pth already holds every weight, and load_state_dict(strict=True) checks it."""
    import torch
    import torchvision
    from deepgaze_pytorch.features.efficientnet_pytorch import EfficientNet

    hub = {"resnext50_32x4d": lambda: torchvision.models.resnext50_32x4d(weights=None),
           "densenet201": lambda: torchvision.models.densenet201(weights=None)}
    patches = [
        mock.patch("torch.hub.load", lambda repo, name, *a, **k: hub[name]()),
        mock.patch.object(EfficientNet, "from_pretrained",
                          classmethod(lambda cls, name, *a, **k: cls.from_name(name))),
        mock.patch("deepgaze_pytorch.features.shapenet.load_model",
                   lambda name: torch.nn.Sequential(OrderedDict(
                       [("module", torchvision.models.resnet50(weights=None))]))),
    ]
    for p in patches:
        p.start()
    try:
        from deepgaze_pytorch.deepgaze2e import DeepGazeIIE
        model = DeepGazeIIE(pretrained=False)
    finally:
        for p in patches:
            p.stop()
    state = torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.eval()


def column_density(model, img: Image.Image) -> np.ndarray:
    import torch
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    H, W = a.shape[:2]
    x = torch.tensor(a.transpose(2, 0, 1)[None])
    cb = torch.full((1, H, W), -float(np.log(H * W)))
    with torch.no_grad():
        logd = model(x, cb)[0, 0].numpy().astype(np.float64)
    d = np.exp(logd - logd.max())
    d /= d.sum()
    return d.sum(axis=0).astype(np.float32)


def run(pair_ids: list[str], split: str = "test") -> dict[str, np.ndarray]:
    store = pickle.loads(STORE.read_bytes()) if STORE.exists() else {}
    todo = [p for p in pair_ids if p not in store]
    if todo:
        model = load_model()
        for i, pid in enumerate(tqdm(todo, desc="deepgaze")):
            store[pid] = column_density(model, Image.open(composite_path(pid, split)))
            if (i + 1) % 200 == 0 or i + 1 == len(todo):
                STORE.parent.mkdir(parents=True, exist_ok=True)
                tmp = STORE.with_suffix(".tmp")
                tmp.write_bytes(pickle.dumps(store, protocol=pickle.HIGHEST_PROTOCOL))
                os.replace(tmp, STORE)
    return store
