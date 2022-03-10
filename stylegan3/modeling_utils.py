import os
import re
import logging
from typing import *

import torch
import omegaconf
import yaml
import requests
import dnnlib
import numpy as np
import PIL.Image

from stylegan3 import legacy

logging.basicConfig(format='%(message)s', level=logging.INFO)

NETWORK_PKL_PATH = {
    "afhqv2":
    "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl",
    "metafaces":
    "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl",
}


def parse_vec2(s: Union[str, Tuple[float, float]], ) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')


def make_transform(
    translate: Tuple[float, float],
    angle: float,
):
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


def load_stylegan(
    config: omegaconf.dictconfig.DictConfig,
    ckpt_path: str = None,
) -> torch.nn.Module:
    model = None

    return model.eval()


def download_model(
    model_name: str = "imagenet-16384",
    force_download: bool = False,
):
    modeling_config_path = ""
    modeling_ckpt_path = ""
    return modeling_config_path, modeling_ckpt_path


def load_model(model_name: str = "afhqv2", ):
    network_pkl = NETWORK_PKL_PATH[model_name]

    logging.info('Loading networks from "%s"...' % network_pkl)

    with dnnlib.util.open_url(network_pkl, ) as f:
        G = legacy.load_network_pkl(f)['G_ema']

    return G


def generate_images(
    model_name: str = "afhqv2",
    seeds: List[int] = [0],
    truncation_psi: float = 1.,
    noise_mode: str = "const",
    outdir: str = "./outputs",
    translate: Tuple[float, float] = (0, 0),
    rotate: float = 0.,
    class_idx: Optional[int] = None,
    device: str = "cuda",
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    \b
    # Generate uncurated images with truncation using the MetFaces-U dataset
    python gen_images.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl
    """
    if device == "cuda":
        device = torch.device("cuda")

    G = load_model(model_name=model_name, )
    G = G.to(device, )

    os.makedirs(
        outdir,
        exist_ok=True,
    )

    label = torch.zeros(
        [1, G.c_dim],
        device=device,
    )
    if G.c_dim != 0:
        if class_idx is None:
            raise Exception(
                'Must specify class label with --class when using a conditional network'
            )
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print(
                'warn: --class=lbl ignored when running on an unconditional network'
            )

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' %
              (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(
            1, G.z_dim)).to(device)

        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        img = G(
            z,
            label,
            truncation_psi=truncation_psi,
            noise_mode=noise_mode,
        )
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(
            torch.uint8)

        PIL.Image.fromarray(img[0].cpu().numpy(),
                            'RGB').save(f'{outdir}/seed{seed:04d}.png')


if __name__ == "__main__":
    generate_images(model_name="afhqv2", )
