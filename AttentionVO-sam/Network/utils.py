from pathlib import Path
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn

from Datasets.utils import visflow


def freeze_params(params: torch.Tensor):
    if isinstance(params, nn.Module):
        params = params.parameters()
    for p in params:
        p.requires_grad = False

def save_flow(flow_batch: torch.Tensor, batch_index: int, batch_size: int, flowdir: Path):
    # flow: B * N * H * W
    flowcount = batch_index * batch_size
    for b in range(flow_batch.shape[0]):
        # .npy
        flow_b = flow_batch[b].transpose(1,2,0)
        np.save(f"{flowdir / f'{flowcount:06d}.npy'}",flow_b)
        # .png
        flow_vis = visflow(flow_b)
        cv.imwrite(f"{flowdir / f'{flowcount:06d}.png'}",flow_vis)
        flowcount += 1
    pass