from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class PoseNormLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        # epsilon will auto move to the same device when compute
        self.epsilon = torch.tensor(
            epsilon, dtype=torch.float, requires_grad=True)
        self.relu = nn.ReLU()
        self.keepdim = False

    def forward(self, pred, target):
        # (From SE2se()): motion[0:3] = translation, motion[3:6] = rotation
        T_o = pred[:, 0:3]
        R_o = pred[:, 3:6]
        T_t = target[:, 0:3]
        R_t = target[:, 3:6]


        # scale = 1/max(|T|, e), |.| is L2 norm
        scale_o = torch.maximum(torch.norm(pred, p=2, dim=1, keepdim=True),
                                self.epsilon)
        scale_t = torch.maximum(torch.norm(target, p=2, dim=1, keepdim=True),
                                self.epsilon)

        T_loss = torch.norm((T_o / scale_o) - (T_t / scale_t),
                            p=2, dim=1, keepdim=self.keepdim)
        R_loss = torch.norm(R_o - R_t, p=2, dim=1, keepdim=self.keepdim)
        # print(R_o)
        # print(R_t)
        # print(R_o-R_t)
        # print(R_loss)
        # print('===============================')
        # input()
        return T_loss.sum() , R_loss.sum()


class FlowLoss(nn.Module):
    # Reference from mmflow's endpoint_error()
    def __init__(self, p: int = 1,
                 q: Optional[float] = 4e-1,
                 eps: Optional[float] = 1e-2) -> None:
        super().__init__()
        self.p = p
        self.q = q
        self.eps = eps
        self.keepdim = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # shape N, H, W
        flow_loss = torch.norm(pred - target, self.p, dim=1)
        if self.q is not None and self.eps is not None:
            flow_loss = (flow_loss + self.eps)**self.q
        # Sum of all elements of a sample
        # Shape: N
        return flow_loss.mean(dim=(1, 2), keepdim=self.keepdim)


class WeightedLoss(nn.Module):
    def __init__(self, weights: List[float]) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, *losses: Tuple[torch.Tensor]):
        assert len(self.weights) == len(losses), "Length not match!"
        weighted_loss = 0
        for w, l in zip(self.weights, losses):
            weighted_loss += w * l
        return weighted_loss
