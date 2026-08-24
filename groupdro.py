

from typing import List, Optional

import torch
import torch.nn as nn


class GroupDRO(nn.Module):
    def __init__(self, base_loss_fn: nn.Module, num_classes: int,
                 device: torch.device, eta: float = 0.01, ema: float = 0.9,
                 q_floor: float = 0.05, age_edges=(70.0, 80.0),
                 verbose: bool = True):
        super().__init__()
        # clone the trainer's loss with reduction="none" so per-sample losses are
        # available. Every other setting (gamma, class weights, label smoothing)
        # is copied, so the DRO run and the baseline share an identical loss.
        self.per_sample = type(base_loss_fn)(
            gamma=getattr(base_loss_fn, "gamma", 1.5),
            weight=getattr(base_loss_fn, "weight", None),
            label_smoothing=getattr(base_loss_fn, "label_smoothing", 0.0),
            reduction="none",
        )
        self.num_classes = int(num_classes)
        self.age_edges = tuple(float(e) for e in age_edges)
        self.n_groups = 3 * self.num_classes + 1      # +1 catch-all for missing age
        self.eta = float(eta)
        self.ema = float(ema)
        # Exponentiated gradient saturates: with a persistent loss gap, q drifts
        # to 1.0 on a single group and the model then trains on that group alone.
        # Mixing a floor of uniform mass back in each step bounds how far it can
        # go, so no group is ever fully abandoned. Unit-tested: without this, q
        # reached 1.0000 on one group within 600 steps.
        self.q_floor = float(q_floor)
        self.device = device
        self.q = torch.full((self.n_groups,), 1.0 / self.n_groups, device=device)
        self.loss_ema = torch.zeros(self.n_groups, device=device)
        self.seen = torch.zeros(self.n_groups, device=device)
        self.verbose = bool(verbose)
        self._steps = 0

    def age_bin(self, age: Optional[float]) -> int:
        if age is None or age != age:   # None or NaN
            return -1
        if age < self.age_edges[0]:
            return 0
        if age < self.age_edges[1]:
            return 1
        return 2

    def group_ids(self, y: torch.Tensor, meta: List[dict]) -> torch.Tensor:
        ids = []
        for i, lab in enumerate(y.tolist()):
            a = meta[i].get("age") if i < len(meta) else None
            b = self.age_bin(a)
            ids.append(self.n_groups - 1 if b < 0 else b * self.num_classes + int(lab))
        return torch.tensor(ids, dtype=torch.long, device=y.device)

    def forward(self, logits: torch.Tensor, y: torch.Tensor,
                meta: List[dict]) -> torch.Tensor:
        losses = self.per_sample(logits, y)  # [B]
        g = self.group_ids(y, meta)

        present = torch.unique(g)
        batch_mean = torch.zeros(self.n_groups, device=losses.device)
        for gi in present.tolist():
            batch_mean[gi] = losses[g == gi].mean()

        # update the EMA only for groups present in this micro-batch
        with torch.no_grad():
            det = batch_mean.detach()
            for gi in present.tolist():
                if self.seen[gi] == 0:
                    self.loss_ema[gi] = det[gi]
                else:
                    self.loss_ema[gi] = (self.ema * self.loss_ema[gi]
                                         + (1.0 - self.ema) * det[gi])
                self.seen[gi] += 1
            # exponentiated-gradient step on the smoothed per-group losses
            active = self.seen > 0
            if active.any():
                self.q[active] = self.q[active] * torch.exp(
                    self.eta * self.loss_ema[active])
                self.q[active] = self.q[active] / self.q[active].sum()
                if self.q_floor > 0:
                    n_act = int(active.sum())
                    self.q[active] = ((1.0 - self.q_floor) * self.q[active]
                                      + self.q_floor / n_act)
                    self.q[active] = self.q[active] / self.q[active].sum()
                self.q[~active] = 0.0

        self._steps += 1
        if self.verbose and self._steps % 200 == 0:
            top = torch.argsort(self.q, descending=True)[:3].tolist()
            print("  [dro] highest-weight groups "
                  + ", ".join(f"g{t} q={self.q[t]:.3f} L={self.loss_ema[t]:.3f}"
                              for t in top if self.seen[t] > 0))

        # weight only the groups actually present; renormalise over them so the
        # loss scale stays comparable to the baseline mean loss
        qp = self.q[present]
        if float(qp.sum()) <= 0:
            return losses.mean()
        qp = qp / qp.sum()
        return (qp * batch_mean[present]).sum()

    def report(self) -> str:
        lines = ["group  n_updates  q       loss_ema   (group = age_bin * K + label)"]
                                  ## #######
        for gi in range(self.n_groups): ####        ###
            if self.seen[gi] == 0:
                continue
            if gi == self.n_groups - 1:
                name = "missing-age"
            else:
                name = f"bin{gi // self.num_classes}/class{gi % self.num_classes}"
            lines.append(f"{name:<14} {int(self.seen[gi]):>6}  "
                         f"{float(self.q[gi]):.4f}  {float(self.loss_ema[gi]):.4f}")
        return "\n".join(lines)
