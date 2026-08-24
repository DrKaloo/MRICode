

import torch
import torch.nn as nn


class GradientReversal(torch.autograd.Function):
    """Identity forwards; negated and scaled gradient backwards."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradientReversal.apply(x, lambd)


class AgeAdversary(nn.Module):
    """
    Predicts age from the penultimate feature vector, through a reversal layer.

    mode='regress'  -> predicts age in years, SmoothL1 loss (recommended)
    mode='classify' -> predicts the age bin, cross-entropy loss
    """

    def __init__(self, in_dim, hidden=128, mode="regress", n_bins=3, dropout=0.2):
        super().__init__()
        self.mode = mode
        out_dim = 1 if mode == "regress" else n_bins
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, out_dim),
        )
        self.reg_loss = nn.SmoothL1Loss()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, feats, lambd=1.0):
        return self.net(grad_reverse(feats, lambd))

    def loss(self, pred, target, age_mean=75.0, age_sd=8.0):
        if self.mode == "regress":
            t = (target.float() - age_mean) / age_sd      # standardise: keeps
            return self.reg_loss(pred.squeeze(-1), t)     # the scale comparable
        return self.cls_loss(pred, target.long())


def lambda_schedule(epoch, total_epochs, max_lambda=1.0, warmup_frac=0.25):
    """
    Ramp lambda from 0 to max_lambda over the first warmup_frac of training.

    Starting at full strength is the most common cause of collapse: the encoder
    is pushed to destroy age information before it has learned anything useful
    to preserve.
    """
    warm = max(1, int(total_epochs * warmup_frac))
    if epoch >= warm:
        return max_lambda
    p = epoch / warm
    return float(max_lambda * (2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))) - 1.0))


def extract_features(model, x):
    """
    Return the penultimate vector, i.e. the input to the classification head.
    Works for the three backbones used here, all of which end in
    (pooling -> dropout -> single Linear).
    """
    feats = {}
    head = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            head = mod
    if head is None:
        raise RuntimeError("no Linear head found on this model")
    h = head.register_forward_hook(lambda _m, inp, _o: feats.__setitem__("v", inp[0]))
    out = model(x)
    h.remove()
    return out, feats["v"]


# How to connect it to the trainer
#
# 1. argparse, next to the group_dro flags:
#
#       ap.add_argument("--adv_age", action="store_true")
#       ap.add_argument("--adv_lambda", type=float, default=1.0)
#       ap.add_argument("--adv_hidden", type=int, default=128)
#       ap.add_argument("--adv_mode", default="regress",
#                       choices=["regress", "classify"])
#
# 2. after the model is built and moved to the device:
#
#       adv = None
#       if args.adv_age:
#           from age_adversary import (AgeAdversary, lambda_schedule,
#                                      extract_features)
#           feat_dim = <512 for r3d_18 and MedicalNet ResNet-18,
#                       1280 for MobileNet3D-V2>
#           adv = AgeAdversary(feat_dim, hidden=args.adv_hidden,
#                              mode=args.adv_mode).to(device)
#           optimizer.add_param_group({"params": adv.parameters(),
#                                      "lr": args.lr})
#
#    The adversary goes in the == optimizer. The reversal layer already
#    flips the sign for the encoder, so a second optimizer is not needed and
#    would double-count the update.
#
# 3. the batch must carry age. If your Dataset does not return it, add it
#    alongside the label; the split CSVs already have the column.
#
# 4. inside the training loop, replacing the plain forward:
#
#       if adv is not None:
#           logits, feats = extract_features(model, xb)
#           lam = lambda_schedule(epoch, args.epochs, args.adv_lambda)
#           adv_pred = adv(feats, lam)
#           loss_cls = criterion(logits, yb)
#           loss_adv = adv.loss(adv_pred, age_b)
#           loss = loss_cls + loss_adv
#       else:
#           logits = model(xb)
#           loss = criterion(logits, yb)
#
#    With gradient accumulation, divide `loss` by accum_steps exactly as you
#    already do for the classification loss.
#
# 5. log both losses every epoch. loss_adv RISING while loss_cls falls is the
#    signature of the method working. Both falling together means the encoder
#    is winning and no invariance is being imposed. loss_cls rising sharply
#    means collapse: reduce --adv_lambda to 0.3 and re-run.
#
# 6. checkpoint selection stays on validation macro-F1, unchanged, so the
#    comparison against the baseline remains like-for-like.
#
# 7. evaluate exactly as any other run, then re-run the age probe on the
#    resulting checkpoints. The result to report is the pair: change in
#    macro-F1 against change in age R2.
#
# ---------------------------------------------------------------------------
# Self-Test:  python age_adversary.py

if __name__ == "__main__":
    torch.manual_seed(0)
    feat_dim, batch = 512, 8
    enc = nn.Sequential(nn.Linear(64, feat_dim), nn.ReLU())
    head = nn.Linear(feat_dim, 2)
    adv = AgeAdversary(feat_dim, mode="regress")
    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(head.parameters()) + list(adv.parameters()),
        lr=1e-3)

    x = torch.randn(batch, 64)
    y = torch.randint(0, 2, (batch,))
    age = torch.randint(60, 90, (batch,)).float()

    for ep in range(3):
        lam = lambda_schedule(ep, 12, 1.0)
        f = enc(x)
        loss_cls = nn.functional.cross_entropy(head(f), y)
        loss_adv = adv.loss(adv(f, lam), age)
        loss = loss_cls + loss_adv
        opt.zero_grad(); loss.backward(); opt.step()
        print("epoch {}  lambda {:.3f}  cls {:.4f}  adv {:.4f}".format(
            ep, lam, loss_cls.item(), loss_adv.item()))

    g = torch.randn(4, 8, requires_grad=True)
    out = grad_reverse(g, 2.0).sum()
    out.backward()
    assert torch.allclose(g.grad, -2.0 * torch.ones_like(g)), "reversal is wrong"
    print("\ngradient reversal verified: forward identity, backward -lambda")
