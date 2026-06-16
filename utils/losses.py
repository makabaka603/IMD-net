import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.eps ** 2))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()

        self.charb = CharbonnierLoss(eps)

        kernel = torch.tensor(
            [
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0],
            ],
            dtype=torch.float32,
        )

        self.register_buffer("kernel", kernel.view(1, 1, 3, 3))

    def laplacian(self, x):
        k = self.kernel.to(device=x.device, dtype=x.dtype)
        k = k.repeat(x.shape[1], 1, 1, 1)

        y = F.conv2d(x, k, padding=1, groups=x.shape[1])
        return y

    def forward(self, pred, target):
        return self.charb(self.laplacian(pred), self.laplacian(target))


class FFTLoss(nn.Module):
    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
        target_fft = torch.fft.rfft2(target.float(), norm="ortho")

        loss = torch.mean(torch.abs(pred_fft - target_fft))
        return loss


class DecouplingLoss(nn.Module):
    def forward(self, cf_list, di_list):
        if len(cf_list) == 0 or len(di_list) == 0:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return torch.tensor(0.0, device=device)

        total = cf_list[0].new_tensor(0.0)
        count = 0

        for cf, di in zip(cf_list, di_list):
            cfv = cf.flatten(2)
            div = di.flatten(2)

            cos = F.cosine_similarity(cfv, div, dim=1, eps=1e-8)
            total = total + (cos ** 2).mean()
            count += 1

        return total / max(count, 1)


class IMDNetLoss(nn.Module):
    """
    Paper loss:
        L = Lc + delta * Le + lambda * Lf + gamma * Ld

    Side outputs weighted with exponential decay:
        main output: weight 1.0
        side outputs: weight 0.25 each
    """

    def __init__(
        self,
        lambda_fft=0.1,
        delta_edge=0.05,
        gamma_decouple=0.001,
        side_weight=0.25,
    ):
        super().__init__()

        self.charb = CharbonnierLoss()
        self.edge = EdgeLoss()
        self.fft = FFTLoss()
        self.decouple = DecouplingLoss()

        self.lambda_fft = float(lambda_fft)
        self.delta_edge = float(delta_edge)
        self.gamma_decouple = float(gamma_decouple)
        self.side_weight = float(side_weight)

    def _resize_target(self, target, pred):
        if pred.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(
                target,
                size=pred.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return target

    def _single_image_loss(self, pred, target):
        c = self.charb(pred, target)
        e = self.edge(pred, target)

        if self.lambda_fft > 0:
            f = self.fft(pred, target)
        else:
            f = pred.new_tensor(0.0)

        total = c + self.delta_edge * e + self.lambda_fft * f

        return total, c, e, f

    def forward(self, outputs, target):
        if isinstance(outputs, dict):
            preds = [outputs["out"]]
            preds += outputs.get("side_outputs", [])
            cf_list = outputs.get("cf_list", [])
            di_list = outputs.get("di_list", [])
        else:
            preds = [outputs]
            cf_list = []
            di_list = []

        total = target.new_tensor(0.0)

        c_log = 0.0
        e_log = 0.0
        f_log = 0.0
        loss_count = 0

        # Main output: full weight
        target_main = self._resize_target(target, preds[0])
        loss_i, c, e, f = self._single_image_loss(preds[0], target_main)
        total = total + loss_i
        c_log += float(c.detach().cpu())
        e_log += float(e.detach().cpu())
        f_log += float(f.detach().cpu())
        loss_count += 1

        # Side outputs: reduced weight
        for i, pred in enumerate(preds[1:]):
            w = self.side_weight  # 0.25 per side output
            target_i = self._resize_target(target, pred)
            loss_i, c, e, f = self._single_image_loss(pred, target_i)
            total = total + w * loss_i
            c_log += float(c.detach().cpu()) * w
            e_log += float(e.detach().cpu()) * w
            f_log += float(f.detach().cpu()) * w
            loss_count += 1

        # Decoupling loss
        if self.gamma_decouple > 0 and len(cf_list) > 0 and len(di_list) > 0:
            d = self.decouple(cf_list, di_list)
            total = total + self.gamma_decouple * d
        else:
            d = target.new_tensor(0.0)

        denom = max(loss_count, 1)

        logs = {
            "loss": float(total.detach().cpu()),
            "charb": c_log / denom,
            "edge": e_log / denom,
            "fft": f_log / denom,
            "decouple": float(d.detach().cpu()),
        }

        return total, logs
