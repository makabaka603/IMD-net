import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class EdgeLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.charb = CharbonnierLoss(eps)
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer('kernel', kernel.view(1, 1, 3, 3))

    def laplacian(self, x):
        k = self.kernel.repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(x, k, padding=1, groups=x.shape[1])

    def forward(self, pred, target):
        return self.charb(self.laplacian(pred), self.laplacian(target))


class FFTLoss(nn.Module):
    def forward(self, pred, target):
        pf = torch.fft.rfft2(pred, norm='ortho')
        tf = torch.fft.rfft2(target, norm='ortho')
        return torch.mean(torch.abs(pf - tf))


class DecouplingLoss(nn.Module):
    """Cosine similarity penalty between CF and DI.

    Uses cos^2 to produce a non-negative loss that strongly penalises
    large overlap between clean and degradation features.
    """
    def forward(self, cf_list, di_list):
        if not cf_list or not di_list:
            return torch.tensor(0.0, device=cf_list[0].device if cf_list else 'cpu')
        loss = 0.0
        for cf, di in zip(cf_list, di_list):
            cfv = cf.flatten(2)
            div = di.flatten(2)
            cos = F.cosine_similarity(cfv, div, dim=1)
            loss = loss + (cos ** 2).mean()  # cos²: always non-negative
        return loss / len(cf_list)


class IMDNetLoss(nn.Module):
    def __init__(self, lambda_fft=0.1, delta_edge=0.05, gamma_decouple=0.001):
        super().__init__()
        self.charb = CharbonnierLoss()
        self.edge = EdgeLoss()
        self.fft = FFTLoss()
        self.decouple = DecouplingLoss()
        self.lambda_fft = lambda_fft
        self.delta_edge = delta_edge
        self.gamma_decouple = gamma_decouple

    def forward(self, outputs, target):
        pred = outputs['out'] if isinstance(outputs, dict) else outputs
        preds = [pred]
        if isinstance(outputs, dict) and 'side_outputs' in outputs:
            preds += outputs['side_outputs']
        total = 0.0
        logs = {}
        c_total = e_total = f_total = 0.0
        for p in preds:
            # Multi-scale supervision: no interpolation needed
            # side_outputs are now at their native scales
            t = F.interpolate(target, size=p.shape[-2:], mode='bilinear', align_corners=False)
            c = self.charb(p, t)
            e = self.edge(p, t)
            ff = self.fft(p, t)
            total = total + c + self.delta_edge * e + self.lambda_fft * ff
            c_total += c.detach()
            e_total += e.detach()
            f_total += ff.detach()
        d = torch.tensor(0.0, device=target.device)
        if isinstance(outputs, dict):
            d = self.decouple(outputs.get('cf_list', []), outputs.get('di_list', []))
            total = total + self.gamma_decouple * d
        logs['charb'] = float(c_total / len(preds))
        logs['edge'] = float(e_total / len(preds))
        logs['fft'] = float(f_total / len(preds))
        logs['decouple'] = float(d.detach())
        return total, logs
