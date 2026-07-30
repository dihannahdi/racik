"""Zero-cost proxies: menilai arsitektur TANPA training — detik, bukan epoch.

Dua proxy klasik, diimplementasikan dari nol:
  naswot  : log-determinan matriks kesamaan pola aktivasi ReLU pada satu batch
            (Mellor dkk., "Neural Architecture Search without Training", ICML 2021).
            Intuisi: arsitektur bagus memetakan input berbeda ke pola aktivasi
            berbeda; kalau semua input menyalakan neuron yang sama, jaringan
            itu sulit membedakan apa pun.
  synflow : jumlah |w * dL/dw| dengan bobot di-|abs| dan input satu-vektor
            (Tanaka dkk., NeurIPS 2020; dipakai sebagai proxy NAS oleh
            Abdelfattah dkk., ICLR 2021). Mengukur "aliran sinaptik" —
            seberapa hidup jalur gradien arsitektur itu.

Disiplin pakainya (lihat proxycheck di run.py): proxy TIDAK dipercaya buta —
ukur dulu korelasi peringkatnya terhadap akurasi hasil training kita sendiri.
"""

import math


def spearman(xs, ys):
    """Korelasi peringkat Spearman, tanpa scipy (rata-rata peringkat untuk seri)."""

    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (vx * vy + 1e-12)


def _build_model(config, num_classes):
    from torch import nn
    from torchvision import models

    from .vision_backend import _build_tiny, _build_torchvision

    arch = config.get("arch", "tiny")
    if arch == "tiny":
        return _build_tiny(nn, config, num_classes)
    return _build_torchvision(models, nn, arch, num_classes)


def naswot_score(model, x):
    import torch
    from torch import nn

    codes = []

    def hook(_m, _inp, out):
        codes.append((out.detach() > 0).flatten(1).float())

    handles = [m.register_forward_hook(hook)
               for m in model.modules() if isinstance(m, nn.ReLU)]
    model.eval()
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    if not codes:
        return float("-inf")

    c = torch.cat(codes, dim=1)               # [batch, fitur biner]
    k = (c @ c.T + (1 - c) @ (1 - c).T) / c.shape[1]
    k = k + 1e-4 * torch.eye(k.shape[0])      # jitter agar tidak singular
    _sign, logdet = torch.linalg.slogdet(k)
    return logdet.item()


def synflow_score(model, input_shape):
    import torch

    model = model.double().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.data = p.data.abs()             # linearisasi
    x = torch.ones(1, *input_shape, dtype=torch.double)
    model.zero_grad()
    model(x).sum().backward()
    s = sum((p * p.grad).abs().sum().item()
            for p in model.parameters() if p.grad is not None)
    return math.log(s + 1e-20)


class ProxyScorer:
    """Skorer siap pakai: siapkan satu batch data sekali, nilai banyak konfigurasi."""

    def __init__(self, cfg, proxy="naswot", batch_size=64):
        import torch
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms

        from .vision_backend import VisionBackend

        self.proxy = proxy
        self.seed = int(cfg.get("seed", 42))
        vb = VisionBackend(cfg)
        train, _val, self.num_classes = vb._datasets(transforms, datasets, "none")
        gen = torch.Generator().manual_seed(self.seed)
        dl = DataLoader(train, batch_size=batch_size, shuffle=True, generator=gen)
        self.batch, _ = next(iter(dl))
        self.input_shape = tuple(self.batch.shape[1:])

    def score(self, config):
        import torch

        torch.manual_seed(self.seed)          # inisialisasi bobot deterministik
        try:
            model = _build_model(config, self.num_classes)
            if self.proxy == "synflow":
                return synflow_score(model, self.input_shape)
            return naswot_score(model, self.batch)
        except Exception:
            return float("-inf")              # arsitektur tak valid = skor terburuk
