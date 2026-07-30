"""Fungsi objektif vision: bangun & latih CNN sesuai konfigurasi, nilai akurasinya.

Dua keluarga arsitektur dalam satu ruang pencarian:
  - "tiny"  : CNN yang dirakit dari konfigurasi (depth, width, block, use_se)
              — inilah bagian arsitektur yang benar-benar di-mix-and-match.
  - lainnya : nama model torchvision (resnet18, mobilenet_v3_small, ...)
              sebagai pembanding; parameter arsitektur tiny diabaikan untuknya.
"""

import time


def _build_tiny(nn, config, num_classes):
    width = int(config.get("width", 32))
    depth = int(config.get("depth", 3))
    block_type = config.get("block", "basic")
    use_se = bool(config.get("use_se", False))

    class SE(nn.Module):
        """Squeeze-and-Excitation: bobot ulang tiap channel (Hu dkk., 2018)."""

        def __init__(self, ch, r=8):
            super().__init__()
            hidden = max(ch // r, 4)
            self.fc = nn.Sequential(nn.Linear(ch, hidden), nn.ReLU(),
                                    nn.Linear(hidden, ch), nn.Sigmoid())

        def forward(self, x):
            w = self.fc(x.mean(dim=(2, 3))).unsqueeze(-1).unsqueeze(-1)
            return x * w

    class Residual(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout))
            self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
            self.act = nn.ReLU()

        def forward(self, x):
            return self.act(self.body(x) + self.skip(x))

    def make_block(cin, cout):
        if block_type == "residual":
            blk = Residual(cin, cout)
        elif block_type == "depthwise":
            blk = nn.Sequential(
                nn.Conv2d(cin, cin, 3, padding=1, groups=cin),
                nn.BatchNorm2d(cin), nn.ReLU(),
                nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout), nn.ReLU())
        else:  # basic
            blk = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1),
                                nn.BatchNorm2d(cout), nn.ReLU())
        return nn.Sequential(blk, SE(cout)) if use_se else blk

    layers = [nn.Conv2d(3, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU()]
    ch = width
    for i in range(depth):
        out = min(ch * 2, 256) if i > 0 else ch
        layers += [make_block(ch, out), nn.MaxPool2d(2)]
        ch = out
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch, num_classes)]
    return nn.Sequential(*layers)


def _build_torchvision(models, nn, arch, num_classes):
    model = models.get_model(arch, weights=None)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        cls = model.classifier
        if isinstance(cls, nn.Linear):
            model.classifier = nn.Linear(cls.in_features, num_classes)
        else:
            for i in range(len(cls) - 1, -1, -1):
                if isinstance(cls[i], nn.Linear):
                    cls[i] = nn.Linear(cls[i].in_features, num_classes)
                    break
    elif hasattr(model, "heads"):  # keluarga ViT
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


class VisionBackend:
    def __init__(self, cfg):
        self.cfg = cfg
        d = cfg.get("dataset") or {}
        self.name = d.get("name", "cifar10")
        self.dir = d.get("dir", "./data")
        self.image_size = int(d.get("image_size", 32))
        self.limit_train = d.get("limit_train", 2000)
        self.limit_val = d.get("limit_val", 1000)

    def _datasets(self, transforms, datasets, augment):
        s = self.image_size
        train_tf = [transforms.Resize((s, s))]
        if augment == "basic":
            train_tf += [transforms.RandomHorizontalFlip(),
                         transforms.RandomCrop(s, padding=4)]
        train_tf += [transforms.ToTensor()]
        val_tf = [transforms.Resize((s, s)), transforms.ToTensor()]

        if self.name == "cifar10":
            train = datasets.CIFAR10(self.dir, train=True, download=True,
                                     transform=transforms.Compose(train_tf))
            val = datasets.CIFAR10(self.dir, train=False, download=True,
                                   transform=transforms.Compose(val_tf))
            return train, val, 10

        # folder gambar sendiri: <dir>/train/<kelas>/*.jpg dan <dir>/val/<kelas>/*.jpg
        train = datasets.ImageFolder(f"{self.dir}/train",
                                     transform=transforms.Compose(train_tf))
        val = datasets.ImageFolder(f"{self.dir}/val",
                                   transform=transforms.Compose(val_tf))
        return train, val, len(train.classes)

    def run(self, config, fidelity=None):
        # fidelity = jumlah epoch (untuk successive halving); None = pakai cfg
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, Subset
            from torchvision import datasets, models, transforms
        except ImportError:
            return {"score": 0.0,
                    "error": "torch/torchvision belum terpasang — lihat requirements.txt"}

        arch = config.get("arch", "tiny")
        lr = float(config.get("lr", 1e-3))
        opt_name = str(config.get("optimizer", "adam")).lower()
        batch_size = int(config.get("batch_size", 64))
        epochs = int(fidelity) if fidelity else int(self.cfg.get("epochs", 1))
        augment = config.get("augment", "none")
        seed = int(self.cfg.get("seed", 42))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(seed)
        t0 = time.time()

        try:
            train_ds, val_ds, num_classes = self._datasets(transforms, datasets, augment)
            gen = torch.Generator().manual_seed(seed)
            if self.limit_train:
                idx = torch.randperm(len(train_ds), generator=gen)[: int(self.limit_train)]
                train_ds = Subset(train_ds, idx.tolist())
            if self.limit_val:
                idx = torch.randperm(len(val_ds), generator=gen)[: int(self.limit_val)]
                val_ds = Subset(val_ds, idx.tolist())

            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_dl = DataLoader(val_ds, batch_size=batch_size)

            if arch == "tiny":
                model = _build_tiny(nn, config, num_classes)
            else:
                model = _build_torchvision(models, nn, arch, num_classes)
            model = model.to(device)

            loss_fn = nn.CrossEntropyLoss()
            if opt_name == "sgd":
                opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            else:
                opt = torch.optim.Adam(model.parameters(), lr=lr)

            model.train()
            for _ in range(epochs):
                for x, y in train_dl:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    loss_fn(model(x), y).backward()
                    opt.step()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in val_dl:
                    x, y = x.to(device), y.to(device)
                    correct += (model(x).argmax(dim=1) == y).sum().item()
                    total += y.numel()
            acc = correct / max(total, 1)
        except Exception as e:  # konfigurasi tak valid, OOM, dsb. — catat, lanjutkan sweep
            return {"score": 0.0, "error": f"{type(e).__name__}: {e}"}

        return {
            "score": round(acc, 4),
            "metrics": {
                "val_accuracy": round(acc, 4),
                "params_juta": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                "durasi_detik": round(time.time() - t0, 1),
                "device": device,
            },
        }
