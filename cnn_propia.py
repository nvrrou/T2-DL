"""Detector CNN educativo desde cero. Etiquetas YOLO, IDs originales desde cero.

No requiere torchvision ni descarga pesos. Las cajas internas son xyxy [0, 1].
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import time
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


@dataclass
class Config:
    image_size: int = 416
    slots: int = 3
    batch_size: int = 8
    epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    box_weight: float = 5.0
    class_weight: float = 1.0
    background_weight: float = 0.5
    seed: int = 42
    workers: int = 0  # Compatible con Jupyter/Windows.
    missing_labels_are_background: bool = False


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_names(root):
    """Lee la lista names del YAML distribuido con FjordFish, sin PyYAML."""
    text = (Path(root) / 'data.yaml').read_text(encoding='utf-8')
    if 'names:' not in text:
        raise ValueError('Falta names en data.yaml')
    names = []
    for line in text.split('names:', 1)[1].splitlines():
        if line.strip().startswith('-'):
            name = line.strip()[1:].strip().strip("\"'")
            names.append(re.sub(r'^\d+:\s*', '', name))
        elif line.strip() and not line.startswith(' '):
            break
    if not names:
        raise ValueError('Se espera names como lista YAML de nombres de clases.')
    return names


def xywh_to_xyxy(boxes):
    return torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                      boxes[..., :2] + boxes[..., 2:] / 2), dim=-1)


def box_iou(a, b):
    inter = (torch.minimum(a[:, None, 2:], b[None, :, 2:]) -
             torch.maximum(a[:, None, :2], b[None, :, :2])).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-8)


def aligned_giou(a, b):
    inter = (torch.minimum(a[:, 2:], b[:, 2:]) -
             torch.maximum(a[:, :2], b[:, :2])).clamp(min=0).prod(-1)
    union = ((a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1) +
             (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1) - inter)
    enclosing = (torch.maximum(a[:, 2:], b[:, 2:]) -
                 torch.minimum(a[:, :2], b[:, :2])).clamp(min=0).prod(-1)
    return inter / union.clamp(min=1e-8) - (enclosing - union) / enclosing.clamp(min=1e-8)


class FishDataset(Dataset):
    def __init__(self, root, split, config, augment=False):
        self.root, self.config, self.augment = Path(root), config, augment
        self.names = load_names(root)
        folder = self.root / 'images' / split
        self.paths = sorted(p for p in folder.glob('*')
                            if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        if not self.paths:
            raise ValueError(f'No hay imágenes en {folder}')
        self.targets = []
        self.counts = Counter()
        self.missing_labels = []
        for path in self.paths:
            label = self.root / 'labels' / split / f'{path.stem}.txt'
            if not label.exists():
                if not config.missing_labels_are_background:
                    raise FileNotFoundError(f'Falta {label}. Si verificaste que es una imagen '
                                            'sin objetos, usa missing_labels_are_background=True.')
                self.missing_labels.append(str(path))
            rows = []
            for line in (label.read_text().splitlines() if label.exists() else []):
                if not line.strip():
                    continue
                values = list(map(float, line.split()))
                if len(values) != 5:
                    raise ValueError(f'{label}: se espera class cx cy w h')
                cls, cx, cy, w, h = values
                if not all(math.isfinite(v) for v in values):
                    raise ValueError(f'{label}: etiqueta no finita')
                if cls != int(cls) or not 0 <= cls < len(self.names):
                    raise ValueError(f'{label}: ID inválido {cls}')
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError(f'{label}: coordenadas inválidas')
                rows.append(values)
                self.counts[int(cls)] += 1
            self.targets.append(torch.tensor(rows, dtype=torch.float32).reshape(-1, 5))
        if self.missing_labels:
            warnings.warn(f'Se interpretan como fondo imágenes sin .txt: {self.missing_labels}. '
                          'Verifica que realmente no tienen objetos anotables.', stacklevel=2)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert('RGB')
        rows = self.targets[index].clone()
        width, height = image.size
        size = self.config.image_size
        scale = min(size / width, size / height)
        new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
        left, top = (size - new_w) // 2, (size - new_h) // 2
        boxes = xywh_to_xyxy(rows[:, 1:]).clamp(0, 1)
        boxes *= torch.tensor([new_w, new_h, new_w, new_h])
        boxes += torch.tensor([left, top, left, top])
        boxes /= size
        if self.augment:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]
                # Tras voltear la imagen original, el margen puede diferir un píxel.
                left = size - new_w - left
        canvas = Image.new('RGB', (size, size), (114, 114, 114))
        canvas.paste(image.resize((new_w, new_h), Image.Resampling.BILINEAR), (left, top))
        tensor = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float() / 255
        return tensor, {'boxes': boxes, 'labels': rows[:, 0].long(),
                        'image_id': str(self.paths[index])}


def collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.GroupNorm(8, out_channels), nn.SiLU(inplace=True))


class FishCNN(nn.Module):
    """CNN de una escala, stride 16, con varios slots por celda."""
    def __init__(self, num_classes, slots=3):
        super().__init__()
        self.num_classes, self.slots = num_classes, slots
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, 2), ConvBlock(32, 64, 2), ConvBlock(64, 64),
            ConvBlock(64, 128, 2), ConvBlock(128, 128),
            ConvBlock(128, 256, 2), ConvBlock(256, 256))
        self.head = nn.Conv2d(256, slots * (5 + num_classes), 1)
        # Pocos objetos frente a miles de candidatos: prior de objeto = 1 %.
        with torch.no_grad():
            bias = self.head.bias.view(slots, 5 + num_classes)
            bias[:, :4] = 0
            bias[:, 2:4] = math.log(0.1 / 0.9)
            bias[:, 4] = math.log(0.01 / 0.99)

    def forward(self, images):
        if images.shape[-1] % 16 or images.shape[-2] % 16:
            raise ValueError('Alto y ancho deben ser múltiplos de 16.')
        raw = self.head(self.backbone(images))
        n, _, h, w = raw.shape
        return raw.view(n, self.slots, 5 + self.num_classes, h, w).permute(0, 3, 4, 1, 2)


def decode(raw):
    h, w = raw.shape[1:3]
    yy, xx = torch.meshgrid(torch.arange(h, device=raw.device),
                            torch.arange(w, device=raw.device), indexing='ij')
    grid = torch.stack((xx, yy), -1)[None, :, :, None, :]
    center = (raw[..., :2].sigmoid() + grid) / raw.new_tensor([w, h])
    return xywh_to_xyxy(torch.cat((center, raw[..., 2:4].sigmoid()), -1))


class DetectionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, raw, targets):
        boxes = decode(raw)
        n, h, w, slots, _ = raw.shape
        positive = torch.zeros(raw.shape[:-1], dtype=torch.bool, device=raw.device)
        predicted_boxes, true_boxes, logits, classes = [], [], [], []
        for b in range(n):
            tb = targets[b]['boxes'].to(raw.device)
            tl = targets[b]['labels'].to(raw.device)
            cells = (((tb[:, :2] + tb[:, 2:]) / 2) * tb.new_tensor([w, h])).long()
            cells[:, 0].clamp_(0, w - 1)
            cells[:, 1].clamp_(0, h - 1)
            for cell in cells.unique(dim=0):
                x, y = cell.tolist()
                ids = ((cells == cell).all(1)).nonzero().flatten()
                if len(ids) > slots:
                    raise ValueError(f'{targets[b]["image_id"]}: {len(ids)} objetos en una '
                                     f'celda, pero hay {slots} slots. Aumenta slots o resolución.')
                # Asignación uno a uno óptima dentro de la celda (como máximo 3! opciones).
                with torch.no_grad():
                    cost = (self.config.box_weight * torch.cdist(
                        boxes[b, y, x].detach(), tb[ids], p=1) -
                        raw[b, y, x, :, 5:].detach().log_softmax(-1)[:, tl[ids]])
                    choices = list(itertools.permutations(range(slots), len(ids)))
                    candidate = torch.tensor(choices, device=raw.device)
                    costs = cost[candidate, torch.arange(len(ids), device=raw.device)].sum(1)
                    selected = candidate[costs.argmin()]
                positive[b, y, x, selected] = True
                predicted_boxes.append(boxes[b, y, x, selected])
                true_boxes.append(tb[ids])
                logits.append(raw[b, y, x, selected, 5:])
                classes.append(tl[ids])
        zero = raw.sum() * 0
        pos_loss = (F.binary_cross_entropy_with_logits(raw[..., 4][positive],
                    torch.ones_like(raw[..., 4][positive])) if positive.any() else zero)
        neg_loss = (F.binary_cross_entropy_with_logits(raw[..., 4][~positive],
                    torch.zeros_like(raw[..., 4][~positive])) if (~positive).any() else zero)
        box_loss, cls_loss = zero, zero
        if predicted_boxes:
            pb, tb = torch.cat(predicted_boxes), torch.cat(true_boxes)
            box_loss = (1 - aligned_giou(pb, tb)).mean() + F.smooth_l1_loss(pb, tb)
            cls_loss = F.cross_entropy(torch.cat(logits), torch.cat(classes))
        total = (self.config.box_weight * box_loss + self.config.class_weight * cls_loss +
                 pos_loss + self.config.background_weight * neg_loss)
        return {'total': total, 'box': box_loss, 'class': cls_loss,
                'object': pos_loss, 'background': neg_loss}


def nms(boxes, scores, threshold):
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        order = order[1:][box_iou(boxes[i:i + 1], boxes[order[1:]])[0] <= threshold]
    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long, device=boxes.device)


@torch.no_grad()
def predict(model, images, score_threshold=0.01, nms_iou=0.5, max_detections=100):
    model.eval()
    raw = model(images)
    boxes = decode(raw).clamp(0, 1).flatten(1, 3)
    class_probs, labels = raw[..., 5:].softmax(-1).max(-1)
    scores = (class_probs * raw[..., 4].sigmoid()).flatten(1)
    labels = labels.flatten(1)
    output = []
    for b in range(len(images)):
        valid = (scores[b] >= score_threshold) & ((boxes[b, :, 2:] > boxes[b, :, :2]).all(-1))
        bb, ss, ll = boxes[b, valid], scores[b, valid], labels[b, valid]
        keep = []
        for cls in ll.unique():
            indices = (ll == cls).nonzero().flatten()
            keep.append(indices[nms(bb[indices], ss[indices], nms_iou)])
        selected = torch.cat(keep) if keep else torch.empty(0, dtype=torch.long, device=images.device)
        selected = selected[ss[selected].argsort(descending=True)[:max_detections]]
        output.append({'boxes': bb[selected].cpu(), 'scores': ss[selected].cpu(),
                       'labels': ll[selected].cpu()})
    return output


def detection_metrics(predictions, targets, num_classes, operating_threshold=0.25):
    """AP de 101 puntos a IoU .50:.05:.95, sin reglas COCO area/ignore/crowd.

    AP macro sobre clases con GT. Clases sin GT = None. PR micro a IoU .50.
    Una detección sólo puede emparejarse con un GT, por imagen y clase.
    """
    per_class = []
    tp_total = fp_total = gt_total = 0
    thresholds = [0.5 + 0.05 * i for i in range(10)]
    for cls in range(num_classes):
        gt = [t['boxes'][t['labels'] == cls].cpu() for t in targets]
        support = sum(len(b) for b in gt)
        rows = [(float(p['scores'][j]), i, p['boxes'][j].cpu())
                for i, p in enumerate(predictions)
                for j in (p['labels'] == cls).nonzero().flatten().tolist()]
        rows.sort(key=lambda row: row[0], reverse=True)
        aps = []
        for k, threshold in enumerate(thresholds):
            used = [torch.zeros(len(b), dtype=torch.bool) for b in gt]
            tp, fp = [], []
            for score, image_id, box in rows:
                match = False
                if len(gt[image_id]):
                    overlaps = box_iou(box[None], gt[image_id])[0]
                    overlaps[used[image_id]] = -1
                    value, index = overlaps.max(0)
                    if value >= threshold:
                        used[image_id][index] = True
                        match = True
                tp.append(float(match))
                fp.append(float(not match))
                if k == 0 and score >= operating_threshold:
                    tp_total += int(match)
                    fp_total += int(not match)
            if support:
                if rows:
                    cumulative_tp = torch.tensor(tp).cumsum(0)
                    precision = cumulative_tp / (cumulative_tp + torch.tensor(fp).cumsum(0))
                    recall = cumulative_tp / support
                    interpolated = [precision[recall >= r].max().item()
                                    if (recall >= r).any() else 0.
                                    for r in torch.linspace(0, 1, 101)]
                    aps.append(sum(interpolated) / 101)
                else:
                    aps.append(0.)
        gt_total += support
        per_class.append({'class_id': cls, 'support': support,
                          'ap50': aps[0] if aps else None,
                          'ap50_95': sum(aps) / len(aps) if aps else None})
    evaluated = [c for c in per_class if c['support']]
    return {'map50': sum(c['ap50'] for c in evaluated) / max(len(evaluated), 1),
            'map50_95': sum(c['ap50_95'] for c in evaluated) / max(len(evaluated), 1),
            'precision': tp_total / max(tp_total + fp_total, 1),
            'recall': tp_total / max(gt_total, 1), 'per_class': per_class,
            'score_threshold_pr': operating_threshold, 'iou_pr': 0.5}


@torch.no_grad()
def evaluate(model, loader, device):
    predictions, targets = [], []
    for images, batch_targets in loader:
        predictions.extend(predict(model, images.to(device)))
        targets.extend(batch_targets)
    return detection_metrics(predictions, targets, model.num_classes)


def train(root='data/FjordFish', output='runs/cnn_propia', config=None):
    config = config or Config()
    if config.image_size < 32 or config.image_size % 16:
        raise ValueError('image_size debe ser múltiplo de 16 y >= 32.')
    if config.slots < 1 or config.batch_size < 1 or config.epochs < 1 or config.patience < 1:
        raise ValueError('slots, batch_size, epochs y patience deben ser positivos.')
    seed_everything(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = FishDataset(root, 'train', config, augment=True)
    val_data = FishDataset(root, 'val', config)
    if {p.name for p in train_data.paths} & {p.name for p in val_data.paths}:
        raise ValueError('train y val contienen nombres de imagen repetidos; revisar particiones.')
    loader_kwargs = dict(batch_size=config.batch_size, num_workers=config.workers, collate_fn=collate)
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    model = FishCNN(len(train_data.names), config.slots).to(device)
    criterion = DetectionLoss(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate / 100)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # No sobrescribir experimentos.
    (output / 'config.json').write_text(json.dumps(asdict(config), indent=2))
    (output / 'splits.json').write_text(json.dumps({
        'train': [str(p) for p in train_data.paths], 'val': [str(p) for p in val_data.paths]
    }, indent=2))
    print(f'{device} | {len(train_data)} train, {len(val_data)} val | '
          f'{len(train_data.names)} clases | {sum(p.numel() for p in model.parameters()):,} parámetros', flush=True)
    best, stale, history = -1., 0, []
    if device.type == 'cuda':
        torch.cuda.synchronize()
    started = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        totals = Counter()
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            losses = criterion(model(images.to(device)), targets)
            if not torch.isfinite(losses['total']):
                raise RuntimeError('Loss no finita: revisar datos y learning rate.')
            losses['total'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            for key, value in losses.items():
                totals[key] += value.detach().item() * len(images)
        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for images, targets in val_loader:
                val_loss += criterion(model(images.to(device)), targets)['total'].item() * len(images)
        metrics = evaluate(model, val_loader, device)
        row = {'epoch': epoch + 1, 'lr': optimizer.param_groups[0]['lr'],
               'train_loss': totals['total'] / len(train_data),
               'val_loss': val_loss / len(val_data),
               **{f'train_{k}': v / len(train_data) for k, v in totals.items() if k != 'total'},
               **{k: v for k, v in metrics.items() if k != 'per_class'}}
        history.append(row)
        print(f"Época {epoch + 1}/{config.epochs}: loss={row['train_loss']:.4f} "
              f"val={row['val_loss']:.4f} mAP50:95={metrics['map50_95']:.4f}", flush=True)
        if metrics['map50_95'] > best:
            best, stale = metrics['map50_95'], 0
            torch.save({'model': model.state_dict(), 'config': asdict(config),
                        'names': train_data.names, 'epoch': epoch + 1}, output / 'best.pt')
            (output / 'best_val_metrics.json').write_text(json.dumps(metrics, indent=2))
        else:
            stale += 1
        (output / 'history.json').write_text(json.dumps(history, indent=2))
        scheduler.step()
        if stale >= config.patience:
            break
    if device.type == 'cuda':
        torch.cuda.synchronize()
    (output / 'runtime.json').write_text(json.dumps({
        'training_and_validation_seconds': time.perf_counter() - started,
        'device': str(device), 'torch': str(torch.__version__),
        'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
        'epochs_completed': len(history)}, indent=2))
    checkpoint = torch.load(output / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, history


def load_detector(checkpoint, device='cpu'):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    config = Config(**saved['config'])
    model = FishCNN(len(saved['names']), config.slots).to(device)
    model.load_state_dict(saved['model'])
    return model.eval(), config, saved['names']


def draw_prediction(image, prediction, names, threshold=0.25):
    """Dibuja sobre el tensor letterbox; cajas de predict están normalizadas."""
    canvas = Image.fromarray((image.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype('uint8'))
    draw = ImageDraw.Draw(canvas)
    for box, score, cls in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
        if score < threshold:
            continue
        pixels = box * box.new_tensor([canvas.width, canvas.height, canvas.width, canvas.height])
        draw.rectangle(pixels.tolist(), outline='red', width=2)
        draw.text((float(pixels[0]), float(pixels[1])), f'{names[int(cls)]} {score:.2f}', fill='yellow')
    return canvas


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/FjordFish')
    parser.add_argument('--output', default='runs/cnn_propia')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--missing-labels-are-background', action='store_true',
                        help='Sólo si comprobaste que las imágenes sin .txt son negativas.')
    args = parser.parse_args()
    train(args.root, args.output, Config(epochs=args.epochs, batch_size=args.batch_size,
          missing_labels_are_background=args.missing_labels_are_background))
