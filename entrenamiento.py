"""Entrenamiento de detectores con las anotaciones YOLO de FjordFish."""
import csv
import math
import random
import warnings
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def nombres_clases(raiz):
    with (Path(raiz) / "data.yaml").open(encoding="utf-8") as archivo:
        nombres = yaml.safe_load(archivo)["names"]
    if isinstance(nombres, dict):
        nombres = [nombres[i] for i in range(len(nombres))]
    return [str(nombre).split(": ", 1)[-1] for nombre in nombres]


def preparar_yaml_yolo(raiz, salida):
    raiz = Path(raiz).resolve()
    destino = Path(salida).resolve() / "fjordfish.yaml"
    destino.parent.mkdir(parents=True, exist_ok=True)
    config = {"path": str(raiz), "names": nombres_clases(raiz)}
    for split in ("train", "val"):
        dataset = FjordFish(raiz, split)
        lista = destino.parent / f"fjordfish_{split}.txt"
        lista.write_text("".join(str(p.resolve()) + "\n" for p in dataset.imagenes),
                         encoding="utf-8")
        config[split] = str(lista)
    destino.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return destino.resolve()


class FjordFish(Dataset):
    def __init__(self, raiz, split):
        self.raiz = Path(raiz)
        self.split = split
        self.num_clases = len(nombres_clases(raiz))
        carpeta = self.raiz / "images" / split
        self.imagenes = sorted(p for p in carpeta.iterdir()
                               if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        sin_etiqueta = [p for p in self.imagenes
                        if not (self.raiz / "labels" / split / (p.stem + ".txt")).is_file()]
        if sin_etiqueta:
            warnings.warn(f"{split}: se excluyen {len(sin_etiqueta)} imagenes sin etiqueta: "
                          + ", ".join(p.name for p in sin_etiqueta[:5]), stacklevel=2)
            excluidas = set(sin_etiqueta)
            self.imagenes = [p for p in self.imagenes if p not in excluidas]
        if not self.imagenes:
            raise ValueError(f"No hay imagenes con etiquetas en {carpeta}")

    def __len__(self):
        return len(self.imagenes)

    def __getitem__(self, indice):
        ruta = self.imagenes[indice]
        imagen = read_image(str(ruta), mode=ImageReadMode.RGB).float() / 255
        alto, ancho = imagen.shape[-2:]
        etiqueta = self.raiz / "labels" / self.split / (ruta.stem + ".txt")
        # Una etiqueta ausente puede ocultar un error de ruta; las imagenes
        # negativas deben tener un .txt vacio explicito.
        cajas, clases = [], []
        for numero, linea in enumerate(etiqueta.read_text(encoding="utf-8").splitlines(), 1):
            if not linea.strip():
                continue
            valores = list(map(float, linea.split()))
            if len(valores) != 5:
                raise ValueError(f"{etiqueta}:{numero}: se esperan clase cx cy w h")
            clase, cx, cy, w, h = valores
            if not all(math.isfinite(v) for v in valores):
                raise ValueError(f"{etiqueta}:{numero}: valores no finitos")
            if not clase.is_integer() or not 0 <= clase < self.num_clases:
                raise ValueError(f"{etiqueta}:{numero}: clase fuera de rango")
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError(f"{etiqueta}:{numero}: coordenadas YOLO invalidas")
            x1, y1 = max(0, (cx-w/2)*ancho), max(0, (cy-h/2)*alto)
            x2, y2 = min(ancho, (cx+w/2)*ancho), min(alto, (cy+h/2)*alto)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"{etiqueta}:{numero}: caja sin area")
            cajas.append([x1, y1, x2, y2])
            clases.append(int(clase) + 1)  # Faster R-CNN reserva 0 para fondo.
        cajas = torch.tensor(cajas, dtype=torch.float32).reshape(-1, 4)
        target = {"boxes": cajas, "labels": torch.tensor(clases, dtype=torch.int64),
                  "image_id": torch.tensor(indice),
                  "area": (cajas[:, 2]-cajas[:, 0]) * (cajas[:, 3]-cajas[:, 1]),
                  "iscrowd": torch.zeros(len(clases), dtype=torch.int64)}
        return imagen, target


def crear_faster(num_clases, tamano=640, preentrenado=True):
    pesos = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if preentrenado else None
    modelo = fasterrcnn_resnet50_fpn_v2(
        weights=pesos, weights_backbone=None, min_size=tamano, max_size=tamano,
    )
    entradas = modelo.roi_heads.box_predictor.cls_score.in_features
    modelo.roi_heads.box_predictor = FastRCNNPredictor(entradas, num_clases + 1)
    return modelo


def agrupar(batch):
    return tuple(zip(*batch))


def perdida_epoca(modelo, loader, dispositivo, optimizador=None):
    # Torchvision devuelve perdidas solo en modo train. Para validacion,
    # no se calculan gradientes y se congelan las estadisticas de BatchNorm.
    modelo.train()
    for modulo in modelo.modules():
        if isinstance(modulo, torch.nn.modules.batchnorm._BatchNorm):
            modulo.eval()
    total, cantidad = 0.0, 0
    with torch.set_grad_enabled(optimizador is not None):
        for paso, (imagenes, targets) in enumerate(loader, 1):
            imagenes = [imagen.to(dispositivo) for imagen in imagenes]
            targets = [{k: v.to(dispositivo) for k, v in t.items()} for t in targets]
            perdida = sum(modelo(imagenes, targets).values())
            if not torch.isfinite(perdida):
                raise RuntimeError(f"Perdida no finita en paso {paso}")
            if optimizador is not None:
                optimizador.zero_grad(set_to_none=True)
                perdida.backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 10.0)
                optimizador.step()
            total += perdida.item() * len(imagenes)
            cantidad += len(imagenes)
            if paso == 1 or paso % 50 == 0:
                print(f"  Paso {paso}/{len(loader)} | loss={perdida.item():.4f}", flush=True)
    return total / cantidad


def entrenar_faster(raiz, salida="resultados/faster_entrenamiento", epocas=10,
                     batch=2, lr=0.005, paciencia=3, tamano=640, semilla=42):
    if min(epocas, batch, paciencia, tamano) < 1 or lr <= 0:
        raise ValueError("Los parametros de entrenamiento deben ser positivos")
    random.seed(semilla)
    torch.manual_seed(semilla)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nombres = ["fondo"] + nombres_clases(raiz)
    train = DataLoader(FjordFish(raiz, "train"), batch_size=batch, shuffle=True,
                       num_workers=0, collate_fn=agrupar)
    val = DataLoader(FjordFish(raiz, "val"), batch_size=batch, shuffle=False,
                     num_workers=0, collate_fn=agrupar)
    modelo = crear_faster(len(nombres)-1, tamano).to(dispositivo)
    optimizador = torch.optim.SGD([p for p in modelo.parameters() if p.requires_grad],
                                  lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizador, step_size=3, gamma=0.1)
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    mejor, sin_mejora, historial = float("inf"), 0, []
    print(f"Dispositivo: {dispositivo}; train={len(train.dataset)}; val={len(val.dataset)}")
    for epoca in range(1, epocas+1):
        print(f"Epoca {epoca}/{epocas}", flush=True)
        loss_train = perdida_epoca(modelo, train, dispositivo, optimizador)
        # El muestreo de propuestas de validacion usa la misma semilla cada epoca,
        # sin cambiar la secuencia aleatoria del entrenamiento.
        dispositivos_rng = [torch.cuda.current_device()] if dispositivo.type == "cuda" else []
        with torch.random.fork_rng(devices=dispositivos_rng):
            torch.manual_seed(semilla)
            loss_val = perdida_epoca(modelo, val, dispositivo)
        mejora = loss_val < mejor
        mejor = min(mejor, loss_val)
        sin_mejora = 0 if mejora else sin_mejora + 1
        historial.append({"epoca": epoca, "train_loss": loss_train, "val_loss": loss_val,
                          "lr": optimizador.param_groups[0]["lr"]})
        scheduler.step()
        checkpoint = {"model": modelo.state_dict(), "optimizer": optimizador.state_dict(),
                      "scheduler": scheduler.state_dict(), "epoca": epoca, "clases": nombres,
                      "mejor_val_loss": mejor, "tamano": tamano}
        torch.save(checkpoint, salida / "last.pt")
        if mejora:
            torch.save(checkpoint, salida / "best.pt")
        with (salida / "historial.csv").open("w", newline="", encoding="utf-8") as archivo:
            writer = csv.DictWriter(archivo, fieldnames=list(historial[0]))
            writer.writeheader()
            writer.writerows(historial)
        print(f"train_loss={loss_train:.4f} | val_loss={loss_val:.4f} | mejor={mejor:.4f}")
        if sin_mejora >= paciencia:
            print("Parada temprana: la perdida de validacion dejo de mejorar.")
            break
    modelo.load_state_dict(torch.load(salida / "best.pt", map_location=dispositivo,
                                      weights_only=True)["model"])
    modelo.eval()
    return modelo, nombres, historial
