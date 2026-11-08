import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from torchvision.io import write_png

from entrenamiento import FjordFish, preparar_yaml_yolo


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporal.cleanup)
        self.raiz = Path(self.temporal.name)
        (self.raiz / "images/train").mkdir(parents=True)
        (self.raiz / "labels/train").mkdir(parents=True)
        (self.raiz / "data.yaml").write_text("names: ['0: pez']", encoding="utf-8")
        write_png(torch.zeros((3, 80, 100), dtype=torch.uint8),
                  str(self.raiz / "images/train/pez.png"))
        self.etiqueta = self.raiz / "labels/train/pez.txt"

    def test_converts_yolo_to_pixels_and_reserves_background(self):
        self.etiqueta.write_text("0 0.5 0.5 0.2 0.5", encoding="utf-8")
        imagen, target = FjordFish(self.raiz, "train")[0]
        self.assertEqual(tuple(imagen.shape), (3, 80, 100))
        torch.testing.assert_close(target["boxes"], torch.tensor([[40., 20., 60., 60.]]))
        self.assertEqual(target["labels"].tolist(), [1])
        self.assertEqual(target["area"].tolist(), [800.])

    def test_empty_annotation_has_valid_tensor_shapes(self):
        self.etiqueta.write_text("", encoding="utf-8")
        _, target = FjordFish(self.raiz, "train")[0]
        self.assertEqual(tuple(target["boxes"].shape), (0, 4))
        self.assertEqual(tuple(target["labels"].shape), (0,))

    def test_missing_annotation_is_not_silently_negative(self):
        with self.assertWarnsRegex(UserWarning, "sin etiqueta"):
            with self.assertRaisesRegex(ValueError, "No hay imagenes con etiquetas"):
                FjordFish(self.raiz, "train")

    def test_invalid_class_is_rejected(self):
        self.etiqueta.write_text("1 0.5 0.5 0.2 0.5", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "clase fuera de rango"):
            FjordFish(self.raiz, "train")[0]

    def test_yolo_yaml_uses_local_paths_and_clean_names(self):
        self.etiqueta.write_text("", encoding="utf-8")
        (self.raiz / "images/val").mkdir()
        (self.raiz / "labels/val").mkdir()
        write_png(torch.zeros((3, 80, 100), dtype=torch.uint8),
                  str(self.raiz / "images/val/pez.png"))
        (self.raiz / "labels/val/pez.txt").write_text("", encoding="utf-8")
        destino = preparar_yaml_yolo(self.raiz, self.raiz / "salida")
        config = yaml.safe_load(destino.read_text(encoding="utf-8"))
        self.assertEqual(config["path"], str(self.raiz.resolve()))
        self.assertEqual(config["names"], ["pez"])
        for split in ("train", "val"):
            lista = Path(config[split]).read_text(encoding="utf-8").splitlines()
            self.assertEqual(lista, [str(self.raiz / "images" / split / "pez.png")])


if __name__ == "__main__":
    unittest.main()
