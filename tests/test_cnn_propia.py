import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from cnn_propia import (Config, DetectionLoss, FishCNN, FishDataset, box_iou,
                        detection_metrics, nms, predict)


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_iou_and_class_agnostic_nms_primitive(self):
        boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.], [.8, .8, 1., 1.]])
        self.assertAlmostEqual(box_iou(boxes[:1], boxes[:1]).item(), 1.)
        self.assertEqual(nms(boxes, torch.tensor([.9, .8, .7]), .5).tolist(), [0, 2])

    def test_backward_multiple_objects_same_cell_and_empty_image(self):
        model = FishCNN(3)
        images = torch.rand(2, 3, 64, 64)
        targets = [dict(boxes=torch.tensor([[.1, .1, .2, .2], [.11, .11, .21, .21]]),
                        labels=torch.tensor([0, 2]), image_id='two'),
                   dict(boxes=torch.empty(0, 4), labels=torch.empty(0, dtype=torch.long), image_id='empty')]
        losses = DetectionLoss(Config())(model(images), targets)
        losses['total'].backward()
        self.assertTrue(torch.isfinite(losses['total']))
        self.assertGreater(model.head.weight.grad.abs().sum().item(), 0.)
        outputs = predict(model, images, score_threshold=0.)
        self.assertEqual(len(outputs), 2)
        self.assertTrue(((outputs[0]['boxes'] >= 0) & (outputs[0]['boxes'] <= 1)).all())

    def test_overflow_is_explicit(self):
        raw = torch.zeros(1, 4, 4, 3, 8, requires_grad=True)
        target = dict(boxes=torch.tensor([[.1, .1, .2, .2]] * 4),
                      labels=torch.zeros(4, dtype=torch.long), image_id='overflow')
        with self.assertRaisesRegex(ValueError, 'slots'):
            DetectionLoss(Config())(raw, [target])

    def test_perfect_ap_and_absent_class(self):
        target = dict(boxes=torch.tensor([[.1, .1, .2, .2]]), labels=torch.tensor([0]))
        pred = {**target, 'scores': torch.tensor([.9])}
        metrics = detection_metrics([pred], [target], 2)
        self.assertAlmostEqual(metrics['map50_95'], 1.)
        self.assertEqual(metrics['precision'], 1.)
        self.assertEqual(metrics['recall'], 1.)
        self.assertIsNone(metrics['per_class'][1]['ap50'])

    def test_duplicate_and_wrong_class_count_as_false_positives(self):
        box = torch.tensor([[.1, .1, .2, .2]])
        target = dict(boxes=box, labels=torch.tensor([0]))
        pred = dict(boxes=box.repeat(3, 1), labels=torch.tensor([0, 0, 1]), scores=torch.tensor([.9, .8, .7]))
        metrics = detection_metrics([pred], [target], 2)
        self.assertAlmostEqual(metrics['precision'], 1 / 3)
        self.assertEqual(metrics['recall'], 1.)

    def test_no_detection_has_zero_ap(self):
        target = dict(boxes=torch.tensor([[.1, .1, .2, .2]]), labels=torch.tensor([0]))
        pred = dict(boxes=torch.empty(0, 4), labels=torch.empty(0, dtype=torch.long), scores=torch.empty(0))
        metrics = detection_metrics([pred], [target], 1)
        self.assertEqual(metrics['map50_95'], 0.)

    def test_letterbox_flip_and_missing_labels(self):
        with tempfile.TemporaryDirectory(prefix='.cnn_test_', dir=Path.cwd()) as directory:
            root = Path(directory).resolve()
            self.assertEqual(root.parent, Path.cwd().resolve())
            (root / 'images/train').mkdir(parents=True)
            (root / 'labels/train').mkdir(parents=True)
            (root / 'data.yaml').write_text("names:\n- '0: fish'\n")
            Image.new('RGB', (200, 100)).save(root / 'images/train/a.jpg')
            label = root / 'labels/train/a.txt'
            label.write_text('0 0.25 0.5 0.2 0.4\n')
            cfg = Config(image_size=64)
            dataset = FishDataset(root, 'train', cfg, augment=True)
            with patch('cnn_propia.random.random', return_value=0.):
                image, target = dataset[0]
            self.assertEqual(list(image.shape), [3, 64, 64])
            self.assertTrue(torch.allclose(target['boxes'][0], torch.tensor([.65, .4, .85, .6])))
            label.unlink()
            with self.assertRaises(FileNotFoundError):
                FishDataset(root, 'train', cfg)


if __name__ == '__main__':
    unittest.main()
