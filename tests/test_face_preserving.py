"""Boundary checks for the standalone square-preview rule."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cropbias.face_preserving import choose, render
from cropbias.geometry import BOTH, ONLY1, crop_width, inside_fraction
from cropbias.rules import View


def view(face2):
    W, H = 400, 256
    xs = np.arange(10, W, 20, dtype=np.float32)
    scores = np.full((2, len(xs)), -4, dtype=np.float32)
    scores[0, np.where(xs == 30)[0][0]] = 2
    scores[1, np.where(xs == 270)[0][0]] = 1
    return View(W, H, crop_width(H, W, 1.0), (20., 60.), face2,
                xs, scores, 30)


class FacePreservingTests(unittest.TestCase):
    def test_search_finds_square_argmax_misses(self):
        v = view((230., 270.))
        self.assertEqual(v.arg["out"], ONLY1)
        c = choose(v)
        self.assertTrue(c.feasible)
        self.assertFalse(c.fallback)
        self.assertEqual(c.outcome, BOTH)
        self.assertGreaterEqual(inside_fraction(v.face1, c.x0, c.x1), .9)
        self.assertGreaterEqual(inside_fraction(v.face2, c.x0, c.x1), .9)
        # The right peak enters only for some of the otherwise feasible squares.
        self.assertGreaterEqual(c.x0, 15)

    def test_impossible_square_uses_padded_full_image(self):
        c = choose(view((340., 380.)))
        self.assertTrue(c.fallback)
        self.assertFalse(c.feasible)
        self.assertEqual(c.outcome, BOTH)
        self.assertEqual(c.saliency_retained, 1)
        with tempfile.TemporaryDirectory() as temp:
            src, dst = Path(temp) / "src.png", Path(temp) / "preview.png"
            Image.new("RGB", (400, 256), (220, 10, 10)).save(src)
            render(src, c, dst)
            with Image.open(dst) as img:
                self.assertEqual(img.size, (256, 256))
                self.assertTrue(all(abs(a - b) <= 2 for a, b in zip(
                    img.getpixel((128, 128)), (220, 10, 10))))
                self.assertTrue(all(abs(a - b) <= 2 for a, b in zip(
                    img.getpixel((128, 0)), (32, 32, 32))))

    def test_invalid_narrow_composite_fails_explicitly(self):
        xs = np.array([10., 30.], np.float32)
        v = View(200, 256, 200, (10., 20.), (40., 50.),
                 xs, np.zeros((2, 2), np.float32), 10)
        with self.assertRaisesRegex(ValueError, "wide composite"):
            choose(v)


if __name__ == "__main__":
    unittest.main()
