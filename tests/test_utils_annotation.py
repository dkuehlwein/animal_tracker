"""Tests for MotionVisualizer species-box drawing."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _write_blank_image(path, w=1920, h=1080):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_create_annotated_image_draws_species_box(tmp_path):
    from utils import MotionVisualizer
    from config import Config

    img_path = tmp_path / "capture.jpg"
    _write_blank_image(img_path)

    config = Config.create_test_config()
    # A box in the centre: normalized x=0.4,y=0.4,w=0.2,h=0.2 -> pixels ~ (768,432)-(1152,648)
    boxes = [{"bbox": [0.4, 0.4, 0.2, 0.2], "confidence": 0.91}]

    out = MotionVisualizer.create_annotated_image(
        img_path, None, config, None, bounding_boxes=boxes
    )

    assert out is not None
    out = Path(out)
    assert out.exists()

    annotated = cv2.imread(str(out))
    # Cyan rectangle (BGR 255,255,0) should appear somewhere inside the box region.
    region = annotated[432:648, 768:1152]
    cyan = (region[:, :, 0] > 200) & (region[:, :, 1] > 200) & (region[:, :, 2] < 60)
    assert cyan.sum() > 0, "expected cyan box pixels in the species-box region"


def test_create_annotated_image_no_boxes_returns_none_without_motion(tmp_path):
    from utils import MotionVisualizer
    from config import Config

    img_path = tmp_path / "capture.jpg"
    _write_blank_image(img_path)
    config = Config.create_test_config()

    # No motion_result and no boxes -> nothing to draw, must not raise.
    out = MotionVisualizer.create_annotated_image(
        img_path, None, config, None, bounding_boxes=None
    )
    assert out is None
