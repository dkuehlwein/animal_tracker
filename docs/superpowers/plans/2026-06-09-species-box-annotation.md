# Species Box Annotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Send a second annotated image (motion overlay + MegaDetector species box) alongside the original whenever a detection box exists, and render the `blank` verdict as 🚫 No animal.

**Architecture:** Extend `MotionVisualizer.create_annotated_image` to also draw normalized MegaDetector boxes in cyan; wire `wildlife_system` to generate the combined image whenever `detection_result.bounding_boxes` is non-empty (default-on); special-case the `blank` species name in `_build_caption`.

**Tech Stack:** Python 3.13, OpenCV (cv2), pytest. Run tests with `uv run pytest`.

---

### Task 1: Render `blank` verdict as 🚫 No animal

**Files:**
- Modify: `src/wildlife_system.py` (`_build_caption`, IDENTIFIED branch ~line 382-387)
- Test: `tests/test_detection_status.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_detection_status.py` (uses existing `_make_system` / `_species_result_for_status` helpers):

```python
def test_caption_blank_shows_no_animal(monkeypatch, tmp_path):
    from data_models import DetectionStatus
    sys_obj = _make_system(monkeypatch, tmp_path)
    sr = _species_result_for_status(DetectionStatus.IDENTIFIED, "blank", 1.0)
    caption = sys_obj._build_caption(sr, 5000, datetime(2026, 6, 9, 14, 30, 0))
    assert "🚫 No animal" in caption
    # Must NOT fall back to a species emoji like the default 🦌
    assert "🦌" not in caption
    # Detector box confidence still surfaced
    assert "Box:" in caption
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_detection_status.py::test_caption_blank_shows_no_animal -v`
Expected: FAIL (caption contains `🦌 Blank`, not `🚫 No animal`).

- [ ] **Step 3: Implement the special-case**

In `src/wildlife_system.py`, inside `_build_caption`, the `if status == DetectionStatus.IDENTIFIED:` branch currently reads:

```python
        if status == DetectionStatus.IDENTIFIED:
            emoji = self._get_species_emoji(species_name)
            species_line = f"{emoji} {species_name} ({confidence:.0%})"
            if max_detection_conf > 0:
                species_line += f" | Box: {max_detection_conf:.0%}"
            caption += f"\n{species_line}"
```

Change the first three lines to special-case the `blank` sentinel:

```python
        if status == DetectionStatus.IDENTIFIED:
            if species_name.strip().lower() == "blank":
                species_line = f"🚫 No animal ({confidence:.0%})"
            else:
                emoji = self._get_species_emoji(species_name)
                species_line = f"{emoji} {species_name} ({confidence:.0%})"
            if max_detection_conf > 0:
                species_line += f" | Box: {max_detection_conf:.0%}"
            caption += f"\n{species_line}"
```

Note: `species_name` here is the output of `_extract_species_name`, which title-cases tokens, so the raw `blank` sentinel becomes `Blank`. The `.lower()` comparison handles both.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_detection_status.py::test_caption_blank_shows_no_animal -v`
Expected: PASS.

- [ ] **Step 5: Run the full caption test group to confirm no regression**

Run: `uv run pytest tests/test_detection_status.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/wildlife_system.py tests/test_detection_status.py
git commit -m "feat(notify): render blank verdict as 🚫 No animal"
```

---

### Task 2: Draw MegaDetector species boxes in the annotated image

**Files:**
- Modify: `src/utils.py` (`MotionVisualizer.create_annotated_image`, ~line 67-165)
- Test: `tests/test_utils_annotation.py` (create)

**Context on coordinates:** Each entry of `bounding_boxes` is
`{'bbox': [x_min, y_min, width, height], 'confidence': float, ...}` with all bbox
values **normalized to 0–1** relative to the full image. Pixel conversion against an
image of size `(img_height, img_width)`:
`x1 = int(x_min * img_width)`, `y1 = int(y_min * img_height)`,
`x2 = int((x_min + width) * img_width)`, `y2 = int((y_min + height) * img_height)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_utils_annotation.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_utils_annotation.py -v`
Expected: FAIL — `create_annotated_image` has no `bounding_boxes` parameter (TypeError) and/or boxes not drawn.

- [ ] **Step 3: Read the current function and adapt the signature + guard**

Open `src/utils.py`. The current signature (line ~67) is:

```python
    @staticmethod
    def create_annotated_image(image_path: Path, motion_frame, config, motion_result) -> Optional[Path]:
```

Change it to add the optional boxes argument:

```python
    @staticmethod
    def create_annotated_image(image_path: Path, motion_frame, config, motion_result,
                               bounding_boxes=None) -> Optional[Path]:
```

The function reads the image, then draws the motion overlay using `motion_result`.
The existing body assumes `motion_result` is present. Two changes:

(a) Early in the function, after `img = cv2.imread(...)` and computing `img_height,
img_width`, guard the motion-overlay drawing so it only runs when `motion_result` is
not None. Wrap the existing central-region + motion-point + motion legend drawing in:

```python
            if motion_result is not None:
                # ... existing motion-overlay drawing (central region, marker, legend) ...
```

(b) If both `motion_result is None` and not `bounding_boxes`, return None before
writing any file (nothing to annotate):

```python
            if motion_result is None and not bounding_boxes:
                return None
```

- [ ] **Step 4: Add species-box drawing**

After the motion-overlay block (and before `cv2.imwrite`), add:

```python
            # Draw MegaDetector species boxes (normalized bbox -> pixels), cyan.
            if bounding_boxes:
                box_color = (255, 255, 0)  # cyan in BGR
                font = cv2.FONT_HERSHEY_SIMPLEX
                for box in bounding_boxes:
                    bbox = box.get("bbox")
                    if not bbox or len(bbox) < 4:
                        continue
                    x_min, y_min, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                    x1 = int(x_min * img_width)
                    y1 = int(y_min * img_height)
                    x2 = int((x_min + w) * img_width)
                    y2 = int((y_min + h) * img_height)
                    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)
                    conf = box.get("confidence", 0.0)
                    label = f"Box: {conf:.0%}"
                    ly = max(y1 - 8, 20)
                    cv2.putText(img, label, (x1, ly), font, 0.7, box_color, 2)
```

If the existing function uses a different local variable name for image dimensions
(e.g. `height, width` instead of `img_height, img_width`), use whatever names are
already defined in the function — do not introduce duplicates.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_utils_annotation.py -v`
Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add src/utils.py tests/test_utils_annotation.py
git commit -m "feat(utils): draw MegaDetector species boxes in annotated image"
```

---

### Task 3: Wire the combined annotation into the detection notification

**Files:**
- Modify: `src/wildlife_system.py` (annotation block ~line 719-735)

**Context:** `species_result` is a dict containing `detection_result` (a
`DetectionResult` with `.bounding_boxes`). The current block only builds the motion
overlay when `send_annotated_image` is set. We want: boxes exist -> always build the
combined image; else fall back to the old flag-gated motion-only overlay.

- [ ] **Step 1: Replace the annotation block**

In `src/wildlife_system.py`, the current block is:

```python
                                # Create annotated image showing motion detection regions (debug only)
                                annotated_path = None
                                if (self.config.performance.send_annotated_image
                                        and self.last_motion_frame is not None
                                        and self.last_motion_result is not None):
                                    annotated_path = await loop.run_in_executor(
                                        self.executor,
                                        MotionVisualizer.create_annotated_image,
                                        image_path,
                                        self.last_motion_frame,
                                        self.config,
                                        self.last_motion_result
                                    )
```

Replace it with:

```python
                                # Annotated image: combined motion overlay + MegaDetector
                                # species box. Sent whenever a detection box exists (default-on);
                                # otherwise the motion-only overlay is built only when the
                                # send_annotated_image debug flag is set.
                                annotated_path = None
                                detection_result = species_result.get('detection_result')
                                species_boxes = (
                                    detection_result.bounding_boxes if detection_result else None
                                )
                                motion_result_for_overlay = (
                                    self.last_motion_result
                                    if self.last_motion_frame is not None else None
                                )
                                want_annotation = bool(species_boxes) or (
                                    self.config.performance.send_annotated_image
                                    and motion_result_for_overlay is not None
                                )
                                if want_annotation:
                                    annotated_path = await loop.run_in_executor(
                                        self.executor,
                                        functools.partial(
                                            MotionVisualizer.create_annotated_image,
                                            image_path,
                                            self.last_motion_frame,
                                            self.config,
                                            motion_result_for_overlay,
                                            bounding_boxes=species_boxes,
                                        )
                                    )
```

- [ ] **Step 2: Ensure `functools` is imported**

Check the top of `src/wildlife_system.py` for `import functools`. If absent, add it
alongside the other stdlib imports. (`run_in_executor` does not accept keyword args,
so `functools.partial` is required to pass `bounding_boxes=`.)

Run: `grep -n "^import functools" src/wildlife_system.py` — if no output, add the import.

- [ ] **Step 3: Verify the module imports cleanly**

Run: `uv run python -c "import sys; sys.path.insert(0,'src'); import wildlife_system"`
Expected: no output, exit 0 (no syntax/import error).

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest tests/ -q`
Expected: all PASS (no regressions; note some hardware/camera tests may already be
skipped in this environment — only real failures matter).

- [ ] **Step 5: Commit**

```bash
git add src/wildlife_system.py
git commit -m "feat(notify): send combined motion+species-box image when a box exists"
```

---

## Self-review notes

- **Spec coverage:** Goal 1 (2nd image when box exists) → Task 3. Goal 2 (combined
  motion+species overlay) → Tasks 2 + 3. Goal 3 (blank → 🚫 No animal) → Task 1.
  Behavior matrix rows map to Task 3's `want_annotation` logic.
- **Fallback on annotation failure:** `create_annotated_image` already returns None
  inside its try/except on error; `send_notification` then sends the single original
  image. No extra handling needed.
- **No new config flags** — species box is default-on, consistent with the spec.
