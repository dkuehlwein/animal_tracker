# Species Box Annotation on Telegram Notifications

**Date:** 2026-06-09
**Status:** Design — awaiting user review

## Problem

When a detection notification arrives in Telegram, the user often cannot visually
locate the animal in the photo and worries about missing real sightings. The
current single-image notification gives no visual cue of *where* the system thinks
the animal is.

Separately, the `blank` classifier verdict currently renders with a real-species
emoji (default 🦌), making a likely false positive look like a genuine sighting.

## Goals

1. Send a second, annotated image alongside the original whenever the detection
   pipeline produced a bounding box, so the user can see exactly where the animal
   (or false-positive object) was detected.
2. The annotated image is a **combined debug view**: existing motion overlay **plus**
   the MegaDetector species box(es).
3. Fix the `blank` verdict to render as **🚫 No animal** instead of a species emoji.

## Non-Goals

- No change to the two-image media-group transport (already exists).
- No change to motion detection, capture, or the species pipeline itself.
- No new config flags for the species box (it is default-on when a box exists).

## Background — what already exists

- `NotificationService.send_media_group` (`src/notification_service.py:145`) already
  sends `[original, annotated]` as a Telegram media group.
- `wildlife_system.send_notification` (`src/wildlife_system.py:482`) already routes to
  the media group when an `annotated_path` is present, then sends the feedback
  keyboard as a follow-up message (media groups can't carry an inline keyboard).
- `MotionVisualizer.create_annotated_image` (`src/utils.py:67`) draws the motion
  overlay (green central region, red motion point, legend) scaled from motion
  resolution onto the high-res image.
- MegaDetector boxes are already captured in
  `species_result['detection_result'].bounding_boxes`, each
  `{'bbox': [x_min, y_min, width, height], 'confidence': float, 'category': ...}`
  in **normalized** (0–1) coordinates.

## Design

### 1. Combined annotation drawing (`src/utils.py`)

Extend `MotionVisualizer.create_annotated_image` to accept an optional
`bounding_boxes` argument (default `None`):

```
create_annotated_image(image_path, motion_frame, config, motion_result,
                       bounding_boxes=None) -> Optional[Path]
```

- Existing motion-overlay drawing is unchanged.
- When `bounding_boxes` is non-empty, for each box: scale the normalized
  `[x_min, y_min, w, h]` to pixel coords against the high-res image size, draw a
  rectangle in **cyan** (BGR `(255, 255, 0)` — visually separate from the green
  central region and red motion marker), and a small `Box: NN%` label above it.
- Extend the legend to name the species-box color.
- Keep the function pure/IO-light: one input image → one output `<name>_boxed.jpg`,
  testable with a synthetic image and a known box.

Rationale for extending `MotionVisualizer` rather than a new class: the output is a
single combined image, so one function owning the full overlay avoids re-reading and
re-writing the image twice.

### 2. Wiring (`src/wildlife_system.py`, ~line 719)

Replace the current annotation block with:

- Pull `detection_result` from `species_result`; compute its `bounding_boxes`.
- **If boxes exist:** always generate the combined annotated image (motion overlay +
  species boxes) in the executor, pass as `annotated_path`. This is the default-on
  path — independent of the `send_annotated_image` flag.
- **If no boxes but `send_annotated_image` is True:** generate the motion-only
  overlay as today (preserves motion-debug capability for false-positive tuning).
- **Otherwise:** `annotated_path = None` (original image only).

`send_notification` needs no change.

### 3. Blank emoji fix (`src/wildlife_system.py`, `_build_caption`)

In the `IDENTIFIED` branch, when the extracted species name is `blank` (case-
insensitive), render the verdict line as `🚫 No animal (CONF)` instead of
`{species_emoji} Blank (CONF)`. The `| Box: NN%` suffix is retained so the user can
still see the detector confidence. (`_get_species_emoji` is unchanged; the special-
case lives in the caption builder.)

## Behavior matrix

| Detection outcome              | Box exists? | 2nd image sent          |
|--------------------------------|-------------|-------------------------|
| identified                     | yes         | combined (motion+box)   |
| animal_uncertain               | yes         | combined (motion+box)   |
| unclassifiable                 | yes         | combined (motion+box)   |
| blank (with detector box)      | yes         | combined (motion+box)   |
| no_animal (zero boxes)         | no          | motion-only if flag, else none |
| error                          | no          | none                    |

## Data flow

Unchanged except the annotated image now also consumes
`detection_result.bounding_boxes`. No new persistence, no schema change.

## Error handling

- Box drawing wrapped in the existing try/except in `create_annotated_image`; on any
  failure it returns `None` and the notification falls back to the single original
  image (current behavior). A failed annotation never blocks the notification.

## Testing

- **New unit test** for `create_annotated_image` with `bounding_boxes`: synthetic
  1920×1080 image + one known normalized box → assert output file created and that
  pixels at the expected box location changed (rectangle drawn).
- **Caption test**: `_build_caption` with a `blank` species name → asserts `🚫 No
  animal` in the output and absence of a species emoji.
- Existing notification/media-group tests already cover the two-image send path.

## Open questions

None outstanding — all design decisions confirmed with the user.
