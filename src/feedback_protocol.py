"""
Shared protocol for Telegram human-feedback buttons (ADR-004 Phase 1, extended
2026-07-09 for the 5-button/2-row label redesign).

One source of truth for the inline-button `callback_data` so the *send* path
(`notification_service`) and the *receive* sidecar (`telegram_feedback`) can never
drift apart. `callback_data` format: ``fb:<detection_id>:<code>`` — short codes
keep us well under Telegram's 64-byte limit.

`ws` (-> `wrong_species`) is legacy: it is still accepted by `parse_callback_data`
so buttons on old messages already sitting in the channel keep working, but it is
no longer shown on freshly built keyboards (superseded by `wid`/`p`).
"""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

CALLBACK_PREFIX = "fb"

# Short wire code -> canonical label stored in detection_feedback.label.
# `ws` is parse-only legacy (see module docstring) - never in _BUTTON_ROWS.
CODE_TO_LABEL = {
    "a": "animal",
    "wid": "animal_wrong_id",
    "p": "person",
    "fp": "false_positive",
    "ct": "cant_tell",
    "ws": "wrong_species",
}

# Codes shown on freshly built keyboards, in display order.
DISPLAYED_CODES = ["a", "wid", "p", "fp", "ct"]

# Button text shown to the human, grouped into keyboard rows: row 1 = "something
# real was there", row 2 = "nothing / unusable" (spatial grouping does the
# explaining - see docs/superpowers/specs/2026-07-09-feedback-label-redesign-design.md).
_BUTTON_ROWS = [
    [("✅ Animal", "a"), ("🐦 Animal, wrong ID", "wid"), ("👤 Human", "p")],
    [("❌ Nothing there", "fp"), ("🤷 Can't tell", "ct")],
]


def build_feedback_keyboard(detection_id: int) -> InlineKeyboardMarkup:
    """Build the two-row feedback keyboard for a given detection."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(text, callback_data=f"{CALLBACK_PREFIX}:{detection_id}:{code}")
            for text, code in row
        ]
        for row in _BUTTON_ROWS
    ])


def parse_callback_data(data: str) -> tuple[int, str]:
    """Parse ``fb:<detection_id>:<code>`` into ``(detection_id, label)``.

    Raises ValueError on anything malformed or an unknown code, so a bad/foreign
    callback can't write a garbage row.
    """
    if not data:
        raise ValueError("empty callback data")
    parts = data.split(":")
    if len(parts) != 3 or parts[0] != CALLBACK_PREFIX:
        raise ValueError(f"unrecognised callback data: {data!r}")
    _, raw_id, code = parts
    try:
        detection_id = int(raw_id)
    except ValueError as e:
        raise ValueError(f"non-integer detection id in {data!r}") from e
    if code not in CODE_TO_LABEL:
        raise ValueError(f"unknown feedback code {code!r} in {data!r}")
    return detection_id, CODE_TO_LABEL[code]
