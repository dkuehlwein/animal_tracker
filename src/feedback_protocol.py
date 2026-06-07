"""
Shared protocol for Telegram human-feedback buttons (ADR-004 Phase 1).

One source of truth for the inline-button `callback_data` so the *send* path
(`notification_service`) and the *receive* sidecar (`telegram_feedback`) can never
drift apart. `callback_data` format: ``fb:<detection_id>:<code>`` — short codes
keep us well under Telegram's 64-byte limit.
"""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

CALLBACK_PREFIX = "fb"

# Short wire code -> canonical label stored in detection_feedback.label.
CODE_TO_LABEL = {
    "a": "animal",
    "fp": "false_positive",
    "ws": "wrong_species",
}

# Button text shown to the human, in display order.
_BUTTONS = [
    ("✅ Animal", "a"),
    ("❌ False positive", "fp"),
    ("🐦 Wrong species", "ws"),
]


def build_feedback_keyboard(detection_id: int) -> InlineKeyboardMarkup:
    """Build the one-row feedback keyboard for a given detection."""
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(text, callback_data=f"{CALLBACK_PREFIX}:{detection_id}:{code}")
        for text, code in _BUTTONS
    ]])


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
