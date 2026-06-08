#!/usr/bin/env python3
"""
Telegram feedback sidecar (ADR-004 Phase 1).

A standalone, long-lived process that listens for taps on the inline feedback
keyboard (✅ Animal / ❌ False positive / 🐦 Wrong species) attached to each
detection notification, and appends the human label to `detection_feedback`.

Runs as a *separate process* from the main detection loop, sharing the SQLite DB
via WAL. The main process is send-only, so only this sidecar polls getUpdates —
there is no getUpdates conflict.

    uv run python src/telegram_feedback.py
"""

import logging
import sys as _sys
from pathlib import Path as _Path

from telegram import Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

from config import Config
from database_manager import DatabaseManager
from feedback_protocol import CALLBACK_PREFIX, parse_callback_data

_SRC = _Path(__file__).resolve().parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))
from loop import state as _state_mod
from loop import deploy as _deploy

logger = logging.getLogger(__name__)

# Human-facing confirmation shown after a tap, per recorded label.
_LABEL_CONFIRMATION = {
    "animal": "✅ Recorded: animal",
    "false_positive": "❌ Recorded: false positive",
    "wrong_species": "🐦 Recorded: wrong species",
}


def record_feedback_callback(data: str, database: DatabaseManager) -> str:
    """Parse callback data and append the label. Returns a user-facing message.

    Pure of any Telegram I/O so it is unit-testable. Raises ValueError on
    malformed data (the caller answers the toast accordingly).
    """
    detection_id, label = parse_callback_data(data)
    database.add_feedback(detection_id, label, source="human")
    logger.info(f"Recorded human feedback: detection={detection_id}, label={label}")
    return _LABEL_CONFIRMATION.get(label, f"Recorded: {label}")


DEFAULT_STATE_PATH = "experiments/state.json"
DEFAULT_ENV_PATH = "experiments/deployed_config.env"


def handle_pause(state_path: str = DEFAULT_STATE_PATH) -> str:
    """Set paused=true in state.json. Pure of Telegram I/O for testability."""
    st = _state_mod.load_state(state_path)
    st["paused"] = True
    _state_mod.save_state(state_path, st)
    logger.info("Loop paused via /pause command")
    return "Tuning paused. The loop will hold best_known_good until resumed."


def handle_rollback(
    state_path: str = DEFAULT_STATE_PATH,
    env_path: str = DEFAULT_ENV_PATH,
    restart_at: str = None,
) -> str:
    """Roll back to best_known_good via deploy.rollback()."""
    from datetime import datetime, timedelta

    if restart_at is None:
        # Default: ASAP (next minute) — the deploy timer applies it pre-sunrise,
        # but a manual rollback should restart at the next opportunity.
        restart_at = (datetime.now().astimezone() + timedelta(minutes=1)).isoformat()
    result = _deploy.rollback(state_path, env_path, restart_at)
    logger.info(f"Rollback requested via /rollback: {result}")
    return (
        "Rollback queued: restored best_known_good "
        f"({result['deployed']}); camera restart stamped."
    )


async def _on_pause(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = handle_pause()
    await update.message.reply_text(msg)


async def _on_rollback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = handle_rollback()
    await update.message.reply_text(msg)


async def _on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """CallbackQueryHandler entry point: record the tap and confirm to the user."""
    query = update.callback_query
    database: DatabaseManager = context.application.bot_data["database"]
    try:
        message = record_feedback_callback(query.data, database)
        await query.answer(text=message)
        # Reflect the recorded label inline so the channel shows a persistent
        # acknowledgement (the toast above is ephemeral). Photos carry a caption;
        # the debug media-group path attaches the keyboard to a text message.
        if query.message is None:
            pass
        elif query.message.caption is not None:
            await query.edit_message_caption(
                caption=f"{query.message.caption}\n— {message}"
            )
        elif query.message.text is not None:
            await query.edit_message_text(text=f"{query.message.text}\n— {message}")
    except ValueError as e:
        logger.warning(f"Ignoring bad feedback callback {query.data!r}: {e}")
        await query.answer(text="Could not record that.")
    except Exception as e:
        logger.error(f"Error handling feedback callback: {e}", exc_info=True)
        await query.answer(text="Error recording feedback.")


def build_application(config: Config, database: DatabaseManager) -> Application:
    """Construct the polling Application with the feedback handler wired up."""
    application = Application.builder().token(config.telegram_token).build()
    application.bot_data["database"] = database
    application.add_handler(
        CallbackQueryHandler(_on_callback, pattern=f"^{CALLBACK_PREFIX}:")
    )
    application.add_handler(CommandHandler("pause", _on_pause))
    application.add_handler(CommandHandler("rollback", _on_rollback))
    return application


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    for noisy in ("httpx", "httpcore", "telegram"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    config = Config()
    database = DatabaseManager(config)
    application = build_application(config, database)

    logger.info("Telegram feedback sidecar starting (polling for button taps)...")
    application.run_polling(allowed_updates=["callback_query", "message"])


if __name__ == "__main__":
    main()
