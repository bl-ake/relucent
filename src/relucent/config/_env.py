"""Read ``RELUCENT_<SETTING>`` environment variables for the config modules."""

from __future__ import annotations

import os
from collections.abc import Sequence


def env_name(setting: str) -> str:
    return f"RELUCENT_{setting}"


def env_str(setting: str, default: str) -> str:
    return os.getenv(env_name(setting), default)


def env_choice(setting: str, default: str, choices: Sequence[str]) -> str:
    value = env_str(setting, default)
    check_choice(setting, value, choices)
    return value


def check_choice(setting: str, value: object, choices: Sequence[str]) -> None:
    if value not in choices:
        raise ValueError(f"Invalid value for {setting}: {value!r}; expected one of {list(choices)}")


def env_float(setting: str, default: float) -> float:
    raw = os.getenv(env_name(setting))
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid float value for {env_name(setting)!r}: {raw!r}") from exc


def env_optional_float(setting: str) -> float | None:
    """Like :func:`env_float` with no default: unset (or ``"auto"``) gives ``None``."""
    raw = os.getenv(env_name(setting))
    if raw is None or raw.strip().lower() == "auto":
        return None
    return env_float(setting, 0.0)


def env_int(setting: str, default: int) -> int:
    raw = os.getenv(env_name(setting))
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid int value for {env_name(setting)!r}: {raw!r}") from exc


def env_bool(setting: str, default: bool) -> bool:
    raw = os.getenv(env_name(setting))
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid bool value for {env_name(setting)!r}: {raw!r}")


def env_float_list(setting: str, default: list[float]) -> list[float]:
    raw = os.getenv(env_name(setting))
    if raw is None:
        return default

    # Accepts "0.1,1,10" or "[0.1, 1, 10]".
    cleaned = raw.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"Invalid float list value for {env_name(setting)!r}: {raw!r}")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid float list value for {env_name(setting)!r}: {raw!r}") from exc
