# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
Heretic: Fully automatic censorship removal for language models.
"""

from .config import Settings
from .model import Model
from .system import (
    detect_tpu,
    get_xla_device,
    get_xla_device_count,
    is_torch_xla_available,
    mark_step,
    setup_tpu_environment,
)

__version__ = "1.4.0"

__all__ = [
    "Settings",
    "Model",
    "detect_tpu",
    "get_xla_device",
    "get_xla_device_count",
    "is_torch_xla_available",
    "mark_step",
    "setup_tpu_environment",
]