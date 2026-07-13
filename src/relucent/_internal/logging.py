"""Package-level logger for relucent.

All user-facing progress and status messages flow through the ``"relucent"``
logger.  The level is driven by :data:`relucent.config.VERBOSE`:

* ``VERBOSE >= 1`` (default) → ``logging.INFO``   — normal progress messages
* ``VERBOSE = 0``             → ``logging.WARNING`` — silent (errors/warnings only)

Higher ``VERBOSE`` values are reserved for future ``DEBUG``-level output.

The logger ships with a single :class:`~logging.StreamHandler` writing plain
messages (no ``levelname`` prefix) to *stderr*.  Applications that configure
their own handlers on the ``"relucent"`` logger will take precedence; set
``logger.propagate = True`` to additionally forward records up to the root
logger.
"""

from __future__ import annotations

import logging

logger: logging.Logger = logging.getLogger("relucent")

_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)  # mirrors VERBOSE=1 default
logger.propagate = False


def _apply_verbose(verbose: int) -> None:
    """Adjust the relucent logger level to match a VERBOSE config value."""
    logger.setLevel(logging.INFO if verbose >= 1 else logging.WARNING)
