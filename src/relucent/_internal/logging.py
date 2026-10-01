"""Package-level logger for relucent.

Progress and status messages go through the ``"relucent"`` logger, whose level
follows :data:`relucent.config.VERBOSE`:

* ``VERBOSE >= 1`` (default): ``INFO``, normal progress.
* ``VERBOSE = 0``: ``WARNING``, quiet.

It has one :class:`~logging.StreamHandler` writing plain messages to stderr. Handlers
you add to the ``"relucent"`` logger take precedence; set ``logger.propagate = True``
to also forward records to the root logger.
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
