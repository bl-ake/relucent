"""Cached Gurobi environment, started without picking up a stray on-disk license when that fails."""

import os
import shutil
import tempfile

from gurobipy import GRB, Env, GurobiError, disposeDefaultEnv

from relucent._internal.logging import logger

__all__ = ["get_env"]


_env: Env | None = None


_default_env_disposed: bool = False


_LICENSE_ENV_START_FALLBACK_ERRNOS: frozenset[int] = frozenset(
    {
        GRB.ERROR_NO_LICENSE,
        GRB.ERROR_JOB_REJECTED,
        GRB.ERROR_NETWORK,
        GRB.ERROR_SECURITY,
    }
)


def _license_env_start_failure(exc: GurobiError) -> bool:
    if exc.errno in _LICENSE_ENV_START_FALLBACK_ERRNOS:
        return True
    msg = exc.message.lower()
    return "license" in msg or "token server" in msg or "web license service" in msg or "wls" in msg or "compute server" in msg


def _configure_cached_gurobi_env(env: Env, num_threads: int | None) -> None:
    if num_threads is not None:
        env.setParam("Threads", num_threads)
    env.setParam("OutputFlag", 0)
    env.setParam("LogToConsole", 0)


def _start_env_with_hidden_license_files(env: Env) -> None:
    """Run :meth:`gurobipy.Env.start` while hiding typical on-disk license paths.

    Temporarily clears ``GRB_LICENSE_FILE`` and points ``HOME`` at an empty
    directory so Gurobi does not pick up ``$HOME/gurobi.lic``. Environment
    variables are restored immediately after a successful ``start()``; Gurobi
    keeps using the session started here.

    Note:
        A license file under ``/opt/gurobi`` (Linux) or ``/Library/gurobi``
        (macOS) is still visible to Gurobi; this fallback cannot bypass a
        machine-wide installation license without OS-level changes.
    """
    saved_home = os.environ.get("HOME")
    saved_grb = os.environ.get("GRB_LICENSE_FILE")
    tmp = tempfile.mkdtemp(prefix="relucent-gurobi-nolic-")
    try:
        os.environ.pop("GRB_LICENSE_FILE", None)
        os.environ["HOME"] = tmp
        env.start()
    finally:
        if saved_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = saved_home
        if saved_grb is None:
            os.environ.pop("GRB_LICENSE_FILE", None)
        else:
            os.environ["GRB_LICENSE_FILE"] = saved_grb
        shutil.rmtree(tmp, ignore_errors=True)


def get_env(num_threads: int | None = None) -> Env:
    """Get a cached Gurobi environment.

    Creates and caches a Gurobi environment with logging disabled. This avoids
    the overhead of creating multiple environments. For more control over the
    environment, create and pass one directly to functions that need it.

    On first creation, Gurobi starts with the usual license discovery (license
    file, token server, WLS credentials in the environment, etc.). If that
    checkout fails with an error that indicates licensing rather than an
    internal bug, a second empty environment is started while typical on-disk
    license paths are hidden so Gurobi can fall back to its built-in size-
    limited license. Models larger than that license allows may raise
    ``ERROR_SIZE_LIMIT_EXCEEDED`` during solves.

    If ``num_threads`` is provided, the environment's ``Threads`` parameter is
    set accordingly on first creation. Subsequent calls ignore ``num_threads``
    and return the cached environment.

    Args:
        num_threads: Optional limit on the number of threads Gurobi may use.
            If None, Gurobi uses its default threading behavior.

    Returns:
        gurobipy.Env: A Gurobi environment with logging disabled.
    """
    global _env, _default_env_disposed
    if _env is not None:
        return _env
    if not _default_env_disposed:
        disposeDefaultEnv()
        _default_env_disposed = True
    candidate = Env(logfilename="", empty=True)
    _configure_cached_gurobi_env(candidate, num_threads)
    try:
        candidate.start()
    except GurobiError as exc:
        if not _license_env_start_failure(exc):
            candidate.close()
            raise
        candidate.close()
        candidate = Env(logfilename="", empty=True)
        _configure_cached_gurobi_env(candidate, num_threads)
        try:
            _start_env_with_hidden_license_files(candidate)
        except GurobiError:
            candidate.close()
            raise
        logger.warning(
            "Gurobi licensed checkout failed (%s); using built-in size-limited license instead.",
            exc.message,
        )
    _env = candidate
    return _env
