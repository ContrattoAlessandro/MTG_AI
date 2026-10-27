"""A version counter that wakes the threads waiting for something to change."""

import threading


class ChangeFeed:
    """Every change to the table bumps the version; readers wait for a version newer than theirs."""

    def __init__(self):
        self._cond = threading.Condition()
        self._version = 0

    @property
    def version(self) -> int:
        with self._cond:
            return self._version

    def bump(self) -> None:
        with self._cond:
            self._version += 1
            self._cond.notify_all()

    def wait(self, seen: int, timeout: float) -> int:
        """The current version, as soon as it differs from `seen` or after `timeout` seconds."""
        with self._cond:
            self._cond.wait_for(lambda: self._version != seen, timeout=timeout)
            return self._version
