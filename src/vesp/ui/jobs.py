"""Background execution for the Mission Console.

Two kinds of work, two mechanisms:

- :class:`ProcessJob` -- the heavy, documented pipelines (``python -m vesp.uq.run``,
  ``python -m vesp.uq.screen``) run as **subprocesses** with live line streaming and clean
  cancellation. The UI stays a thin controller over the same reproducible CLIs a user would run
  by hand, and reads results back from the artifact layer.
- :class:`FnWorker` -- light in-process tasks (loading a model, building plot data, a sequential
  update) run on a ``QThread``; ``vesp.uq``/torch are imported lazily inside the callable so the
  window itself opens instantly.
"""

from __future__ import annotations

import os
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QThread, pyqtSignal

from vesp.ui.paths import ROOT


class ProcessJob(QObject):
    """One cancellable subprocess with merged stdout/stderr line streaming."""

    line = pyqtSignal(str)
    started = pyqtSignal()
    finished = pyqtSignal(int)  # exit code (-1 for crash/kill)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process: QProcess | None = None
        self._buffer = ""

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning

    def start_module(self, module: str, args: list[str]) -> None:
        """Run ``python -m <module> <args...>`` from the repository root."""

        self.start([sys.executable, "-u", "-m", module, *args])

    def start(self, command: list[str]) -> None:
        if self.running:
            raise RuntimeError("a job is already running")
        process = QProcess(self)
        process.setWorkingDirectory(str(ROOT))
        env = QProcessEnvironment.systemEnvironment()
        # Make `vesp` importable in the child even without an editable install.
        src = str(ROOT / "src")
        existing = env.value("PYTHONPATH", "")
        if src not in existing.split(os.pathsep):
            env.insert("PYTHONPATH", src + (os.pathsep + existing if existing else ""))
        env.insert("PYTHONIOENCODING", "utf-8")
        process.setProcessEnvironment(env)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._drain)
        process.finished.connect(self._on_finished)
        # A crashed process still emits finished(); FailedToStart is the one terminal state
        # that never would, so surface it explicitly or the UI would wait forever.
        process.errorOccurred.connect(self._on_error)
        self._process = process
        self._buffer = ""
        process.start(command[0], command[1:])
        self.started.emit()

    def cancel(self) -> None:
        process = self._process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            process.kill()

    def _drain(self) -> None:
        if self._process is None:
            return
        data = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.line.emit(line.rstrip("\r"))

    def _on_error(self, error) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.line.emit("[job] subprocess failed to start (python interpreter not reachable?)")
            self.finished.emit(-1)

    def _on_finished(self, exit_code: int, _status) -> None:
        self._drain()
        if self._buffer:
            self.line.emit(self._buffer)
            self._buffer = ""
        self.finished.emit(int(exit_code))


class FnWorker(QThread):
    """Run ``fn()`` on a worker thread; emit the result or a formatted traceback."""

    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn: Callable[[], object], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:  # pragma: no cover - thread body exercised via the app
        try:
            self.done.emit(self._fn())
        except Exception:
            self.failed.emit(traceback.format_exc())


def open_in_file_manager(path: Path) -> None:
    """Reveal a file/folder in the OS file manager (best effort, never raises)."""

    from PyQt6.QtCore import QUrl
    from PyQt6.QtGui import QDesktopServices

    target = path if path.is_dir() else path.parent
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))


def open_file(path: Path) -> None:
    """Open a file with its default application (best effort, never raises)."""

    from PyQt6.QtCore import QUrl
    from PyQt6.QtGui import QDesktopServices

    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
