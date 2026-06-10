"""VESP-UQ Mission Console: a PyQt6 desktop app managing the full layer lifecycle.

Pages: Dashboard (fleet overview), Train (fit + package a model), Screen (serve a persisted
model over new ensembles), Model (inspect a saved artifact + model card + uncertainty profile),
Update (exact sequential posterior update), Runs (artifact/manifest browser).

Architecture: the heavy pipelines run as subprocesses of the documented CLIs
(``python -m vesp.uq.run`` / ``python -m vesp.uq.screen``) via :class:`vesp.ui.jobs.ProcessJob`
-- the UI is a controller over the same reproducible entry points, and reads results back from
the artifact/manifest layer. Light in-process work (model inspection, plots, updates) runs on a
:class:`vesp.ui.jobs.FnWorker` thread with lazy ``vesp.uq``/torch imports so the window opens
instantly. Launch with ``python ui/app_vespuq.py``.
"""
