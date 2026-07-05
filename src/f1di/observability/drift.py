from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import asdict

from f1di.agents.classifier_utils import _CALIBRATION_DIR

logger = logging.getLogger("f1di.observability.drift")

_ALERT_THRESHOLD = 3.5
_MIN_BASELINE = 50
_BUFFER_SIZE = 200
_DRIFT_RETRAIN_COOLDOWN_S = 3600
_DRIFT_RETRAIN_STAMP = _CALIBRATION_DIR / ".last_drift_retrain"
_DRIFT_RETRAIN_HISTORY = _CALIBRATION_DIR / ".drift_retrain_history"
_CONCEPT_DRIFT_THRESHOLD = 3  # retrains within 24h without clearing = concept drift
_CONCEPT_DRIFT_WINDOW_S = 86400

_TRACKED = frozenset({
    "fl_wear", "fr_wear", "rear_wear_mean",
    "fl_wear_slope", "fr_wear_slope", "rear_wear_slope",
    "brake_temp_front_max", "battery_soc", "battery_soc_slope",
    "rain_intensity", "grip_estimate", "lockup_count",
    "mean_speed_kph", "axle_imbalance_fl_rl",
    "laps_remaining", "stint_fraction", "race_phase",
})


class FeatureDriftTracker:
    """Rolling Z-score drift detector for telemetry input features.

    Maintains a circular buffer of recent observations. Once at least
    _MIN_BASELINE samples have been seen, every new observation is scored
    against the rolling mean/std for each feature. Z-scores > _ALERT_THRESHOLD
    are logged as warnings and reflected in Prometheus gauges.
    """

    def __init__(
        self,
        buffer_size: int = _BUFFER_SIZE,
        min_baseline: int = _MIN_BASELINE,
        alert_threshold: float = _ALERT_THRESHOLD,
    ) -> None:
        self._buffer: deque[dict[str, float]] = deque(maxlen=buffer_size)
        self._min_baseline = min_baseline
        self._alert_threshold = alert_threshold
        self._baseline: dict[str, tuple[float, float]] = {}
        self._last_zscores: dict[str, float] = {}
        self._last_updated: str | None = None
        self._track_buffers: dict[str, deque[dict[str, float]]] = {}
        self._track_baselines: dict[str, dict[str, tuple[float, float]]] = {}
        self._last_track: str | None = None
        self._prev_alert: bool = False

    def update(self, features: dict[str, float], track_id: str | None = None) -> dict[str, float]:
        """Record one observation. Returns Z-scores or {} during warmup."""
        from f1di.observability.metrics import DRIFT_ALERT_ACTIVE, FEATURE_DRIFT_ZSCORE

        tracked = {k: v for k, v in features.items() if k in _TRACKED}
        self._buffer.append(tracked)
        self._recompute_baseline()

        if track_id is not None:
            self._last_track = track_id
            track_buf = self._track_buffers.setdefault(track_id, deque(maxlen=self._buffer.maxlen))
            track_buf.append(tracked)
            if len(track_buf) >= self._min_baseline:
                self._recompute_track_baseline(track_id)

        if len(self._buffer) < self._min_baseline:
            return {}

        active_baseline = (
            self._track_baselines[track_id]
            if track_id is not None and track_id in self._track_baselines
            else self._baseline
        )

        zscores: dict[str, float] = {}
        any_alert = False
        for feat, val in tracked.items():
            mean, std = active_baseline.get(feat, (val, 1.0))
            z = (val - mean) / std if std > 1e-9 else 0.0
            zscores[feat] = round(z, 3)
            FEATURE_DRIFT_ZSCORE.labels(feature=feat).set(z)
            if abs(z) > self._alert_threshold:
                any_alert = True
                logger.warning(
                    "Feature drift: %s Z=%.2f (mean=%.4f std=%.4f obs=%.4f)",
                    feat, z, mean, std, val,
                )

        DRIFT_ALERT_ACTIVE.set(1.0 if any_alert else 0.0)
        if any_alert:
            self._maybe_trigger_retrain()
        elif self._prev_alert:
            self._on_drift_cleared()
        self._prev_alert = any_alert
        self._last_zscores = zscores
        import datetime
        self._last_updated = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        return zscores

    def _maybe_trigger_retrain(self) -> None:
        import json as _json
        try:
            if (
                _DRIFT_RETRAIN_STAMP.exists()
                and time.time() - _DRIFT_RETRAIN_STAMP.stat().st_mtime < _DRIFT_RETRAIN_COOLDOWN_S
            ):
                return
        except OSError:
            pass

        # Read 24h retrain history to detect concept drift before triggering another retrain.
        now = time.time()
        history: list[float] = []
        if _DRIFT_RETRAIN_HISTORY.exists():
            try:
                history = _json.loads(_DRIFT_RETRAIN_HISTORY.read_text())
            except Exception:
                pass
        recent = [ts for ts in history if ts > now - _CONCEPT_DRIFT_WINDOW_S]

        if len(recent) >= _CONCEPT_DRIFT_THRESHOLD:
            try:
                from f1di.observability.metrics import CONCEPT_DRIFT_SUSPECTED
                CONCEPT_DRIFT_SUSPECTED.set(1)
            except Exception:
                pass
            logger.warning(
                "concept_drift_suspected: %d drift-triggered retrains in 24h — "
                "baseline frozen until manual review. Likely cause: new circuit, "
                "compound generation, or regulation change. Check feature distributions "
                "before clearing the drift stamp.",
                len(recent),
            )
            # Freeze: do NOT recompute baseline from the drifted buffer. Reseeding here
            # would silently absorb a real distribution shift, masking it from operators.
            # A human should review and explicitly clear the stamp when the shift is understood.
            return

        # Record this retrain in the history, then trigger.
        recent.append(now)
        try:
            _DRIFT_RETRAIN_HISTORY.parent.mkdir(parents=True, exist_ok=True)
            _DRIFT_RETRAIN_HISTORY.write_text(_json.dumps(recent))
        except OSError:
            pass

        def _retrain() -> None:
            try:
                from f1di.confidence.online import retrain
                result = retrain()
                logger.info("drift_triggered_retrain result=%s", result)
            except Exception as exc:
                logger.warning("drift_triggered_retrain failed: %s", exc)
            try:
                # Also retrain classifiers if new labels have accumulated since
                # the last fit — safe because maybe_retrain_all checks the delta
                # threshold internally and is a no-op when nothing is new.
                from f1di.agents.auto_retrain import maybe_retrain_all
                maybe_retrain_all()
            except Exception as exc:
                logger.warning("drift_triggered_classifier_retrain failed: %s", exc)
            finally:
                try:
                    _DRIFT_RETRAIN_STAMP.parent.mkdir(parents=True, exist_ok=True)
                    _DRIFT_RETRAIN_STAMP.touch()
                except OSError:
                    pass

        threading.Thread(target=_retrain, daemon=True).start()
        logger.info("drift_alert_triggered_retrain: launching background calibration + classifier retrain")

    def _on_drift_cleared(self) -> None:
        """Called when drift alert transitions from active to clear. Resets concept drift state."""
        import json as _json
        try:
            from f1di.observability.metrics import CONCEPT_DRIFT_SUSPECTED
            CONCEPT_DRIFT_SUSPECTED.set(0)
        except Exception:
            pass
        try:
            _DRIFT_RETRAIN_HISTORY.write_text(_json.dumps([]))
        except OSError:
            pass
        logger.info("drift_cleared: concept drift history reset")

    def _recompute_baseline(self) -> None:
        if len(self._buffer) < 2:
            return
        for feat in _TRACKED:
            vals = [obs[feat] for obs in self._buffer if feat in obs]
            if len(vals) < 2:
                continue
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            self._baseline[feat] = (mean, math.sqrt(variance))

    def _recompute_track_baseline(self, track_id: str) -> None:
        buf = self._track_buffers.get(track_id)
        if not buf or len(buf) < self._min_baseline:
            return
        baseline: dict[str, tuple[float, float]] = {}
        for feat in _TRACKED:
            vals = [obs[feat] for obs in buf if feat in obs]
            if len(vals) < 2:
                continue
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            baseline[feat] = (mean, math.sqrt(variance))
        self._track_baselines[track_id] = baseline

    @property
    def ready(self) -> bool:
        return len(self._buffer) >= self._min_baseline

    @property
    def baseline_size(self) -> int:
        return len(self._buffer)

    def status(self) -> dict:
        alerted = [f for f, z in self._last_zscores.items() if abs(z) > self._alert_threshold]
        return {
            "ready": self.ready,
            "baseline_size": self.baseline_size,
            "min_baseline": self._min_baseline,
            "alert_threshold": self._alert_threshold,
            "last_updated": self._last_updated,
            "alerted_features": alerted,
            "track_baselines": list(self._track_baselines.keys()),
            "active_track": self._last_track,
            "features": {
                feat: {
                    "z_score": z,
                    "alerted": abs(z) > self._alert_threshold,
                    "mean": round(self._baseline[feat][0], 4) if feat in self._baseline else None,
                    "std": round(self._baseline[feat][1], 4) if feat in self._baseline else None,
                }
                for feat, z in self._last_zscores.items()
            },
        }

    def seed_from_db(self, limit: int = 200) -> int:
        """Seed baseline from stored telemetry rows. Returns count seeded.

        Rows are sorted by (session_id, driver_id, lap) and grouped so that
        per-lap slopes (wear, SoC) can be computed from consecutive observations
        within the same driver stint.
        """
        try:
            from collections import defaultdict as _dd
            from f1di.storage.database import db_session
            from f1di.storage.models import TelemetrySampleRecord
            from sqlalchemy import select
            with db_session() as session:
                rows = list(session.scalars(
                    select(TelemetrySampleRecord)
                    .order_by(
                        TelemetrySampleRecord.session_id,
                        TelemetrySampleRecord.driver_id,
                        TelemetrySampleRecord.lap,
                    )
                    .limit(limit)
                ))

            # Group and sort by (session_id, driver_id) to compute slopes.
            groups: dict = _dd(list)
            for row in rows:
                groups[(row.session_id, row.driver_id)].append(row)

            for group_rows in groups.values():
                group_rows.sort(key=lambda r: r.lap)
                for i, row in enumerate(group_rows):
                    if i > 0:
                        prev = group_rows[i - 1]
                        delta_lap = max(row.lap - prev.lap, 1)
                        fl_slope = (row.tire_wear_fl - prev.tire_wear_fl) / delta_lap
                        fr_slope = (row.tire_wear_fr - prev.tire_wear_fr) / delta_lap
                        rear_now = (row.tire_wear_rl + row.tire_wear_rr) / 2.0
                        rear_prev = (prev.tire_wear_rl + prev.tire_wear_rr) / 2.0
                        rear_slope = (rear_now - rear_prev) / delta_lap
                        soc_slope = (row.battery_soc - prev.battery_soc) / delta_lap
                    else:
                        fl_slope = fr_slope = rear_slope = soc_slope = 0.0

                    obs = {
                        "fl_wear": row.tire_wear_fl,
                        "fr_wear": row.tire_wear_fr,
                        "rear_wear_mean": (row.tire_wear_rl + row.tire_wear_rr) / 2.0,
                        "fl_wear_slope": fl_slope,
                        "fr_wear_slope": fr_slope,
                        "rear_wear_slope": rear_slope,
                        "battery_soc": row.battery_soc,
                        "battery_soc_slope": soc_slope,
                        "rain_intensity": row.rain_intensity,
                        "grip_estimate": row.grip_estimate,
                        "mean_speed_kph": row.speed_kph,
                    }
                    self._buffer.append(obs)
                    if row.track_id:
                        track_buf = self._track_buffers.setdefault(row.track_id, deque(maxlen=self._buffer.maxlen))
                        track_buf.append(obs)

            self._recompute_baseline()
            for track_id in list(self._track_buffers):
                if len(self._track_buffers[track_id]) >= self._min_baseline:
                    self._recompute_track_baseline(track_id)
            # Populate _last_zscores from the most recent row so status() shows
            # real values immediately instead of empty {} until the first update().
            if self._buffer and self._baseline:
                last = self._buffer[-1]
                for feat, val in last.items():
                    if feat in self._baseline:
                        mean, std = self._baseline[feat]
                        z = (val - mean) / std if std > 1e-9 else 0.0
                        self._last_zscores[feat] = round(z, 3)
                import datetime
                self._last_updated = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            logger.info("Drift tracker seeded with %d telemetry rows from DB", len(rows))
            return len(rows)
        except Exception as exc:
            logger.warning("Drift seed_from_db failed: %s", exc)
            return 0


def features_as_dict(features) -> dict[str, float]:
    """Convert a RaceFeatures dataclass to a plain float dict."""
    return {k: float(v) for k, v in asdict(features).items() if isinstance(v, (int, float))}


_TRACKER: FeatureDriftTracker | None = None


def get_tracker() -> FeatureDriftTracker:
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = FeatureDriftTracker()
    return _TRACKER
