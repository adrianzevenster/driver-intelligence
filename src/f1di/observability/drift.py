from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import asdict

logger = logging.getLogger("f1di.observability.drift")

_ALERT_THRESHOLD = 3.5
_MIN_BASELINE = 50
_BUFFER_SIZE = 200

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
        self._last_zscores = zscores
        import datetime
        self._last_updated = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        return zscores

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
        """Seed baseline from stored telemetry rows. Returns count seeded."""
        try:
            from f1di.storage.database import db_session
            from f1di.storage.models import TelemetrySampleRecord
            from sqlalchemy import select
            with db_session() as session:
                rows = list(session.scalars(
                    select(TelemetrySampleRecord)
                    .order_by(TelemetrySampleRecord.created_at.desc())
                    .limit(limit)
                ))
            for row in rows:
                obs = {
                    "fl_wear": row.tire_wear_fl,
                    "fr_wear": row.tire_wear_fr,
                    "rear_wear_mean": (row.tire_wear_rl + row.tire_wear_rr) / 2,
                    "battery_soc": row.battery_soc,
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
