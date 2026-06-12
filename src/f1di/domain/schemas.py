from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field, computed_field


class Compound(str, Enum):
    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET = "WET"


class InsightAudience(str, Enum):
    DRIVER = "DRIVER"
    ENGINEER = "ENGINEER"
    STRATEGY = "STRATEGY"
    REPLAY = "REPLAY"


class RiskLevel(str, Enum):
    INFO = "INFO"
    WATCH = "WATCH"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class TelemetrySample(BaseModel):
    session_id: str
    driver_id: str
    track_id: str
    timestamp_ms: int
    lap: int = Field(ge=0)
    sector: int = Field(ge=1, le=3)
    distance_m: float = Field(ge=0)
    corner_id: str | None = None

    speed_kph: float = Field(ge=0)
    acceleration_g: float
    throttle_pct: float = Field(ge=0, le=100)
    brake_pressure_bar: float = Field(ge=0)
    steering_angle_deg: float
    yaw_rate_deg_s: float
    slip_angle_deg: float
    wheel_speed_fl: float
    wheel_speed_fr: float
    wheel_speed_rl: float
    wheel_speed_rr: float

    compound: Compound
    stint_lap: int = Field(ge=0)
    tire_temp_fl_c: float
    tire_temp_fr_c: float
    tire_temp_rl_c: float
    tire_temp_rr_c: float
    tire_wear_fl: float = Field(ge=0, le=1)
    tire_wear_fr: float = Field(ge=0, le=1)
    tire_wear_rl: float = Field(ge=0, le=1)
    tire_wear_rr: float = Field(ge=0, le=1)
    grip_estimate: float = Field(ge=0, le=1)
    lockup_event: bool = False

    battery_soc: float = Field(ge=0, le=1)
    ers_deploy_kw: float = Field(ge=0)
    ers_regen_kw: float = Field(ge=0)
    pu_thermal_state: float = Field(ge=0, le=1)

    track_temp_c: float
    ambient_temp_c: float
    humidity_pct: float = Field(ge=0, le=100)
    wind_speed_kph: float = Field(ge=0)
    wind_direction_deg: float = Field(ge=0, le=360)
    rain_intensity: float = Field(ge=0, le=1)
    evolving_grip: float = Field(ge=0, le=1)

    brake_temp_fl_c: float
    brake_temp_fr_c: float
    brake_temp_rl_c: float
    brake_temp_rr_c: float


class TelemetryWindow(BaseModel):
    session_id: str
    driver_id: str
    track_id: str
    samples: list[TelemetrySample]

    @computed_field
    @property
    def latest(self) -> TelemetrySample:
        if not self.samples:
            raise ValueError("TelemetryWindow must contain at least one sample")
        return self.samples[-1]


class RetrievedEvidence(BaseModel):
    source_id: str
    title: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentFinding(BaseModel):
    agent: str
    risk: RiskLevel
    summary: str
    confidence: float = Field(ge=0, le=1)
    evidence: list[RetrievedEvidence] = Field(default_factory=list)
    features: dict[str, float | str | bool] = Field(default_factory=dict)


class PredictionPoint(BaseModel):
    lap: int
    p10_time_s: float | None = None
    p50_time_s: float | None = None
    p90_time_s: float | None = None
    wear_fl: float | None = None
    wear_fr: float | None = None
    grip: float | None = None


class RaceProjection(BaseModel):
    session_id: str
    driver_id: str
    track_id: str
    current_lap: int
    remaining_laps: int
    projections: list[PredictionPoint]
    summary: str
    confidence: float = Field(ge=0, le=1)
    latency_ms: float


class StrategyScenario(BaseModel):
    label: str
    pit_lap: int | None
    total_time_s: float
    delta_s: float
    cliff_lap: int | None
    end_wear_fl: float
    recommended: bool


class StrategyComparison(BaseModel):
    session_id: str
    driver_id: str
    track_id: str
    current_lap: int
    remaining_laps: int
    scenarios: list[StrategyScenario]
    recommendation: str
    latency_ms: float


class DriverInsight(BaseModel):
    insight_id: str
    session_id: str
    driver_id: str
    audience: InsightAudience
    risk: RiskLevel
    recommendation: str
    confidence: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    raw_score: float | None = None
    supporting_factors: list[str]
    evidence: list[RetrievedEvidence]
    findings: list[AgentFinding]
    policy: Literal["SHOW", "ENGINEER_ONLY", "SUPPRESS"]
    latency_ms: float
