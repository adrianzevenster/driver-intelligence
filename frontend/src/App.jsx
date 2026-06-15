import React, { useState, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity, ChevronDown, ChevronRight, Gauge, ShieldAlert,
  Cpu, Database, MessageSquare, BarChart2, Send, User, Bot,
  RefreshCw, BookOpen, Radio, History, FlaskConical,
  ThumbsUp, ThumbsDown, TrendingUp, Upload, Table, Search,
  Mic, LineChart, FileText,
  Bell, Mail, Plus, X, Settings, Clock, Play, Workflow,
  Flag, Fuel, Zap, Cloud,
} from 'lucide-react';
import './style.css';

// ── Constants ──────────────────────────────────────────────────────────────

const RISK_META = {
  INFO:     { color: '#64748b', bg: '#0f1824', order: 0 },
  WATCH:    { color: '#f59e0b', bg: '#1c1505', order: 1 },
  WARNING:  { color: '#f97316', bg: '#1e0f04', order: 2 },
  CRITICAL: { color: '#ef4444', bg: '#1e0707', order: 3 },
};
const AGENT_LABELS = {
  telemetry:    'Telemetry',
  tire_strategy:'Tire Strategy',
  weather:      'Weather',
  battery:      'Battery / ERS',
  safety_car:   'Safety Car / VSC',
  fuel:         'Fuel Strategy',
};
const AGENT_ICON = {
  telemetry:    { Icon: Cpu,         color: '#38bdf8' },
  tire_strategy:{ Icon: Activity,    color: '#a78bfa' },
  weather:      { Icon: Cloud,       color: '#64748b' },
  battery:      { Icon: Zap,         color: '#4ade80' },
  safety_car:   { Icon: Flag,        color: '#fbbf24' },
  fuel:         { Icon: Fuel,        color: '#f97316' },
};
const COMPOUNDS     = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'];
const COMPOUND_COLOR = { SOFT:'#ef4444', MEDIUM:'#eab308', HARD:'#cbd5e1', INTERMEDIATE:'#22c55e', WET:'#3b82f6' };
const DRIVERS = [
  { code: 'VER', name: 'Max Verstappen' },
  { code: 'NOR', name: 'Lando Norris' },
  { code: 'LEC', name: 'Charles Leclerc' },
  { code: 'PIA', name: 'Oscar Piastri' },
  { code: 'SAI', name: 'Carlos Sainz' },
  { code: 'RUS', name: 'George Russell' },
  { code: 'HAM', name: 'Lewis Hamilton' },
  { code: 'ANT', name: 'Kimi Antonelli' },
  { code: 'ALO', name: 'Fernando Alonso' },
  { code: 'STR', name: 'Lance Stroll' },
  { code: 'TSU', name: 'Yuki Tsunoda' },
  { code: 'LAW', name: 'Liam Lawson' },
  { code: 'GAS', name: 'Pierre Gasly' },
  { code: 'DOO', name: 'Jack Doohan' },
  { code: 'HUL', name: 'Nico Hülkenberg' },
  { code: 'BEA', name: 'Oliver Bearman' },
  { code: 'ALB', name: 'Alex Albon' },
  { code: 'COL', name: 'Franco Colapinto' },
  { code: 'MAG', name: 'Kevin Magnussen' },
  { code: 'BOT', name: 'Valtteri Bottas' },
  { code: 'ZHO', name: 'Guanyu Zhou' },
  { code: 'RIC', name: 'Daniel Ricciardo' },
  { code: 'OCO', name: 'Esteban Ocon' },
  { code: 'PER', name: 'Sergio Perez' },
];

const TRACKS        = [
  'bahrain', 'jeddah', 'melbourne', 'shanghai', 'miami',
  'imola', 'monaco', 'barcelona', 'montreal', 'spielberg',
  'silverstone', 'budapest', 'spa', 'zandvoort', 'monza',
  'baku', 'singapore', 'austin', 'mexico_city', 'interlagos',
  'las_vegas', 'lusail', 'abu_dhabi', 'suzuka',
];
const SKIP_KEYS     = new Set(['lap', 'sector', 'lockup_count']);
const GATE_META = {
  pass_case_recall:            'Case Recall',
  pass_nominal_false_positive: 'No False Pos.',
  pass_agent_activation:       'Agent Activation',
  pass_evidence:               'Evidence',
  pass_expected_sources:       'Source Retrieval',
  pass_policy_correctness:     'Policy',
};
const LLM_META = {
  rules:             { label: 'Rules Engine',  color: '#64748b' },
  anthropic:         { label: 'Claude',         color: '#a78bfa' },
  openai_compatible: { label: 'Open Source',    color: '#34d399' },
};
const VEC_META = {
  memory:   { label: 'In-memory', color: '#64748b' },
  qdrant:   { label: 'Qdrant',    color: '#38bdf8' },
  pgvector: { label: 'pgvector',  color: '#818cf8' },
};

// ── Build TelemetryWindow ──────────────────────────────────────────────────

function buildWindow(s) {
  const N = 12, samples = [];
  const wearRate = s.stintLap > 0 ? s.flWear / (s.stintLap * 30) : 0.002;
  for (let i = 0; i < N; i++) {
    const t = i / (N - 1), phase = 0.25 + t * 0.20;
    const sector = Math.min(3, Math.floor(phase * 3) + 1);
    const braking = (phase % 0.18) < 0.035;
    const speed = Math.max(60, 265 + Math.sin(phase * Math.PI * 4) * 35);
    const steer = 6 + Math.sin(phase * Math.PI * 4) * 18;
    const back = N - 1 - i;
    const scale = v => Math.min(0.99, Math.max(0, v - wearRate * back * (v / Math.max(s.flWear, 0.01))));
    const flW = Math.min(0.99, Math.max(0, s.flWear - wearRate * back));
    const soc = Math.min(0.95, Math.max(0.05, s.batterySoc + back * 0.001));
    const isLockup = s.lockupEvent && i >= N - 3;
    const brakeTempFl = s.highBrakeTemps ? 700 + t * 260 : 370 + t * 55;
    const tireTemp = 88 + flW * 38 + (braking ? 9 : 0);
    samples.push({
      session_id: s.sessionId, driver_id: s.driverId, track_id: s.trackId,
      timestamp_ms: i * 2800, lap: s.lap, sector, distance_m: 5891 * (s.lap - 1 + phase),
      corner_id: `T${Math.floor(phase * 18) + 1}`,
      speed_kph: speed, acceleration_g: braking ? -0.8 : 0.3,
      throttle_pct: braking ? 10 : 76, brake_pressure_bar: braking ? 110 : 8,
      steering_angle_deg: steer, yaw_rate_deg_s: steer * speed / 200,
      slip_angle_deg: Math.abs(steer) / 18,
      wheel_speed_fl: speed * (isLockup ? 0.97 : 1.0), wheel_speed_fr: speed * (isLockup ? 0.975 : 1.0),
      wheel_speed_rl: speed, wheel_speed_rr: speed,
      compound: s.compound, stint_lap: s.stintLap,
      tire_temp_fl_c: tireTemp, tire_temp_fr_c: tireTemp - 2,
      tire_temp_rl_c: tireTemp - 4, tire_temp_rr_c: tireTemp - 5,
      tire_wear_fl: flW, tire_wear_fr: scale(s.frWear), tire_wear_rl: scale(s.rlWear), tire_wear_rr: scale(s.rrWear),
      grip_estimate: s.grip, lockup_event: isLockup,
      battery_soc: soc, ers_deploy_kw: speed > 255 ? 120 : 25, ers_regen_kw: braking ? 68 : 5,
      pu_thermal_state: Math.min(1.0, 0.45 + 76 / 200),
      track_temp_c: s.trackTemp, ambient_temp_c: 23, humidity_pct: 55 + s.rainIntensity * 30,
      wind_speed_kph: s.windSpeed, wind_direction_deg: 245, rain_intensity: s.rainIntensity,
      evolving_grip: Math.max(0.4, 0.88 - s.rainIntensity * 0.4),
      brake_temp_fl_c: brakeTempFl, brake_temp_fr_c: brakeTempFl - 8,
      brake_temp_rl_c: 335 + t * 20, brake_temp_rr_c: 330 + t * 20,
    });
  }
  return { session_id: s.sessionId, driver_id: s.driverId, track_id: s.trackId, samples };
}

// ── Shared primitives ──────────────────────────────────────────────────────

function wearColor(v) {
  return v > 0.70 ? '#ef4444' : v > 0.50 ? '#f97316' : v > 0.30 ? '#f59e0b' : '#22c55e';
}

function RangeRow({ label, value, onChange, min = 0, max = 1, step = 0.01, fmt, colorFn }) {
  const pct = ((value - min) / (max - min)) * 100;
  const col = colorFn ? colorFn(value) : '#3b82f6';
  return (
    <div className="range-row">
      <span className="range-label">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))} className="range-input"
        style={{ background: `linear-gradient(to right, ${col} ${pct}%, #1a2540 ${pct}%)` }} />
      <span className="range-val" style={{ color: col }}>{fmt ? fmt(value) : value}</span>
    </div>
  );
}

function WearGrid({ fl, fr, rl, rr, onChange }) {
  const row = (label, key, val) => (
    <RangeRow label={label} value={val} onChange={v => onChange(key, v)}
      fmt={v => `${Math.round(v * 100)}%`} colorFn={wearColor} />
  );
  return (
    <div className="wear-grid">
      <div>{row('FL', 'flWear', fl)}{row('RL', 'rlWear', rl)}</div>
      <div>{row('FR', 'frWear', fr)}{row('RR', 'rrWear', rr)}</div>
    </div>
  );
}

function Section({ title, children }) {
  return (
    <div className="form-section">
      <div className="form-section-title">{title}</div>
      {children}
    </div>
  );
}

function BackendBadges({ version }) {
  if (!version) return null;
  const llm = LLM_META[version.model_backend] ?? { label: version.model_backend, color: '#64748b' };
  const vec = VEC_META[version.vector_backend] ?? { label: version.vector_backend, color: '#64748b' };
  const modelLabel = version.model_backend === 'anthropic'
    ? (version.llm_advice_model ?? llm.label)
    : version.model_backend === 'openai_compatible'
      ? (version.llm_open_source_model ?? llm.label)
      : llm.label;
  return (
    <div className="backend-badges">
      <div className="backend-badge"><Cpu size={11} /><span style={{ color: llm.color }}>{modelLabel}</span></div>
      <div className="backend-badge"><Database size={11} /><span style={{ color: vec.color }}>{vec.label}</span></div>
    </div>
  );
}

// ── Stats form ─────────────────────────────────────────────────────────────

const DEFAULT_STATS = {
  driverId: 'VER', sessionId: 'live-race', trackId: 'silverstone',
  lap: 12, compound: 'MEDIUM', stintLap: 8,
  flWear: 0.48, frWear: 0.43, rlWear: 0.37, rrWear: 0.35, grip: 0.76,
  batterySoc: 0.64, rainIntensity: 0.0, windSpeed: 14, trackTemp: 38,
  lockupEvent: false, highBrakeTemps: false,
};

function StatsForm({ stats, onChange }) {
  const set = (k, v) => onChange({ ...stats, [k]: v });
  return (
    <div className="stats-form">
      <Section title="Race Context">
        <div className="context-row">
          <label><span>Driver</span>
            <select className="text-input" value={stats.driverId}
              onChange={e => set('driverId', e.target.value)}>
              {DRIVERS.map(d => (
                <option key={d.code} value={d.code}>{d.name}</option>
              ))}
            </select>
          </label>
          <label><span>Track</span>
            <select className="text-input" value={stats.trackId} onChange={e => set('trackId', e.target.value)}>
              {TRACKS.map(t => <option key={t} value={t}>{t.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</option>)}
            </select>
          </label>
          <label><span>Lap</span>
            <input type="number" className="text-input num-input" min={1} max={80} value={stats.lap}
              onChange={e => set('lap', Number(e.target.value))} />
          </label>
        </div>
      </Section>

      <Section title="Tire State">
        <div className="compound-row">
          <div className="compound-btns">
            {COMPOUNDS.map(c => (
              <button key={c} className={`compound-btn${stats.compound === c ? ' active' : ''}`}
                style={stats.compound === c ? { background: COMPOUND_COLOR[c]+'33', borderColor: COMPOUND_COLOR[c], color: COMPOUND_COLOR[c] } : {}}
                onClick={() => set('compound', c)}>{c[0]}</button>
            ))}
          </div>
          <span className="compound-label" style={{ color: COMPOUND_COLOR[stats.compound] }}>{stats.compound}</span>
          <label className="stint-label">
            <span>Age</span>
            <input type="number" className="text-input num-input" min={1} max={50} value={stats.stintLap}
              onChange={e => set('stintLap', Number(e.target.value))} />
            <span className="unit">laps</span>
          </label>
        </div>
        <WearGrid fl={stats.flWear} fr={stats.frWear} rl={stats.rlWear} rr={stats.rrWear}
          onChange={(k, v) => set(k, v)} />
        <RangeRow label="Grip" value={stats.grip} onChange={v => set('grip', v)}
          fmt={v => `${Math.round(v * 100)}%`}
          colorFn={v => v < 0.50 ? '#ef4444' : v < 0.70 ? '#f59e0b' : '#22c55e'} />
      </Section>

      <Section title="Power Unit">
        <RangeRow label="Battery SOC" value={stats.batterySoc} onChange={v => set('batterySoc', v)}
          fmt={v => `${Math.round(v * 100)}%`}
          colorFn={v => v < 0.25 ? '#ef4444' : v < 0.45 ? '#f59e0b' : '#22c55e'} />
      </Section>

      <Section title="Track Conditions">
        <RangeRow label="Rain" value={stats.rainIntensity} onChange={v => set('rainIntensity', v)}
          fmt={v => `${Math.round(v * 100)}%`}
          colorFn={v => v > 0.35 ? '#3b82f6' : v > 0.12 ? '#60a5fa' : '#64748b'} />
        <RangeRow label="Wind" value={stats.windSpeed} onChange={v => set('windSpeed', v)}
          min={0} max={80} step={1} fmt={v => `${v} km/h`}
          colorFn={v => v > 40 ? '#f97316' : '#64748b'} />
        <RangeRow label="Track temp" value={stats.trackTemp} onChange={v => set('trackTemp', v)}
          min={10} max={65} step={1} fmt={v => `${v}°C`}
          colorFn={v => v > 50 ? '#ef4444' : v > 38 ? '#f97316' : '#64748b'} />
      </Section>

      <Section title="Events this window">
        <div className="events-row">
          <label className="check-label">
            <input type="checkbox" checked={stats.lockupEvent} onChange={e => set('lockupEvent', e.target.checked)} />
            <span>Lockup detected</span>
          </label>
          <label className="check-label">
            <input type="checkbox" checked={stats.highBrakeTemps} onChange={e => set('highBrakeTemps', e.target.checked)} />
            <span>High brake temps</span>
          </label>
        </div>
      </Section>
    </div>
  );
}

// ── Insight panel ──────────────────────────────────────────────────────────

function rm(risk) { return RISK_META[risk] ?? RISK_META.INFO; }
function topFinding(findings) {
  return sortFindings(findings)[0];
}
function sortFindings(findings) {
  return [...findings].sort((a, b) => {
    // Safety car CRITICAL always sorts first — most time-critical call
    const scA = a.agent === 'safety_car' && a.risk === 'CRITICAL' ? 1 : 0;
    const scB = b.agent === 'safety_car' && b.risk === 'CRITICAL' ? 1 : 0;
    if (scB !== scA) return scB - scA;
    return rm(b.risk).order - rm(a.risk).order;
  });
}

function RiskPill({ risk, policy }) {
  const m = rm(risk);
  return (
    <span className="risk-pill" style={{ color: m.color, background: m.bg, borderColor: m.color+'55' }}>
      {risk}
      {policy === 'SUPPRESS'      && <span className="policy-tag suppressed">SUPPRESSED</span>}
      {policy === 'ENGINEER_ONLY' && <span className="policy-tag eng-only">ENG ONLY</span>}
    </span>
  );
}

function ConfidenceBar({ confidence, uncertainty }) {
  const pct = confidence * 100;
  const color = confidence >= 0.75 ? '#ef4444' : confidence >= 0.55 ? '#f97316' : confidence >= 0.35 ? '#f59e0b' : '#64748b';
  return (
    <div className="conf-wrap">
      <div className="conf-labels">
        <span>Confidence <strong style={{ color }}>{pct.toFixed(1)}%</strong></span>
        <span className="muted">Uncertainty {(uncertainty * 100).toFixed(1)}%</span>
      </div>
      <div className="conf-track" style={{ background: `linear-gradient(to right, ${color} ${pct}%, #1a2540 ${pct}%)` }} />
    </div>
  );
}

function ClassProbBar({ classProbs, activeRisk }) {
  const ORDER = ['INFO', 'WATCH', 'WARNING', 'CRITICAL'];
  const pairs = ORDER.filter(k => k in classProbs).map(k => [k, classProbs[k]]);
  if (pairs.length === 0) return null;
  return (
    <div style={{ marginTop: 6 }}>
      <div style={{ display: 'flex', borderRadius: 3, overflow: 'hidden', height: 5 }}>
        {pairs.map(([label, p]) => (
          <div key={label} style={{
            width: `${(p * 100).toFixed(1)}%`,
            background: rm(label).color,
            opacity: label === activeRisk ? 1 : 0.35,
          }} />
        ))}
      </div>
      <div style={{ display: 'flex', gap: 8, marginTop: 3, flexWrap: 'wrap' }}>
        {pairs.map(([label, p]) => (
          <span key={label} style={{
            fontSize: 9, fontFamily: 'monospace',
            color: label === activeRisk ? rm(label).color : '#475569',
            fontWeight: label === activeRisk ? 700 : 400,
          }}>
            {label} {(p * 100).toFixed(0)}%
          </span>
        ))}
      </div>
    </div>
  );
}

function AgentCard({ finding, isTop, expanded, onToggle }) {
  const m = rm(finding.risk);
  const confPct = (finding.confidence * 100).toFixed(0);
  const entries = Object.entries(finding.features ?? {}).filter(([k, v]) => !SKIP_KEYS.has(k) && typeof v === 'number');
  const preview = entries.slice(0, 3);
  const hasProbs = finding.class_probabilities && Object.keys(finding.class_probabilities).length > 0;
  const agentMeta = AGENT_ICON[finding.agent];
  const isSCCritical = finding.agent === 'safety_car' && finding.risk === 'CRITICAL';
  const isSCWarning  = finding.agent === 'safety_car' && finding.risk === 'WARNING';
  const borderColor  = isSCCritical ? '#fbbf24' : isSCWarning ? '#fbbf2466' : m.color + (isTop ? 'aa' : '44');
  return (
    <div className={`agent-card${isTop ? ' agent-top' : ''}`} style={{ borderColor }}>
      {isSCCritical && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '4px 10px', marginBottom: 2,
          background: 'rgba(251,191,36,0.12)', borderBottom: '1px solid rgba(251,191,36,0.3)',
          borderRadius: '6px 6px 0 0',
        }}>
          <Flag size={11} color="#fbbf24" />
          <span style={{ fontSize: 10, color: '#fbbf24', fontWeight: 700, letterSpacing: '0.06em' }}>
            SAFETY CAR DEPLOYED — PIT WINDOW OPEN
          </span>
        </div>
      )}
      <button className="agent-header" onClick={onToggle}>
        <div className="agent-name-row">
          {isTop && <span className="top-dot" style={{ background: isSCCritical ? '#fbbf24' : m.color }} />}
          {agentMeta && (
            <agentMeta.Icon size={11} color={agentMeta.color} style={{ marginRight: 4, flexShrink: 0 }} />
          )}
          <span className="agent-name">{AGENT_LABELS[finding.agent] ?? finding.agent}</span>
          {finding.clf_source && (
            <span style={{
              fontSize: 8, padding: '1px 4px', borderRadius: 3, marginLeft: 4,
              background: '#0f2418', color: '#4ade80', border: '1px solid #166534',
              fontFamily: 'monospace', letterSpacing: '0.02em',
            }}>LR</span>
          )}
        </div>
        <div className="agent-header-right">
          <span className="agent-risk" style={{ color: m.color }}>{finding.risk}</span>
          <span className="agent-conf-pct" style={{ color: m.color }}>{confPct}%</span>
          {expanded ? <ChevronDown size={12} color="#64748b" /> : <ChevronRight size={12} color="#64748b" />}
        </div>
      </button>
      <div className="agent-conf-track">
        <div className="agent-conf-fill" style={{ width: `${confPct}%`, background: m.color }} />
      </div>
      {hasProbs && (
        <ClassProbBar classProbs={finding.class_probabilities} activeRisk={finding.risk} />
      )}
      {finding.ood_flagged && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 5, marginTop: 5,
          padding: '3px 7px', borderRadius: 4,
          background: 'rgba(249,115,22,0.08)', border: '1px solid rgba(249,115,22,0.3)',
        }}>
          <span style={{ fontSize: 9, color: '#f97316', fontFamily: 'monospace', fontWeight: 700 }}>OOD</span>
          <span style={{ fontSize: 9, color: '#94a3b8' }}>
            z={finding.ood_score?.toFixed(1)} — features outside training distribution, confidence penalised
          </span>
        </div>
      )}
      <p className="agent-summary">{finding.summary}</p>
      {preview.length > 0 && (
        <div className="chip-row">
          {preview.map(([k, v]) => <span key={k} className="chip">{k.replace(/_/g, ' ')}: {v.toFixed(3)}</span>)}
        </div>
      )}
      {expanded && entries.length > 0 && (
        <div className="feat-table">
          {entries.map(([k, v]) => (
            <div key={k} className={`feat-row${preview.find(([pk]) => pk === k) ? ' feat-highlight' : ''}`}>
              <span className="feat-key">{k.replace(/_/g, ' ')}</span>
              <span className="feat-val">{v.toFixed(4)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function InsightPanel({ insight, modelBackend }) {
  const [expanded, setExpanded] = useState(new Set());
  const top = topFinding(insight.findings);
  const lapNum = insight.findings?.[0]?.features?.lap;
  const isAI = modelBackend && modelBackend !== 'rules';
  const llm = LLM_META[modelBackend] ?? null;

  const toggle = agent => setExpanded(prev => {
    const s = new Set(prev);
    s.has(agent) ? s.delete(agent) : s.add(agent);
    return s;
  });

  return (
    <>
      <div className="insight-top">
        <RiskPill risk={insight.risk} policy={insight.policy} />
        <span className="meta-badge">{lapNum != null ? `Lap ${lapNum} · ` : ''}{insight.driver_id}</span>
      </div>
      {insight.policy !== 'SHOW' && (
        <div className={`policy-banner ${insight.policy === 'SUPPRESS' ? 'suppress' : 'eng'}`}>
          {insight.policy === 'SUPPRESS'
            ? 'Insight suppressed — confidence below threshold for this audience.'
            : 'Restricted to engineers — below driver display threshold.'}
        </div>
      )}
      <div className={`rec-block${isAI ? ' rec-ai' : ''}`}>
        <div className="rec-source">
          <span className="rec-dot" style={{ background: rm(top?.risk).color }} />
          {AGENT_LABELS[top?.agent] ?? top?.agent}
          {isAI && llm && (
            <span className="rec-ai-badge" style={{ color: llm.color, borderColor: llm.color+'55' }}>{llm.label}</span>
          )}
        </div>
        <p className="recommendation">{insight.recommendation}</p>
      </div>
      <ConfidenceBar confidence={insight.confidence} uncertainty={insight.uncertainty} />
      <h4>Agent Findings</h4>
      <div className="agent-grid">
        {sortFindings(insight.findings).map(f => (
          <AgentCard key={f.agent} finding={f} isTop={f.agent === top?.agent}
            expanded={expanded.has(f.agent)} onToggle={() => toggle(f.agent)} />
        ))}
      </div>
      {insight.supporting_factors?.length > 0 && (
        <><h4>Supporting Factors</h4>
          <ul className="factor-list">
            {insight.supporting_factors.map((f, i) => <li key={i}>{f}</li>)}
          </ul>
        </>
      )}
      {insight.evidence?.length > 0 && (
        <><h4>Evidence <span className="muted">({insight.evidence.length} sources)</span></h4>
          <ul className="ev-list">
            {insight.evidence.map(e => (
              <li key={e.source_id} className="ev-item">
                <div className="ev-header">
                  <span className="ev-title">{e.title}</span>
                  <span className="ev-score">{Math.round(e.score * 100)}%</span>
                </div>
                <p className="ev-text">{e.text.slice(0, 220)}…</p>
              </li>
            ))}
          </ul>
        </>
      )}
      <FeedbackWidget insightId={insight.insight_id} />
      <JudgeScoreWidget insightId={insight.insight_id} />
      <p className="footer-line">{insight.latency_ms.toFixed(1)} ms · session {insight.session_id}</p>
    </>
  );
}

function FeedbackWidget({ insightId }) {
  const [sent, setSent]       = useState(false);
  const [rating, setRating]   = useState(0);
  const [correct, setCorrect] = useState(null);

  async function submit() {
    await fetch('/api/v1/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ insight_id: insightId, rating: rating || 3, correct }),
    }).catch(() => {});
    setSent(true);
  }

  if (sent) return <p style={{ fontSize: 11, color: 'var(--muted)', margin: '6px 0 0' }}>Feedback recorded.</p>;

  const canSubmit = correct !== null || rating > 0;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 8, flexWrap: 'wrap' }}>
      <span style={{ fontSize: 11, color: 'var(--muted)' }}>Accurate?</span>
      {[true, false].map(v => (
        <button key={String(v)} onClick={() => setCorrect(v)}
          style={{ background: correct === v ? (v ? '#166534' : '#7f1d1d') : 'var(--card-border)',
                   border: 'none', borderRadius: 4, padding: '2px 6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 3 }}>
          {v ? <ThumbsUp size={11} /> : <ThumbsDown size={11} />}
        </button>
      ))}
      <span style={{ color: 'var(--muted)', fontSize: 11 }}>·</span>
      {[1,2,3,4,5].map(n => (
        <button key={n} onClick={() => setRating(n)}
          style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '0 1px',
                   color: rating >= n ? '#f59e0b' : 'var(--muted)', fontSize: 13, lineHeight: 1 }}>★</button>
      ))}
      {canSubmit && (
        <button onClick={submit}
          style={{ background: 'var(--accent)', border: 'none', borderRadius: 4, padding: '2px 8px',
                   cursor: 'pointer', fontSize: 11, color: '#fff', marginLeft: 4 }}>
          Send
        </button>
      )}
    </div>
  );
}

// ── LLM Judge score widget ────────────────────────────────────────────────

const JUDGE_DIMS = [
  { key: 'safety',        label: 'Safety',        color: '#ef4444' },
  { key: 'actionability', label: 'Actionability',  color: '#22c55e' },
  { key: 'register',      label: 'Register',       color: '#38bdf8' },
  { key: 'calibration',   label: 'Calibration',    color: '#a78bfa' },
];

function JudgeScoreWidget({ insightId }) {
  const [score, setScore]       = useState(null);
  const [pending, setPending]   = useState(true);
  const retriesRef               = useRef(0);

  useEffect(() => {
    if (!insightId) return;
    let cancelled = false;

    async function poll() {
      if (cancelled) return;
      const res = await fetch(`/api/v1/insights/${insightId}/judge`).catch(() => null);
      if (cancelled) return;
      if (res && res.ok) {
        setScore(await res.json());
        setPending(false);
        return;
      }
      if (res && res.status !== 404) { setPending(false); return; }
      retriesRef.current += 1;
      if (retriesRef.current < 8) setTimeout(poll, 2500);
      else setPending(false);
    }
    poll();
    return () => { cancelled = true; };
  }, [insightId]);

  if (!score && pending) return (
    <p style={{ fontSize: 10, color: 'var(--muted)', margin: '6px 0 0' }}>Judge scoring…</p>
  );
  if (!score) return null;

  return (
    <div style={{ marginTop: 10, borderTop: '1px solid var(--card-border)', paddingTop: 8 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 5 }}>
        <span style={{ fontSize: 11, color: 'var(--muted)', fontWeight: 600, letterSpacing: '0.04em' }}>LLM JUDGE</span>
        <span style={{ fontSize: 13, fontWeight: 700, color: score.mean_score >= 0.75 ? '#22c55e' : score.mean_score >= 0.55 ? '#f59e0b' : '#ef4444' }}>
          {(score.mean_score * 100).toFixed(0)}%
        </span>
      </div>
      {JUDGE_DIMS.map(({ key, label, color }) => {
        const pct = (score[key] ?? 0) * 100;
        return (
          <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
            <span style={{ fontSize: 10, color: 'var(--muted)', width: 76, flexShrink: 0 }}>{label}</span>
            <div style={{ flex: 1, height: 4, borderRadius: 2, background: '#1a2540', overflow: 'hidden' }}>
              <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 2 }} />
            </div>
            <span style={{ fontSize: 10, color, width: 28, textAlign: 'right' }}>{pct.toFixed(0)}%</span>
          </div>
        );
      })}
      {score.rationale && (
        <p style={{ fontSize: 10, color: 'var(--muted)', margin: '5px 0 0', fontStyle: 'italic', lineHeight: 1.4 }}>
          {score.rationale}
        </p>
      )}
    </div>
  );
}

// ── Knowledge bar ──────────────────────────────────────────────────────────

const SOURCE_META = {
  openf1:    { label: 'OpenF1',   endpoint: '/api/v1/knowledge/ingest',          defaultN: 8  },
  fastf1:    { label: 'FastF1',   endpoint: '/api/v1/knowledge/ingest/fastf1',   defaultN: 5  },
  jolpica:   { label: 'Jolpica',  endpoint: '/api/v1/knowledge/ingest/jolpica',  defaultN: 8  },
  knowledge: { label: 'Circuits', endpoint: null, defaultN: null },
};

function KnowledgeBar() {
  const [status, setStatus]         = useState(null);
  const [loading, setLoading]       = useState({});
  const [result, setResult]         = useState('');
  const [years, setYears]           = useState(new Date().getFullYear().toString());
  const [n, setN]                   = useState(8);
  const [uploading, setUploading]   = useState(false);
  const [uploadResult, setUploadResult] = useState('');

  function refreshStatus() {
    fetch('/api/v1/knowledge/status').then(r => r.ok ? r.json() : null).then(d => { if (d) setStatus(d); }).catch(() => {});
  }

  useEffect(() => { refreshStatus(); }, []);

  async function uploadDoc(file) {
    if (!file) return;
    setUploading(true); setUploadResult('');
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await fetch('/api/v1/documents/ingest', { method: 'POST', body: fd });
      const data = await res.json();
      refreshStatus();
      setUploadResult(`Indexed ${data.chunks_indexed ?? 0} chunks from "${data.filename ?? file.name}"`);
    } catch (e) { setUploadResult(`Upload error: ${e.message}`); }
    finally { setUploading(false); }
  }

  async function ingest(source) {
    const meta = SOURCE_META[source];
    if (!meta.endpoint) return;
    setLoading(l => ({ ...l, [source]: true }));
    setResult('');
    try {
      const nParam = source === 'fastf1' ? Math.min(n, 5) : n;
      const res = await fetch(`${meta.endpoint}?years=${encodeURIComponent(years)}&n=${nParam}`, { method: 'POST' });
      const data = await res.json();
      refreshStatus();
      setResult(`[${meta.label}] Indexed ${data.ingested} — ${data.documents_total} docs total (${data.latency_ms}ms)`);
    } catch (e) { setResult(`[${meta.label}] Error: ${e.message}`); }
    finally { setLoading(l => ({ ...l, [source]: false })); }
  }

  const bySource = status?.by_source ?? {};

  return (
    <div className="kb-bar">
      <div className="kb-left">
        <BookOpen size={13} />
        <span className="kb-count">{status ? `${status.documents} docs` : '…'}</span>
        <span className="kb-sep">·</span>
        {Object.entries(SOURCE_META).map(([key, meta]) => (
          <span
            key={key}
            className={`kb-source-badge ${bySource[key] ? 'kb-source-active' : 'kb-source-empty'}`}
            title={bySource[key] ? `${bySource[key]} ${meta.label} docs` : `No ${meta.label} docs indexed`}
          >
            {meta.label}{bySource[key] ? ` ${bySource[key]}` : ''}
          </span>
        ))}
      </div>
      <div className="kb-right">
        <input className="kb-input" value={years} onChange={e => setYears(e.target.value)}
          title="Comma-separated years, e.g. 2024,2025" placeholder="years" />
        <input className="kb-input kb-n" type="number" value={n} min={1} max={24}
          onChange={e => setN(Number(e.target.value))} title="Races per year" />
        {['openf1', 'fastf1', 'jolpica'].map(source => {
          const busy = !!loading[source];
          return (
            <button key={source} className="kb-btn" onClick={() => ingest(source)} disabled={busy}>
              <RefreshCw size={12} className={busy ? 'spin' : ''} />
              {busy ? 'Fetching…' : `Sync ${SOURCE_META[source].label}`}
            </button>
          );
        })}
      </div>
      {result && <span className="kb-result">{result}</span>}
      <div className="kb-right" style={{ marginTop: 4 }}>
        <label className="kb-btn" style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}>
          <Upload size={12} className={uploading ? 'spin' : ''} />
          {uploading ? 'Uploading…' : 'Upload Doc'}
          <input type="file" accept=".pdf,.txt,.md,.png,.jpg,.jpeg"
            style={{ display: 'none' }} onChange={e => { uploadDoc(e.target.files[0]); e.target.value = ''; }} />
        </label>
        {uploadResult && <span className="kb-result">{uploadResult}</span>}
      </div>
    </div>
  );
}

// ── Chat panel ─────────────────────────────────────────────────────────────

const CHAT_SUGGESTIONS = [
  'What causes front-left tyre degradation at Silverstone?',
  'When should a driver consider an ERS deployment cut?',
  'How does wet weather affect brake temperature management?',
  'Explain the relationship between axle imbalance and pit timing.',
];

function ChatBubble({ msg }) {
  const isUser = msg.role === 'user';
  return (
    <div className={`chat-bubble-row ${isUser ? 'chat-user' : 'chat-assistant'}`}>
      <div className="chat-avatar">{isUser ? <User size={13} /> : <Bot size={13} />}</div>
      <div className="chat-bubble">
        <p className="chat-text">{msg.content}</p>
        {msg.evidence?.length > 0 && (
          <div className="chat-evidence">
            {msg.evidence.map(e => (
              <span key={e.source_id} className="chat-ev-chip" title={e.text.slice(0, 200)}>
                {e.title} · {Math.round(e.score * 100)}%
              </span>
            ))}
          </div>
        )}
        {msg.latency_ms != null && (
          <span className="chat-latency">{msg.latency_ms.toFixed(0)} ms</span>
        )}
      </div>
    </div>
  );
}

const RISK_COLOR = { CRITICAL: '#ef4444', WARNING: '#f59e0b', INFO: '#22c55e' };

function DocAnalysisBubble({ msg }) {
  const a = msg.analysis;
  const rColor = RISK_COLOR[a?.risk_signal] ?? '#64748b';
  return (
    <div className="chat-bubble-row chat-assistant">
      <div className="chat-avatar"><FileText size={13} /></div>
      <div className="chat-bubble" style={{ maxWidth: '90%' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
          <span style={{ fontWeight: 700, fontSize: 12 }}>{msg.filename}</span>
          <span style={{ fontSize: 10, fontWeight: 700, color: rColor, border: `1px solid ${rColor}`, borderRadius: 4, padding: '1px 6px' }}>
            {a?.risk_signal ?? 'INFO'}
          </span>
          <span style={{ fontSize: 10, color: 'var(--muted)' }}>{msg.chunks_indexed} chunk{msg.chunks_indexed !== 1 ? 's' : ''} indexed</span>
        </div>
        {a?.summary && <p className="chat-text" style={{ marginBottom: 6 }}>{a.summary}</p>}
        {a?.key_findings?.length > 0 && (
          <ul style={{ margin: '0 0 6px 0', paddingLeft: 16, fontSize: 12 }}>
            {a.key_findings.map((f, i) => <li key={i} style={{ marginBottom: 2 }}>{f}</li>)}
          </ul>
        )}
        {a?.recommended_action && (
          <p style={{ fontSize: 11, color: 'var(--muted)', margin: 0 }}>
            <strong>Action:</strong> {a.recommended_action}
          </p>
        )}
        {msg.latency_ms != null && (
          <span className="chat-latency">{msg.latency_ms.toFixed(0)} ms</span>
        )}
      </div>
    </div>
  );
}

function ChatPanel({ version }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput]       = useState('');
  const [loading, setLoading]   = useState(false);
  const [docUploading, setDocUploading] = useState(false);
  const [error, setError]       = useState('');
  const [isListening, setIsListening] = useState(false);
  const bottomRef = useRef(null);
  const llm = LLM_META[version?.model_backend] ?? LLM_META.rules;

  function startListening() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert('Speech recognition not supported in this browser.');
      return;
    }
    const rec = new SpeechRecognition();
    rec.onstart = () => setIsListening(true);
    rec.onresult = (e) => setInput(e.results[0][0].transcript);
    rec.onend = () => setIsListening(false);
    rec.start();
  }

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  async function uploadAndAnalyse(file) {
    if (!file || docUploading) return;
    setDocUploading(true);
    setError('');
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await fetch('/api/v1/documents/analyse', { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      const data = await res.json();
      setMessages(prev => [...prev, { role: 'document', ...data }]);
    } catch (e) {
      setError(`Upload failed: ${e.message}`);
    } finally {
      setDocUploading(false);
    }
  }

  const history = messages.map(m => ({ role: m.role, content: m.content }));

  async function send(text) {
    const q = (text ?? input).trim();
    if (!q || loading) return;
    setInput('');
    setError('');
    setMessages(prev => [...prev, { role: 'user', content: q }]);
    setLoading(true);
    try {
      const res = await fetch('/api/v1/chat', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ message: q, history }),
      });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      const data = await res.json();
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.response,
        evidence: data.evidence,
        latency_ms: data.latency_ms,
      }]);
    } catch (e) {
      setError(String(e.message ?? e));
      setMessages(prev => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  }

  return (
    <div className="chat-shell">
      <KnowledgeBar />
      <div className="chat-thread">
        {messages.length === 0 && (
          <div className="chat-empty">
            <Bot size={32} color="#1e3a5c" strokeWidth={1.5} />
            <p>Ask anything about F1 race engineering — tire strategy, ERS, braking, weather adaptation.</p>
            <div className="chat-suggestions">
              {CHAT_SUGGESTIONS.map(s => (
                <button key={s} className="chat-suggestion" onClick={() => send(s)}>{s}</button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) =>
          m.role === 'document'
            ? <DocAnalysisBubble key={i} msg={m} />
            : <ChatBubble key={i} msg={m} />
        )}
        {loading && (
          <div className="chat-bubble-row chat-assistant">
            <div className="chat-avatar"><Bot size={13} /></div>
            <div className="chat-bubble chat-thinking">
              <span className="dot" /><span className="dot" /><span className="dot" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {error && <pre className="error chat-error">{error}</pre>}

      <div className="chat-input-row">
        <div className="chat-input-wrap">
          <textarea
           className="chat-input"
           placeholder="Ask about tire strategy, ERS, braking, weather…"
           value={input}
           onChange={e => setInput(e.target.value)}
           onKeyDown={handleKey}
           rows={1}
          />
          <button className={`chat-mic${isListening ? ' active' : ''}`} onClick={startListening} title="Voice Command">
           <Mic size={16} />
          </button>
          <label className="chat-mic" title="Upload & Analyse Document" style={{ cursor: 'pointer' }}>
            <FileText size={16} className={docUploading ? 'spin' : ''} />
            <input type="file" accept=".pdf,.txt,.md,.png,.jpg,.jpeg"
              style={{ display: 'none' }}
              onChange={e => { uploadAndAnalyse(e.target.files[0]); e.target.value = ''; }} />
          </label>
          <button className="chat-send" onClick={() => send()} disabled={loading || !input.trim()}
            style={{ color: llm.color }}>
            <Send size={16} />
          </button>
        </div>
        <span className="chat-model-tag" style={{ color: llm.color }}>
          {version?.model_backend === 'openai_compatible'
            ? version.llm_open_source_model
            : version?.model_backend === 'anthropic'
              ? version.llm_advice_model
              : 'Rules Engine'}
        </span>
      </div>
    </div>
  );
}

// ── Telemetry panel ────────────────────────────────────────────────────────

function TelemetryPanel({ version }) {
  const [stats, setStats]       = useState(DEFAULT_STATS);
  const [audience, setAudience] = useState('DRIVER');
  const [insight, setInsight]   = useState(null);
  const [error, setError]       = useState('');
  const [loading, setLoading]   = useState(false);

  async function analyze() {
    setError(''); setInsight(null); setLoading(true);
    try {
      const win = buildWindow(stats);
      const res = await fetch(`/api/v1/insights?audience=${audience}`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(win),
      });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      setInsight(await res.json());
    } catch (e) {
      setError(String(e.message ?? e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid">
      <div className="card input-card">
        <div className="input-header">
          <h2>Race Stats Input</h2>
          <div className="seg">
            {['DRIVER', 'ENGINEER', 'STRATEGY'].map(a => (
              <button key={a} className={`seg-btn${audience === a ? ' active' : ''}`}
                onClick={() => setAudience(a)}>{a}</button>
            ))}
          </div>
        </div>
        <StatsForm stats={stats} onChange={setStats} />
        <button className="analyze-btn" onClick={analyze} disabled={loading}>
          {loading ? <><Activity size={13} /> Analyzing…</> : 'Get Insight'}
        </button>
        {error && <pre className="error">{error}</pre>}
      </div>

      <div className="card insight-card">
        <h2><ShieldAlert size={15} /> Insight</h2>
        {!insight
          ? <p className="muted empty-hint">Set race stats and click Get Insight.</p>
          : <InsightPanel insight={insight} modelBackend={version?.model_backend} />}
      </div>
    </div>
  );
}

// ── Live panel ─────────────────────────────────────────────────────────────

function fmtLapTime(s) {
  if (s == null) return '';
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(3).padStart(6, '0');
  return `${m}:${sec}`;
}

const COMPOUND_COLOURS = {
  SOFT: '#e53e3e', MEDIUM: '#d69e2e', HARD: '#cbd5e0',
  INTERMEDIATE: '#38a169', WET: '#3182ce',
};

function deriveStints(laps) {
  const stints = [];
  let cur = null;
  for (const lap of laps) {
    const c = (lap.compound ?? 'UNKNOWN').toUpperCase();
    if (!cur || cur.compound !== c) {
      if (cur) stints.push(cur);
      cur = { compound: c, start: lap.lap_number, end: lap.lap_number };
    } else {
      cur.end = lap.lap_number;
    }
  }
  if (cur) stints.push(cur);
  return stints;
}

function RaceSummaryBar({ laps, label }) {
  if (!laps.length) return null;
  const stints = deriveStints(laps);
  const fl = [...laps].filter(l => l.lap_time_s).sort((a, b) => a.lap_time_s - b.lap_time_s)[0];
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 5, flexWrap: 'wrap', marginTop: 4 }}>
      {label && <span style={{ fontSize: '0.68rem', fontWeight: 700, color: 'var(--muted)', minWidth: 28 }}>{label}</span>}
      {stints.map(s => (
        <span key={s.start} style={{
          fontSize: '0.66rem', padding: '1px 5px', borderRadius: 3,
          background: COMPOUND_COLOURS[s.compound] ?? '#555',
          color: s.compound === 'HARD' ? '#1a1a2e' : '#fff', fontWeight: 600,
        }}>
          {s.compound[0]} {s.start}–{s.end}
        </span>
      ))}
      {fl && (
        <span style={{ fontSize: '0.66rem', color: 'var(--muted)', marginLeft: 2 }}>
          FL L{fl.lap_number} {fmtLapTime(fl.lap_time_s)}
        </span>
      )}
    </div>
  );
}

function StintChart({ laps, selectedLap, onSelect }) {
  if (!laps.length) return null;
  const stints = deriveStints(laps);
  const minLap = laps[0].lap_number;
  const maxLap = laps[laps.length - 1].lap_number;
  const span = Math.max(1, maxLap - minLap + 1);

  function handleClick(e) {
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    onSelect(Math.round(minLap + pct * (span - 1)));
  }

  const cursorPct = selectedLap != null ? ((selectedLap - minLap) / span) * 100 : null;

  return (
    <div style={{ marginTop: 6 }}>
      <svg width="100%" height="18" style={{ cursor: 'pointer', borderRadius: 3, display: 'block', overflow: 'visible' }}
           onClick={handleClick}>
        {stints.map(s => (
          <rect key={s.start}
            x={`${((s.start - minLap) / span) * 100}%`} y="0"
            width={`${((s.end - s.start + 1) / span) * 100}%`} height="18"
            fill={COMPOUND_COLOURS[s.compound] ?? '#555'} />
        ))}
        {cursorPct != null && (
          <line x1={`${cursorPct}%`} y1="-2" x2={`${cursorPct}%`} y2="20"
                stroke="white" strokeWidth="2" opacity="0.9" />
        )}
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem', color: 'var(--muted)', marginTop: 1 }}>
        <span>L{minLap}</span><span>L{maxLap}</span>
      </div>
    </div>
  );
}

function LapBadge({ lap }) {
  if (!lap) return null;
  const c = lap.compound?.toUpperCase() ?? 'MEDIUM';
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: '0.7rem', color: 'var(--muted)' }}>
      <span style={{ width: 8, height: 8, borderRadius: '50%', background: COMPOUND_COLOURS[c] ?? '#888', flexShrink: 0 }} />
      {fmtLapTime(lap.lap_time_s)}
      {lap.tyre_life != null && <span style={{ opacity: 0.6 }}>· {c[0]} ×{lap.tyre_life}</span>}
    </span>
  );
}

const RISK_COLOURS = { INFO: '#4a5568', WATCH: '#d69e2e', WARNING: '#e53e3e', CRITICAL: '#805ad5' };

function InsightHistoryStrip({ history, onRestore }) {
  if (!history.length) return null;
  return (
    <div style={{ display: 'flex', gap: 6, overflowX: 'auto', paddingBottom: 4 }}>
      {history.map(item => (
        <button key={item.id} onClick={() => onRestore(item)} style={{
          flexShrink: 0, background: '#0d1b2e', border: '1px solid var(--border)',
          borderRadius: 6, padding: '4px 8px', cursor: 'pointer', textAlign: 'left', minWidth: 88,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 2 }}>
            {item.lapInfo && (
              <span style={{ width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
                background: COMPOUND_COLOURS[item.lapInfo.compound?.toUpperCase()] ?? '#888' }} />
            )}
            <span style={{ fontSize: '0.7rem', color: 'var(--text)', fontWeight: 600 }}>
              {item.lap != null ? `L${item.lap}` : 'Best'}
            </span>
            <span style={{ width: 6, height: 6, borderRadius: '50%', marginLeft: 'auto', flexShrink: 0,
              background: RISK_COLOURS[item.risk] ?? '#4a5568' }} />
          </div>
          <div style={{ fontSize: '0.62rem', color: 'var(--muted)', maxWidth: 96,
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {item.recommendation}
          </div>
        </button>
      ))}
    </div>
  );
}

function LapDeltaChart({ laps, laps2, driver, driver2, selectedLap, onSelect }) {
  if (!laps.length || !laps2.length) return null;
  const map2 = Object.fromEntries(laps2.map(l => [l.lap_number, l]));
  const deltas = laps
    .filter(l => l.lap_time_s && map2[l.lap_number]?.lap_time_s)
    .map(l => ({ lap: l.lap_number, delta: l.lap_time_s - map2[l.lap_number].lap_time_s }));
  if (!deltas.length) return null;
  const minLap = deltas[0].lap, maxLap = deltas[deltas.length - 1].lap;
  const span = Math.max(1, maxLap - minLap);
  const maxAbs = Math.max(1, Math.max(...deltas.map(d => Math.abs(d.delta))));
  const W = 1000, H = 80, MID = H / 2;
  const lx = lap => ((lap - minLap) / span) * W;
  const dy = d => MID - (d / maxAbs) * (MID - 5);
  const cursor = selectedLap != null ? lx(selectedLap) : null;
  return (
    <div>
      <p style={{ fontSize: '0.68rem', color: 'var(--muted)', marginBottom: 3 }}>
        Lap delta · <span style={{ color: '#63b3ed' }}>{driver}</span> vs <span style={{ color: '#f6ad55' }}>{driver2}</span>
        <span style={{ opacity: 0.5 }}> · + = {driver2} faster</span>
      </p>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none"
           style={{ width: '100%', height: 64, background: '#0a0f1e', borderRadius: 4, display: 'block', cursor: 'pointer' }}
           onClick={e => {
             const r = e.currentTarget.getBoundingClientRect();
             onSelect(Math.round(minLap + ((e.clientX - r.left) / r.width) * span));
           }}>
        <line x1="0" y1={MID} x2={W} y2={MID} stroke="rgba(255,255,255,0.15)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
        {deltas.map((d, i) => {
          const x = lx(d.lap);
          const nx = i < deltas.length - 1 ? lx(deltas[i + 1].lap) : x + W / deltas.length;
          const w = Math.max(2, nx - x - 1);
          const y = d.delta >= 0 ? dy(d.delta) : MID;
          const h = Math.max(1, Math.abs(dy(d.delta) - MID));
          return <rect key={d.lap} x={x} y={y} width={w} height={h}
            fill={d.delta > 0 ? '#f6ad55' : '#63b3ed'} opacity="0.85" />;
        })}
        {cursor != null && (
          <line x1={cursor} y1="0" x2={cursor} y2={H} stroke="white" strokeWidth="2" opacity="0.8" vectorEffect="non-scaling-stroke" />
        )}
        <text x="4" y="10" fontSize="9" fill="rgba(255,255,255,0.35)" dominantBaseline="hanging">+{maxAbs.toFixed(1)}s</text>
        <text x="4" y={H - 2} fontSize="9" fill="rgba(255,255,255,0.35)">−{maxAbs.toFixed(1)}s</text>
      </svg>
    </div>
  );
}

function TraceChart({ trace, trace2, driver, driver2, lap }) {
  if (!trace.length) return null;
  const maxDist = Math.max(...trace.map(p => p.dist), 1);
  const W = 1000, H = 100, SH = 74;
  const tx = d => (d / maxDist) * W;
  const sy = s => SH - ((Math.max(50, Math.min(350, s)) - 50) / 300) * SH;
  const pts = arr => arr.map(p => `${tx(p.dist)},${sy(p.speed)}`).join(' ');
  const throtPath = trace.length < 2 ? '' : [
    `M${tx(trace[0].dist)},${H}`,
    ...trace.map(p => `L${tx(p.dist)},${SH + (1 - p.throttle / 100) * (H - SH)}`),
    `L${tx(trace[trace.length - 1].dist)},${H}Z`,
  ].join('');
  return (
    <div style={{ marginTop: 10 }}>
      <p style={{ fontSize: '0.68rem', color: 'var(--muted)', marginBottom: 3 }}>
        Speed · Throttle · Brake{lap != null ? ` — Lap ${lap}` : ''}
      </p>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none"
           style={{ width: '100%', height: 80, background: '#0a0f1e', borderRadius: 4, display: 'block' }}>
        {trace.map((p, i) => p.drs && i < trace.length - 1 && (
          <rect key={i} x={tx(p.dist)} y={0}
            width={Math.max(1, tx(trace[i + 1]?.dist ?? p.dist) - tx(p.dist))} height={H}
            fill="#1a4040" />
        ))}
        {throtPath && <path d={throtPath} fill="#1a4a1a" opacity="0.75" />}
        {trace.map((p, i) => p.brake && (
          <rect key={i} x={tx(p.dist)} y={SH}
            width={Math.max(2, tx(trace[i + 1]?.dist ?? p.dist) - tx(p.dist))} height={H - SH}
            fill="#e53e3e" opacity="0.85" />
        ))}
        <polyline points={pts(trace)} fill="none" stroke="#63b3ed" strokeWidth="2" vectorEffect="non-scaling-stroke" />
        {trace2.length > 0 && (
          <polyline points={pts(trace2)} fill="none" stroke="#f6ad55" strokeWidth="1.5"
            strokeDasharray="6,3" vectorEffect="non-scaling-stroke" />
        )}
        <text x="4" y="11" fontSize="9" fill="rgba(255,255,255,0.4)" dominantBaseline="hanging">350</text>
        <text x="4" y={SH - 2} fontSize="9" fill="rgba(255,255,255,0.4)">50</text>
      </svg>
      <div style={{ display: 'flex', gap: 10, fontSize: '0.63rem', color: 'var(--muted)', marginTop: 3 }}>
        <span><span style={{ color: '#63b3ed' }}>━</span> {driver} speed</span>
        {trace2.length > 0 && <span><span style={{ color: '#f6ad55' }}>╌</span> {driver2}</span>}
        <span><span style={{ color: '#38a169' }}>█</span> throttle</span>
        <span><span style={{ color: '#e53e3e' }}>█</span> brake</span>
        <span><span style={{ color: '#1a6060' }}>█</span> DRS</span>
      </div>
    </div>
  );
}

function LivePanel({ version }) {
  const [races, setRaces]             = useState([]);
  const [drivers, setDrivers]         = useState([]);
  const [laps, setLaps]               = useState([]);
  const [laps2, setLaps2]             = useState([]);
  const [roundNum, setRoundNum]       = useState('');
  const [driver, setDriver]           = useState('');
  const [driver2, setDriver2]         = useState('');
  const [selectedLap, setSelectedLap] = useState(null);
  const [replayMode, setReplayMode]   = useState(false);
  const [audience, setAudience]       = useState('DRIVER');
  const [insight, setInsight]         = useState(null);
  const [insight2, setInsight2]       = useState(null);
  const [history, setHistory]         = useState([]);
  const [trace, setTrace]             = useState([]);
  const [trace2, setTrace2]           = useState([]);
  const [error, setError]             = useState('');
  const [loading, setLoading]         = useState(false);
  const [loadingDrivers, setLoadingDrivers] = useState(false);
  const [year, setYear]               = useState(2024);

  useEffect(() => {
    setError(''); setRaces([]); setRoundNum('');
    setDrivers([]); setDriver(''); setDriver2('');
    setLaps([]); setLaps2([]); setInsight(null); setInsight2(null);
    setHistory([]); setTrace([]); setTrace2([]);
    fetch(`/api/v1/session/races?year=${year}`)
      .then(r => r.ok ? r.json() : [])
      .then(setRaces)
      .catch(() => {});
  }, [year]);

  useEffect(() => {
    if (!roundNum) return;
    setDrivers([]); setDriver(''); setDriver2('');
    setLaps([]); setLaps2([]); setSelectedLap(null);
    setInsight(null); setInsight2(null);
    setHistory([]); setTrace([]); setTrace2([]);
    setLoadingDrivers(true);
    fetch(`/api/v1/session/drivers/${year}/${roundNum}`)
      .then(r => r.ok ? r.json() : [])
      .then(rows => {
        setDrivers(rows);
        if (rows.length) setDriver(rows[0].code);
      })
      .catch(() => {})
      .finally(() => setLoadingDrivers(false));
  }, [year, roundNum]);

  useEffect(() => {
    if (!roundNum || !driver) return;
    setLaps([]); setSelectedLap(null); setInsight(null); setHistory([]); setTrace([]);
    fetch(`/api/v1/session/laps/${year}/${roundNum}/${driver}`)
      .then(r => r.ok ? r.json() : [])
      .then(rows => {
        setLaps(rows);
        if (rows.length) setSelectedLap(rows[rows.length - 1].lap_number);
      })
      .catch(() => {});
  }, [year, roundNum, driver]);

  useEffect(() => {
    if (!roundNum || !driver2) { setLaps2([]); setInsight2(null); setTrace2([]); return; }
    fetch(`/api/v1/session/laps/${year}/${roundNum}/${driver2}`)
      .then(r => r.ok ? r.json() : [])
      .then(setLaps2)
      .catch(() => {});
  }, [year, roundNum, driver2]);

  async function fetchTrace(lap) {
    if (!roundNum || !driver || lap == null) return;
    const [r1, r2] = await Promise.all([
      fetch(`/api/v1/session/trace/${year}/${roundNum}/${driver}/${lap}`),
      driver2 ? fetch(`/api/v1/session/trace/${year}/${roundNum}/${driver2}/${lap}`) : Promise.resolve(null),
    ]);
    setTrace(r1.ok ? await r1.json() : []);
    setTrace2(r2 && r2.ok ? await r2.json() : []);
  }

  async function fetchInsight(lapOverride) {
    if (!roundNum || !driver) return;
    setError(''); setLoading(true);
    const lap = lapOverride ?? (replayMode ? selectedLap : null);
    const lapParam = lap != null ? `&lap_number=${lap}` : '';
    const base = `/api/v1/session/insight?year=${year}&round_num=${roundNum}&audience=${audience}${lapParam}`;
    try {
      const reqs = [fetch(`${base}&driver=${driver}`, { method: 'POST' })];
      if (driver2) reqs.push(fetch(`${base}&driver=${driver2}`, { method: 'POST' }));
      const [res1, res2] = await Promise.all(reqs);
      if (!res1.ok) throw new Error(await res1.text());
      const d1 = await res1.json();
      const d2 = res2 && res2.ok ? await res2.json() : null;
      setInsight(d1);
      setInsight2(d2);
      const lapInfo = laps.find(l => l.lap_number === lap) ?? null;
      setHistory(prev => [{
        id: Date.now(), lap, lapInfo, driver, driver2,
        risk: d1.risk, recommendation: d1.recommendation,
        insight: d1, insight2: d2,
      }, ...prev].slice(0, 8));
      if (replayMode && lap != null) fetchTrace(lap);
    } catch (e) {
      setError(String(e.message ?? e));
    } finally {
      setLoading(false);
    }
  }

  const selectedRace = races.find(r => String(r.round) === String(roundNum));
  const minLap = laps.length ? laps[0].lap_number : 1;
  const maxLap = laps.length ? laps[laps.length - 1].lap_number : 1;
  const currentLapInfo = laps.find(l => l.lap_number === selectedLap) ?? null;

  function stepLap(delta) {
    setSelectedLap(prev => {
      const next = Math.min(maxLap, Math.max(minLap, (prev ?? maxLap) + delta));
      fetchInsight(next);
      return next;
    });
  }

  function handleChartClick(lap) {
    setSelectedLap(lap);
    if (replayMode) fetchInsight(lap);
  }

  function restoreHistoryItem(item) {
    setInsight(item.insight);
    setInsight2(item.insight2);
    if (item.lap != null) setSelectedLap(item.lap);
  }

  return (
    <div className="grid">
      <div className="card input-card">
        <div className="input-header">
          <h2><History size={14} /> Session Replay</h2>
          <div className="seg">
            {['DRIVER', 'ENGINEER', 'STRATEGY'].map(a => (
              <button key={a} className={`seg-btn${audience === a ? ' active' : ''}`}
                onClick={() => setAudience(a)}>{a}</button>
            ))}
          </div>
        </div>

        <div className="stats-form">
          <Section title="Session">
            <div className="context-row">
              <label><span>Year</span>
                <input type="number" className="text-input num-input" value={year} min={2018} max={2025}
                  onChange={e => setYear(Number(e.target.value))} />
              </label>
              <label style={{ flex: 2 }}><span>Race</span>
                <select className="text-input" value={roundNum}
                  onChange={e => setRoundNum(e.target.value)}>
                  <option value="">— select —</option>
                  {races.map(r => (
                    <option key={r.round} value={r.round}>
                      R{r.round} · {r.name} ({r.date})
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {selectedRace && (
              <p className="muted" style={{ fontSize: '0.72rem', marginTop: 4 }}>
                {selectedRace.circuit} · {selectedRace.country}
              </p>
            )}
          </Section>

          <Section title="Driver">
            <div className="context-row">
              <label style={{ flex: 1 }}><span>Driver 1</span>
                <select className="text-input" value={driver}
                  onChange={e => { setDriver(e.target.value); setInsight(null); }}
                  disabled={!drivers.length}>
                  {loadingDrivers
                    ? <option>Loading…</option>
                    : !drivers.length
                      ? <option>— pick race first —</option>
                      : drivers.map(d => <option key={d.code} value={d.code}>{d.code}</option>)}
                </select>
              </label>
              <label style={{ flex: 1 }}><span>vs</span>
                <select className="text-input" value={driver2}
                  onChange={e => { setDriver2(e.target.value); setInsight2(null); }}
                  disabled={!drivers.length}>
                  <option value="">— none —</option>
                  {drivers.filter(d => d.code !== driver).map(d => (
                    <option key={d.code} value={d.code}>{d.code}</option>
                  ))}
                </select>
              </label>
            </div>
            {laps.length > 0 && <RaceSummaryBar laps={laps} label={driver2 ? driver : null} />}
            {driver2 && laps2.length > 0 && <RaceSummaryBar laps={laps2} label={driver2} />}
          </Section>

          {laps.length > 0 && (
            <Section title="Lap">
              <StintChart laps={laps} selectedLap={selectedLap} onSelect={handleChartClick} />
              <div className="context-row" style={{ alignItems: 'center', gap: 8, marginTop: 8 }}>
                <label className="check-label">
                  <input type="checkbox" checked={replayMode}
                    onChange={e => setReplayMode(e.target.checked)} />
                  <span>Step through laps</span>
                </label>
                {replayMode && (
                  <>
                    <button className="kb-btn" onClick={() => stepLap(-1)}
                      disabled={loading || selectedLap <= minLap}>◀</button>
                    <div style={{ minWidth: 80, textAlign: 'center' }}>
                      <div style={{ fontSize: '0.8rem', color: 'var(--text)' }}>
                        Lap {selectedLap} / {maxLap}
                      </div>
                      <LapBadge lap={currentLapInfo} />
                    </div>
                    <button className="kb-btn" onClick={() => stepLap(1)}
                      disabled={loading || selectedLap >= maxLap}>▶</button>
                    <input type="range" min={minLap} max={maxLap} value={selectedLap ?? maxLap}
                      onChange={e => setSelectedLap(Number(e.target.value))}
                      style={{ flex: 1 }} />
                  </>
                )}
              </div>
            </Section>
          )}
        </div>

        <button className="analyze-btn" onClick={() => fetchInsight()}
          disabled={loading || !roundNum || !driver} style={{ marginTop: 12, width: '100%' }}>
          {loading
            ? <><Activity size={13} /> Loading telemetry…</>
            : replayMode
              ? <>Analyse Lap {selectedLap}{currentLapInfo ? ` · ${fmtLapTime(currentLapInfo.lap_time_s)}` : ''}</>
              : driver2 ? `Compare ${driver} vs ${driver2}` : 'Analyse Session'}
        </button>

        <p className="muted" style={{ fontSize: '0.70rem', marginTop: 8 }}>
          Uses FastF1. First load per session ~50 MB / 20–30 s. Analyses a 5-lap window.
        </p>
        {error && <pre className="error">{error}</pre>}
      </div>

      <div className="card insight-card">
        <h2><ShieldAlert size={15} />{' '}
          {driver2
            ? `${driver} vs ${driver2}${replayMode ? ` — Lap ${selectedLap}` : ''}`
            : replayMode ? `Lap ${selectedLap} Insight` : 'Session Insight'}
        </h2>

        {history.length > 0 && (
          <InsightHistoryStrip history={history} onRestore={restoreHistoryItem} />
        )}

        {driver2 && laps.length > 0 && laps2.length > 0 && (
          <LapDeltaChart
            laps={laps} laps2={laps2}
            driver={driver} driver2={driver2}
            selectedLap={selectedLap} onSelect={handleChartClick}
          />
        )}

        {driver2 ? (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div>
              <p style={{ fontWeight: 700, fontSize: '0.8rem', marginBottom: 8, color: 'var(--text)' }}>{driver}</p>
              {insight
                ? <InsightPanel insight={insight} modelBackend={version?.model_backend} />
                : <p className="muted empty-hint" style={{ fontSize: '0.75rem' }}>Click Compare to analyse.</p>}
            </div>
            <div style={{ borderLeft: '1px solid var(--border)', paddingLeft: 16 }}>
              <p style={{ fontWeight: 700, fontSize: '0.8rem', marginBottom: 8, color: 'var(--text)' }}>{driver2}</p>
              {insight2
                ? <InsightPanel insight={insight2} modelBackend={version?.model_backend} />
                : <p className="muted empty-hint" style={{ fontSize: '0.75rem' }}>Click Compare to analyse.</p>}
            </div>
          </div>
        ) : (
          !insight
            ? <p className="muted empty-hint">Pick a race and driver, then click Analyse Session.</p>
            : <InsightPanel insight={insight} modelBackend={version?.model_backend} />
        )}

        {trace.length > 0 && (
          <TraceChart
            trace={trace} trace2={trace2}
            driver={driver} driver2={driver2}
            lap={selectedLap}
          />
        )}
      </div>
    </div>
  );
}

const SCENARIO_COLORS = ['#38bdf8', '#22c55e', '#a78bfa'];

function StrategyComparisonTable({ comparison }) {
  const best = comparison.scenarios.find(s => s.recommended);
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <h4 style={{ margin: 0 }}>Strategy Comparison</h4>
        <span className="chat-latency">{comparison.latency_ms.toFixed(0)} ms</span>
      </div>
      <div style={{ padding: '10px 14px', borderRadius: 8, background: '#0a1628', border: '1px solid #1e3a5c', marginBottom: 10 }}>
        <span style={{ fontSize: '0.72rem', fontWeight: 700, color: '#38bdf8', letterSpacing: '0.04em' }}>RECOMMENDATION</span>
        <span style={{ fontSize: '0.82rem', color: 'var(--text)', marginLeft: 10 }}>{comparison.recommendation}</span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${comparison.scenarios.length}, 1fr)`, gap: 8 }}>
        {comparison.scenarios.map((s, i) => {
          const color = SCENARIO_COLORS[i] ?? '#64748b';
          const isRec = s.recommended;
          return (
            <div key={s.label} style={{
              padding: '10px 12px', borderRadius: 8,
              background: isRec ? color + '18' : '#0d1b2e',
              border: `1px solid ${isRec ? color + 'aa' : '#1e3a5c'}`,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, flexShrink: 0 }} />
                <span style={{ fontSize: '0.72rem', fontWeight: 700, color, flex: 1 }}>{s.label}</span>
                {isRec && <span style={{ fontSize: '0.62rem', fontWeight: 700, color, background: color + '33', borderRadius: 3, padding: '1px 5px' }}>BEST</span>}
              </div>
              <div className="feat-table" style={{ gap: 2 }}>
                <div className="feat-row" style={{ padding: '2px 0' }}>
                  <span className="feat-key" style={{ fontSize: '0.68rem' }}>Total time</span>
                  <span className="feat-val" style={{ fontSize: '0.72rem', color: isRec ? color : 'var(--text)' }}>
                    {(s.total_time_s / 60).toFixed(1)} min
                  </span>
                </div>
                <div className="feat-row" style={{ padding: '2px 0' }}>
                  <span className="feat-key" style={{ fontSize: '0.68rem' }}>vs best</span>
                  <span className="feat-val" style={{ fontSize: '0.72rem', color: s.delta_s === 0 ? '#22c55e' : '#f97316' }}>
                    {s.delta_s === 0 ? '—' : `+${s.delta_s.toFixed(1)}s`}
                  </span>
                </div>
                {s.pit_lap != null && (
                  <div className="feat-row" style={{ padding: '2px 0' }}>
                    <span className="feat-key" style={{ fontSize: '0.68rem' }}>Pit lap</span>
                    <span className="feat-val" style={{ fontSize: '0.72rem' }}>L{s.pit_lap}</span>
                  </div>
                )}
                {s.cliff_lap != null && (
                  <div className="feat-row" style={{ padding: '2px 0' }}>
                    <span className="feat-key" style={{ fontSize: '0.68rem' }}>Cliff</span>
                    <span className="feat-val" style={{ fontSize: '0.72rem', color: '#ef4444' }}>L{s.cliff_lap}</span>
                  </div>
                )}
                <div className="feat-row" style={{ padding: '2px 0' }}>
                  <span className="feat-key" style={{ fontSize: '0.68rem' }}>End wear FL</span>
                  <span className="feat-val" style={{ fontSize: '0.72rem', color: s.end_wear_fl > 0.85 ? '#ef4444' : 'inherit' }}>
                    {Math.round(s.end_wear_fl * 100)}%
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function PredictionsPanel({ version }) {
  const [stats, setStats]           = useState(DEFAULT_STATS);
  const [projection, setProjection] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState('');

  async function project() {
    setLoading(true); setProjection(null); setComparison(null); setError('');
    try {
      const win = buildWindow(stats);
      const body = JSON.stringify(win);
      const headers = { 'Content-Type': 'application/json' };
      const [r1, r2] = await Promise.all([
        fetch('/api/v1/predictions/race-projection',    { method: 'POST', headers, body }),
        fetch('/api/v1/predictions/strategy-comparison', { method: 'POST', headers, body }),
      ]);
      if (!r1.ok) throw new Error(await r1.text());
      const [proj, comp] = await Promise.all([r1.json(), r2.ok ? r2.json() : null]);
      setProjection(proj);
      setComparison(comp);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }

  return (
    <div className="grid">
      <div className="card input-card">
        <div className="input-header"><h2>Tire Strategy Projection</h2></div>
        <StatsForm stats={stats} onChange={setStats} />
        <button className="analyze-btn" onClick={project} disabled={loading}>
          {loading ? <><Activity size={13} className="spin" /> Simulating…</> : <><TrendingUp size={13} /> Project Stint</>}
        </button>
        {error && <pre className="error">{error}</pre>}
      </div>

      <div className="card insight-card">
        <h2><TrendingUp size={15} /> Tire Degradation Forecast</h2>
        {!projection ? (
          <p className="muted empty-hint">Set tire state and current lap, then run simulation.</p>
        ) : (
          <div className="projection-results">
            {comparison && <StrategyComparisonTable comparison={comparison} />}

            <div className="projection-header" style={{ marginBottom: 8 }}>
              <span className="projection-summary">{projection.summary}</span>
              <span className="chat-latency">{projection.latency_ms.toFixed(0)} ms</span>
            </div>

            <div className="feat-table">
              {projection.projections.map(p => {
                const atCliff = p.wear_fl > 0.85;
                return (
                  <div key={p.lap} className="feat-row" style={atCliff ? { background: '#1e0f04' } : {}}>
                    <span className="feat-key">Lap {p.lap}</span>
                    <span className="feat-val">
                      <span className="p-time">{p.p50_time_s.toFixed(2)}s</span>
                      <span className="p-wear" style={{ color: atCliff ? 'var(--critical)' : p.wear_fl > 0.65 ? '#f97316' : 'inherit' }}>
                        {Math.round(p.wear_fl * 100)}% FL
                      </span>
                      <span style={{ fontSize: '0.72rem', color: p.grip < 0.55 ? '#f97316' : 'var(--muted)' }}>
                        {Math.round(p.grip * 100)}% grip
                      </span>
                    </span>
                  </div>
                );
              })}
            </div>
            <p className="muted" style={{ fontSize: '0.68rem', marginTop: 8 }}>
              Simulation models tire physics only — lap times are relative, not absolute.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Regression panel ───────────────────────────────────────────────────────


function GatePill({ label, pass }) {
  const color = pass ? '#22c55e' : '#ef4444';
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      padding: '3px 9px', borderRadius: 12, fontSize: '0.72rem', fontWeight: 600,
      background: color + '22', border: `1px solid ${color}55`, color,
    }}>
      {pass ? '✓' : '✗'} {label}
    </span>
  );
}

function RegressionReport({ report }) {
  const gates = Object.entries(GATE_META).map(([key, label]) => ({ key, label, pass: report[key] }));
  const overallPass = gates.every(g => g.pass);
  const cases = report.cases ?? [];
  const byClass = report.by_class ?? {};
  const overallColor = overallPass ? '#22c55e' : '#ef4444';

  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
        <span className="risk-pill" style={{ color: overallColor, background: overallColor + '22', borderColor: overallColor + '55', fontSize: '0.85rem' }}>
          {overallPass ? 'ALL GATES PASS' : 'GATES FAILED'}
        </span>
        <span className="muted" style={{ fontSize: '0.72rem' }}>
          {report.positive_cases} positive · {report.nominal_cases} nominal · {report.latency_ms}ms
        </span>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 16 }}>
        {gates.map(g => <GatePill key={g.key} label={g.label} pass={g.pass} />)}
      </div>

      <div className="feat-table" style={{ marginBottom: 16 }}>
        {[
          ['Case recall',         `${(report.case_recall * 100).toFixed(0)}%`],
          ['Agent activation',    `${(report.agent_activation_rate * 100).toFixed(0)}%`],
          ['Source retrieval',    `${(report.source_retrieval_rate * 100).toFixed(0)}%`],
          ['Policy correctness',  `${(report.policy_correctness * 100).toFixed(0)}%`],
          ['False positive rate', `${(report.false_positive_rate * 100).toFixed(0)}%`],
        ].map(([label, val]) => (
          <div key={label} className="feat-row">
            <span className="feat-key">{label}</span>
            <span className="feat-val">{val}</span>
          </div>
        ))}
      </div>

      {Object.keys(byClass).length > 0 && (
        <>
          <h4>By Class</h4>
          <div className="feat-table" style={{ marginBottom: 16 }}>
            {Object.entries(byClass).map(([cls, info]) => (
              <div key={cls} className="feat-row">
                <span className="feat-key" style={{ fontFamily: 'monospace' }}>{cls}</span>
                <span className="feat-val">
                  {info.passed}/{info.cases} pass
                  {info.recall != null ? ` · recall ${(info.recall * 100).toFixed(0)}%` : ''}
                </span>
              </div>
            ))}
          </div>
        </>
      )}

      {cases.length > 0 && (
        <>
          <h4>Cases ({cases.length})</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.72rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--muted)' }}>
                  {['ID', 'Class', 'Expected', 'Observed', 'Conf', 'ms', '✓'].map((h, i) => (
                    <th key={h} style={{ textAlign: i >= 4 ? 'right' : 'left', padding: '4px 6px', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {cases.map(c => {
                  const riskColor = RISK_META[c.observed_risk]?.color ?? '#64748b';
                  const expLabel = c.expected_min_risk
                    ? `≥${c.expected_min_risk}`
                    : c.expected_max_risk
                      ? `≤${c.expected_max_risk}`
                      : '—';
                  const failedChecks = Object.entries(c.checks ?? {}).filter(([, v]) => !v).map(([k]) => k);
                  return (
                    <tr key={c.case_id} style={{ borderBottom: '1px solid #ffffff08' }}
                        title={failedChecks.length ? `Failed: ${failedChecks.join(', ')}` : 'All checks passed'}>
                      <td style={{ padding: '3px 6px', fontFamily: 'monospace', fontSize: '0.68rem', color: 'var(--muted)' }}>
                        {c.case_id.length > 24 ? c.case_id.slice(0, 24) + '…' : c.case_id}
                      </td>
                      <td style={{ padding: '3px 6px' }}>{c.class}</td>
                      <td style={{ padding: '3px 6px', color: 'var(--muted)' }}>{expLabel}</td>
                      <td style={{ padding: '3px 6px', color: riskColor, fontWeight: 600 }}>{c.observed_risk}</td>
                      <td style={{ padding: '3px 6px', textAlign: 'right', color: 'var(--muted)' }}>{(c.confidence * 100).toFixed(0)}%</td>
                      <td style={{ padding: '3px 6px', textAlign: 'right', color: 'var(--muted)' }}>{c.latency_ms.toFixed(0)}</td>
                      <td style={{ padding: '3px 6px', textAlign: 'right' }}>
                        <span style={{ color: c.pass ? '#22c55e' : '#ef4444', fontWeight: 700 }}>{c.pass ? '✓' : '✗'}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </>
  );
}

// ── History panel ──────────────────────────────────────────────────────────

const PAGE_SIZE = 50;

function HistoryPanel() {
  const [driverFilter, setDriverFilter] = useState('');
  const [trackFilter, setTrackFilter]   = useState('');
  const [riskFilter, setRiskFilter]     = useState('');
  const [insights, setInsights]         = useState([]);
  const [trend, setTrend]               = useState(null);
  const [heatmap, setHeatmap]           = useState(null);
  const [loading, setLoading]           = useState(false);
  const [offset, setOffset]             = useState(0);

  async function load(off = 0) {
    setLoading(true);
    const p = new URLSearchParams({ limit: String(PAGE_SIZE), offset: String(off) });
    if (driverFilter) p.set('driver_id', driverFilter);
    if (trackFilter)  p.set('track_id',  trackFilter);
    if (riskFilter)   p.set('risk',      riskFilter);
    const data = await fetch(`/api/v1/insights/history?${p}`).then(r => r.json()).catch(() => []);
    setInsights(Array.isArray(data) ? data : []);
    setOffset(off);
    setLoading(false);
  }

  async function loadTrend(driver) {
    if (!driver) { setTrend(null); return; }
    const d = await fetch(`/api/v1/insights/trend/${driver}`).then(r => r.json()).catch(() => null);
    setTrend(d);
  }

  async function loadHeatmap(track) {
    if (!track) { setHeatmap(null); return; }
    const d = await fetch(`/api/v1/insights/circuit/${track}`).then(r => r.json()).catch(() => null);
    setHeatmap(d);
  }

  useEffect(() => { load(0); }, []);

  const RC = { INFO: '#64748b', WATCH: '#f59e0b', WARNING: '#f97316', CRITICAL: '#ef4444' };

  return (
    <div className="grid">
      <div className="card input-card">
        <div className="input-header"><h2><History size={14} /> Filters</h2></div>
        <div className="stats-form">
          <Section title="Search">
            <div className="context-row">
              <label style={{ flex: 2 }}><span>Driver ID</span>
                <input className="text-input" value={driverFilter}
                  onChange={e => setDriverFilter(e.target.value)} placeholder="e.g. VER" />
              </label>
              <label style={{ flex: 2 }}><span>Track</span>
                <input className="text-input" value={trackFilter}
                  onChange={e => setTrackFilter(e.target.value)} placeholder="e.g. silverstone" />
              </label>
            </div>
            <div className="context-row">
              <label style={{ flex: 2 }}><span>Risk level</span>
                <select className="text-input" value={riskFilter} onChange={e => setRiskFilter(e.target.value)}>
                  <option value="">All</option>
                  {['INFO','WATCH','WARNING','CRITICAL'].map(r => <option key={r} value={r}>{r}</option>)}
                </select>
              </label>
            </div>
          </Section>
        </div>
        <button className="analyze-btn" onClick={() => { load(0); loadTrend(driverFilter); loadHeatmap(trackFilter); }}
          disabled={loading}>
          {loading ? <><Activity size={13} /> Loading…</> : <><Search size={13} /> Search</>}
        </button>

        {trend && (
          <>
            <h4 style={{ marginTop: 16 }}><TrendingUp size={13} /> Driver Trend — {trend.driver_id}</h4>
            <div className="feat-table">
              {Object.entries(trend.by_risk ?? {}).map(([r, v]) => (
                <div key={r} className="feat-row">
                  <span className="feat-key" style={{ color: RC[r] ?? '#aaa' }}>{r}</span>
                  <span className="feat-val">{v.count} · {(v.avg_confidence * 100).toFixed(0)}% conf</span>
                </div>
              ))}
              <div className="feat-row">
                <span className="feat-key" style={{ color: 'var(--muted)' }}>Total</span>
                <span className="feat-val">{trend.total}</span>
              </div>
            </div>
          </>
        )}

        {heatmap && heatmap.rows?.length > 0 && (
          <>
            <h4 style={{ marginTop: 16 }}><Table size={13} /> Circuit Heatmap — {heatmap.track_id}</h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
                <thead>
                  <tr>{['Driver','Risk','Count','Conf'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '2px 6px', color: 'var(--muted)', fontWeight: 600 }}>{h}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {heatmap.rows.map((r, i) => (
                    <tr key={i} style={{ borderTop: '1px solid var(--card-border)' }}>
                      <td style={{ padding: '3px 6px' }}>{r.driver_id}</td>
                      <td style={{ padding: '3px 6px', color: RC[r.risk] ?? '#aaa', fontWeight: 600 }}>{r.risk}</td>
                      <td style={{ padding: '3px 6px', textAlign: 'right' }}>{r.count}</td>
                      <td style={{ padding: '3px 6px', textAlign: 'right', color: 'var(--muted)' }}>{(r.avg_confidence * 100).toFixed(0)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>

      <div className="card insight-card">
        <h2><Table size={14} /> Insight History</h2>
        {insights.length === 0
          ? <p className="muted empty-hint">{loading ? 'Loading…' : 'No insights found. Run analysis to populate history.'}</p>
          : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
                <thead>
                  <tr>{['Driver','Track','Lap','Compound','Risk','Conf','Judge','Policy','Time'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '3px 6px', color: 'var(--muted)', fontWeight: 600, whiteSpace: 'nowrap' }}>{h}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {insights.map((ins, i) => {
                    const jm = ins.judge_mean;
                    const jColor = jm == null ? '#64748b' : jm >= 0.75 ? '#22c55e' : jm >= 0.55 ? '#f59e0b' : '#ef4444';
                    return (
                    <tr key={ins.insight_id ?? i} style={{ borderTop: '1px solid var(--card-border)' }}>
                      <td style={{ padding: '3px 6px', fontFamily: 'monospace' }}>{ins.driver_id}</td>
                      <td style={{ padding: '3px 6px', color: 'var(--muted)' }}>{ins.track_id}</td>
                      <td style={{ padding: '3px 6px', textAlign: 'right' }}>{ins.lap ?? '—'}</td>
                      <td style={{ padding: '3px 6px', color: COMPOUND_COLOR[ins.compound] ?? '#aaa' }}>{ins.compound ?? '—'}</td>
                      <td style={{ padding: '3px 6px', color: RC[ins.risk] ?? '#aaa', fontWeight: 600 }}>{ins.risk}</td>
                      <td style={{ padding: '3px 6px', textAlign: 'right' }}>{ins.confidence != null ? `${(ins.confidence * 100).toFixed(0)}%` : '—'}</td>
                      <td style={{ padding: '3px 6px', textAlign: 'right', color: jColor, fontWeight: jm != null ? 600 : 400 }}>
                        {jm != null ? `${(jm * 100).toFixed(0)}%` : '—'}
                      </td>
                      <td style={{ padding: '3px 6px', color: 'var(--muted)' }}>{ins.policy}</td>
                      <td style={{ padding: '3px 6px', color: 'var(--muted)', whiteSpace: 'nowrap' }}>
                        {ins.created_at ? new Date(ins.created_at).toLocaleTimeString() : '—'}
                      </td>
                    </tr>
                    );
                  })}
                </tbody>
              </table>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8 }}>
                <button className="kb-btn" onClick={() => load(Math.max(0, offset - PAGE_SIZE))} disabled={offset === 0 || loading}>
                  ← Prev
                </button>
                <span style={{ fontSize: 11, color: 'var(--muted)' }}>
                  {offset + 1}–{offset + insights.length}
                </span>
                <button className="kb-btn" onClick={() => load(offset + PAGE_SIZE)}
                  disabled={insights.length < PAGE_SIZE || loading}>
                  Next →
                </button>
              </div>
            </div>
          )
        }
      </div>
    </div>
  );
}

// ── Drift status card ──────────────────────────────────────────────────────

function DriftStatusCard() {
  const [drift, setDrift]     = useState(null);
  const [loading, setLoading] = useState(true);

  function refresh() {
    setLoading(true);
    fetch('/api/v1/drift/status').then(r => r.ok ? r.json() : null)
      .then(d => { setDrift(d); setLoading(false); }).catch(() => setLoading(false));
  }

  useEffect(() => { refresh(); }, []);

  const features = drift?.features ? Object.entries(drift.features) : [];
  const alerted  = drift?.alerted_features ?? [];

  return (
    <div className="card" style={{ minWidth: 0 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <h2 style={{ margin: 0 }}><Activity size={14} /> Feature Drift</h2>
        <button className="kb-btn" onClick={refresh} disabled={loading}>
          <RefreshCw size={11} className={loading ? 'spin' : ''} /> Refresh
        </button>
      </div>

      {!drift?.ready ? (
        <div>
          <p className="muted" style={{ fontSize: 12, marginBottom: 6 }}>
            Warming up baseline — {drift?.baseline_size ?? 0} / {drift?.min_baseline ?? 50} observations collected.
          </p>
          <div style={{ height: 4, borderRadius: 2, background: 'var(--card-border)', overflow: 'hidden' }}>
            <div style={{ width: `${Math.min(100, ((drift?.baseline_size ?? 0) / (drift?.min_baseline ?? 50)) * 100)}%`, height: '100%', background: '#3b82f6', borderRadius: 2 }} />
          </div>
          <p className="muted" style={{ fontSize: 10, marginTop: 4 }}>Run analyses in Telemetry or Session tabs to build the baseline.</p>
        </div>
      ) : (
        <>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
            <span style={{ fontSize: 11, color: alerted.length > 0 ? '#ef4444' : '#22c55e', fontWeight: 700 }}>
              {alerted.length > 0 ? `⚠ ${alerted.length} feature${alerted.length > 1 ? 's' : ''} drifting` : '✓ No drift detected'}
            </span>
            <span className="muted" style={{ fontSize: 10 }}>baseline {drift.baseline_size} obs · updated {drift.last_updated?.slice(11, 19) ?? '—'}</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {features.sort(([, a], [, b]) => Math.abs(b.z_score) - Math.abs(a.z_score)).map(([feat, info]) => {
              const pct = Math.min(100, (Math.abs(info.z_score) / 5) * 100);
              const color = info.alerted ? '#ef4444' : Math.abs(info.z_score) > 2 ? '#f59e0b' : '#22c55e';
              return (
                <div key={feat} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ fontSize: 10, color: 'var(--muted)', width: 140, flexShrink: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={feat}>{feat}</span>
                  <div style={{ flex: 1, height: 4, borderRadius: 2, background: 'var(--card-border)', overflow: 'hidden' }}>
                    <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 2 }} />
                  </div>
                  <span style={{ fontSize: 10, color, width: 36, textAlign: 'right', fontFamily: 'monospace' }}>
                    {info.z_score > 0 ? '+' : ''}{info.z_score.toFixed(2)}σ
                  </span>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

// ── Calibration health card ────────────────────────────────────────────────

function CalibrationCard() {
  const [stats, setStats]       = useState(null);
  const [retraining, setRetraining] = useState(false);
  const [result, setResult]     = useState(null);
  const [error, setError]       = useState('');

  function load() {
    fetch('/api/v1/feedback/stats').then(r => r.ok ? r.json() : null).then(d => { if (d) setStats(d); }).catch(() => {});
  }

  useEffect(() => { load(); }, []);

  async function retrain() {
    setRetraining(true); setResult(null); setError('');
    try {
      const res = await fetch('/api/v1/calibrator/retrain', { method: 'POST' });
      const d = await res.json();
      if (!res.ok) throw new Error(d.detail ?? `Error ${res.status}`);
      setResult(d);
      load();
    } catch (e) {
      setError(e.message);
    } finally {
      setRetraining(false);
    }
  }

  const pct = stats ? Math.min(100, (stats.total / stats.min_for_retrain) * 100) : 0;
  const eceColor = stats?.current_ece == null ? '#64748b' : stats.current_ece <= 0.05 ? '#22c55e' : stats.current_ece <= 0.15 ? '#f59e0b' : '#ef4444';

  return (
    <div className="card" style={{ minWidth: 0 }}>
      <h2 style={{ marginBottom: 10 }}><TrendingUp size={14} /> Calibration Health</h2>

      {stats?.regression_detected && (
        <div style={{ padding: '6px 10px', background: 'rgba(245,158,11,0.12)', border: '1px solid rgba(245,158,11,0.35)', borderRadius: 6, fontSize: 11, color: '#f59e0b' }}>
          ⚠ Last retrain blocked — ECE regressed above previous model. Live calibrator unchanged.
        </div>
      )}

      {stats && (
        <>
          <div style={{ display: 'flex', gap: 16, marginBottom: 12, flexWrap: 'wrap' }}>
            <div>
              <div style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 2 }}>ECE</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: eceColor, fontFamily: 'monospace' }}>
                {stats.current_ece != null ? stats.current_ece.toFixed(4) : '—'}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 2 }}>Brier</div>
              <div style={{ fontSize: 18, fontWeight: 700, fontFamily: 'monospace' }}>
                {stats.current_brier != null ? stats.current_brier.toFixed(4) : '—'}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 2 }}>Feedback</div>
              <div style={{ fontSize: 18, fontWeight: 700 }}>{stats.total}</div>
            </div>
            <div>
              <div style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 2 }}>Avg Rating</div>
              <div style={{ fontSize: 18, fontWeight: 700 }}>{stats.avg_rating != null ? `${stats.avg_rating}/5` : '—'}</div>
            </div>
          </div>

          <div style={{ marginBottom: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--muted)', marginBottom: 3 }}>
              <span>Feedback for retrain</span>
              <span>{stats.total} / {stats.min_for_retrain}</span>
            </div>
            <div style={{ height: 4, borderRadius: 2, background: 'var(--card-border)', overflow: 'hidden' }}>
              <div style={{ width: `${pct}%`, height: '100%', background: pct >= 100 ? '#22c55e' : '#3b82f6', borderRadius: 2 }} />
            </div>
          </div>

          {stats.last_retrain && (
            <p style={{ fontSize: 10, color: 'var(--muted)', margin: '0 0 10px' }}>
              Last retrained {stats.last_retrain.slice(0, 10)}
              {stats.retrain_dataset?.generator && ` · ${stats.retrain_dataset.generator}`}
              {stats.retrain_dataset?.n_feedback && ` · ${stats.retrain_dataset.n_feedback} feedback samples`}
            </p>
          )}

          <button className="analyze-btn" style={{ padding: '6px 14px', fontSize: 12 }}
            onClick={retrain} disabled={retraining || !stats.ready_to_retrain}
            title={!stats.ready_to_retrain ? `Need ${stats.min_for_retrain - stats.total} more feedback records` : 'Re-calibrate from feedback'}>
            {retraining ? <><Activity size={12} className="spin" /> Retraining…</> : 'Re-calibrate from feedback'}
          </button>
          {!stats.ready_to_retrain && (
            <p style={{ fontSize: 10, color: 'var(--muted)', marginTop: 4 }}>
              Rate insights in Telemetry / Session to build up feedback.
            </p>
          )}

          {result && !result.skipped && (
            result.regression_detected ? (
              <div style={{ marginTop: 8, padding: '6px 10px', background: 'rgba(245,158,11,0.1)', border: '1px solid rgba(245,158,11,0.3)', borderRadius: 6, fontSize: 11, color: '#f59e0b' }}>
                ⚠ Retrain blocked — new ECE {result.ece?.toFixed(4)} regressed from {result.previous_ece?.toFixed(4)}. Versioned model saved; live calibrator unchanged.
              </div>
            ) : (
              <div style={{ marginTop: 8, padding: '6px 10px', background: 'rgba(34,197,94,0.1)', borderRadius: 6, fontSize: 11 }}>
                ✓ Retrained — ECE {result.ece?.toFixed(4)} · Brier {result.brier_score?.toFixed(4)} · {result.n_feedback} feedback samples
              </div>
            )
          )}
          {result?.skipped && (
            <p style={{ fontSize: 11, color: '#f59e0b', marginTop: 6 }}>Skipped: {result.reason}</p>
          )}
          {error && <p style={{ fontSize: 11, color: '#ef4444', marginTop: 6 }}>{error}</p>}
        </>
      )}
    </div>
  );
}

// ── Analytics panel ────────────────────────────────────────────────────────

const ANALYTICS_SUGGESTIONS = [
  'How many WARNING insights occurred per driver this week?',
  'Which track has the highest average confidence score?',
  'Show average tire wear by compound',
  'List the 5 most recent CRITICAL insights',
];

function AnalyticsPanel() {
  const [question, setQuestion]   = useState('');
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [schema, setSchema]       = useState(null);
  const [showSchema, setShowSchema] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    fetch('/api/v1/analytics/schema').then(r => r.ok ? r.json() : null).then(d => { if (d) setSchema(d); }).catch(() => {});
  }, []);

  async function ask() {
    if (!question.trim()) return;
    setLoading(true); setResult(null);
    try {
      const r = await fetch('/api/v1/analytics/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });
      const d = await r.json();
      setResult(r.ok ? d : { error: d.detail ?? d.error ?? `Server error ${r.status}` });
    } catch (e) {
      setResult({ error: e.message });
    }
    setLoading(false);
  }

  const rows = result?.results ?? [];
  const cols = rows.length > 0 ? Object.keys(rows[0]) : [];

  return (
    <div className="grid" style={{ gridTemplateColumns: '1fr' }}>
      <div style={{ gridColumn: '1 / -1', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <DriftStatusCard />
        <CalibrationCard />
      </div>
      <div className="card" style={{ gridColumn: '1 / -1' }}>
        <div className="input-header"><h2><Database size={14} /> Text-to-SQL Analytics</h2></div>

        <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
          <input ref={inputRef} className="chat-input" value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && ask()}
            placeholder="Ask a question about insight history… e.g. 'how many WARNINGs per driver?'" />
          <button className="send-btn" onClick={ask} disabled={loading || !question.trim()}>
            {loading ? <Activity size={14} className="spin" /> : <Send size={14} />}
          </button>
        </div>

        {!result && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 12 }}>
            {ANALYTICS_SUGGESTIONS.map(s => (
              <button key={s} className="suggestion-chip" onClick={() => { setQuestion(s); setTimeout(() => inputRef.current?.focus(), 0); }}>
                {s}
              </button>
            ))}
          </div>
        )}

        {result?.error && <pre className="error">{result.error}</pre>}

        {result && !result.error && (
          <>
            <details style={{ marginBottom: 10 }}>
              <summary style={{ fontSize: 11, color: 'var(--muted)', cursor: 'pointer' }}>
                SQL ({result.model ?? 'unknown'} · {result.latency_ms?.toFixed(0) ?? '?'}ms)
              </summary>
              <pre style={{ fontSize: 11, background: 'var(--card-border)', padding: 8, borderRadius: 4, marginTop: 6, overflowX: 'auto' }}>
                {result.sql}
              </pre>
            </details>
            {rows.length === 0
              ? <p className="muted">Query returned 0 rows.</p>
              : (
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                    <thead>
                      <tr>{cols.map(c => (
                        <th key={c} style={{ textAlign: 'left', padding: '4px 8px', color: 'var(--muted)', fontWeight: 600, borderBottom: '1px solid var(--card-border)' }}>{c}</th>
                      ))}</tr>
                    </thead>
                    <tbody>
                      {rows.map((row, i) => (
                        <tr key={i} style={{ borderTop: '1px solid var(--card-border)' }}>
                          {cols.map(c => (
                            <td key={c} style={{ padding: '4px 8px', fontFamily: typeof row[c] === 'number' ? 'monospace' : undefined }}>
                              {row[c] == null ? '—' : String(row[c])}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 6 }}>
                    <p style={{ fontSize: 11, color: 'var(--muted)', margin: 0 }}>{result.row_count} row{result.row_count !== 1 ? 's' : ''}</p>
                    <button className="kb-btn" onClick={() => {
                      const header = cols.join(',');
                      const body = rows.map(r => cols.map(c => JSON.stringify(r[c] ?? '')).join(',')).join('\n');
                      const blob = new Blob([header + '\n' + body], { type: 'text/csv' });
                      const a = document.createElement('a');
                      a.href = URL.createObjectURL(blob);
                      a.download = 'f1di_analytics.csv';
                      a.click();
                    }}>
                      Export CSV
                    </button>
                  </div>
                </div>
              )
            }
          </>
        )}

        {schema && (
          <div style={{ marginTop: 12 }}>
            <button className="suggestion-chip" onClick={() => setShowSchema(s => !s)}>
              {showSchema ? 'Hide schema' : 'Show schema'}
            </button>
            {showSchema && (
              <pre style={{ fontSize: 11, background: 'var(--card-border)', padding: 8, borderRadius: 4, marginTop: 6, overflowX: 'auto', whiteSpace: 'pre-wrap' }}>
                {schema.schema}
              </pre>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function RegressionPanel() {
  const [fixtures, setFixtures] = useState([]);
  const [fixture, setFixture]   = useState('');
  const [report, setReport]     = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState('');

  useEffect(() => {
    fetch('/api/v1/regression/fixtures')
      .then(r => r.ok ? r.json() : [])
      .then(fs => { setFixtures(fs); if (fs.length) setFixture(fs[0]); })
      .catch(() => {});
  }, []);

  async function run() {
    if (!fixture) return;
    setLoading(true); setError(''); setReport(null);
    try {
      const res = await fetch(`/api/v1/regression/run?fixture=${encodeURIComponent(fixture)}`, { method: 'POST' });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      setReport(await res.json());
    } catch (e) {
      setError(String(e.message ?? e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid">
      <div className="card input-card">
        <div className="input-header">
          <h2><FlaskConical size={14} /> Regression Gates</h2>
        </div>
        <div className="stats-form">
          <Section title="Fixture">
            <div className="context-row">
              <label style={{ flex: 2 }}><span>File</span>
                <select className="text-input" value={fixture} onChange={e => setFixture(e.target.value)}>
                  {fixtures.length === 0 && <option value="">No fixtures found</option>}
                  {fixtures.map(f => <option key={f} value={f}>{f}</option>)}
                </select>
              </label>
            </div>
          </Section>
          <Section title="Gate Thresholds">
            <div className="feat-table">
              {[
                ['Case recall',         '≥ 100%'],
                ['False positive rate', '= 0%'],
                ['Agent activation',    '≥ 100%'],
                ['Evidence present',    'All cases'],
                ['Source retrieval',    '≥ 100%'],
                ['Policy correctness',  '≥ 100%'],
              ].map(([label, threshold]) => (
                <div key={label} className="feat-row">
                  <span className="feat-key">{label}</span>
                  <span className="feat-val" style={{ color: 'var(--muted)' }}>{threshold}</span>
                </div>
              ))}
            </div>
          </Section>
        </div>
        <button className="analyze-btn" onClick={run} disabled={loading || !fixture}>
          {loading ? <><Activity size={13} /> Running…</> : 'Run Regression'}
        </button>
        {error && <pre className="error">{error}</pre>}
      </div>

      <div className="card insight-card">
        <h2><FlaskConical size={15} /> Results</h2>
        {!report
          ? <p className="muted empty-hint">Select a fixture and click Run Regression.</p>
          : <RegressionReport report={report} />}
      </div>
    </div>
  );
}

// ── Model Lab panel ───────────────────────────────────────────────────────

function RiskDistBar({ distribution, total }) {
  if (!total) return <p className="muted" style={{ fontSize: 11 }}>No data yet</p>;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {['CRITICAL', 'WARNING', 'WATCH', 'INFO'].map(r => {
        const n = (distribution ?? {})[r] ?? 0;
        const pct = (n / total) * 100;
        const c = RISK_META[r]?.color ?? '#64748b';
        return (
          <div key={r} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 10, color: c, width: 58, flexShrink: 0 }}>{r}</span>
            <div style={{ flex: 1, height: 5, borderRadius: 2, background: 'var(--card-border)', overflow: 'hidden' }}>
              <div style={{ width: `${pct}%`, height: '100%', background: c }} />
            </div>
            <span style={{ fontSize: 10, color: 'var(--muted)', width: 24, textAlign: 'right' }}>{n}</span>
          </div>
        );
      })}
    </div>
  );
}

function ClassifierHistoryChart({ history, agent }) {
  const agentHistory = (history ?? []).filter(e => e.agent === agent);
  if (agentHistory.length < 2) return (
    <p className="muted" style={{ fontSize: 10 }}>Need ≥2 fits to show trend.</p>
  );
  const W = 1000, H = 60;
  const n = agentHistory.length;
  const xs = agentHistory.map((_, i) => (i / Math.max(n - 1, 1)) * W);
  const accVals  = agentHistory.map(e => e.accuracy);
  const brierVals = agentHistory.map(e => e.brier_score);

  function sparkline(vals, color, invert = false) {
    const finite = vals.filter(v => v != null);
    if (finite.length < 2) return null;
    const lo = Math.min(...finite), hi = Math.max(...finite);
    const range = hi - lo || 0.001;
    const pts = vals.map((v, i) => {
      if (v == null) return null;
      const y = invert
        ? 4 + ((v - lo) / range) * (H - 8)
        : H - 4 - ((v - lo) / range) * (H - 8);
      return `${xs[i]},${y}`;
    }).filter(Boolean).join(' ');
    return <polyline points={pts} fill="none" stroke={color} strokeWidth="2.5" vectorEffect="non-scaling-stroke" />;
  }

  const latest = agentHistory[agentHistory.length - 1];
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none"
        style={{ width: '100%', height: 56, background: '#0a0f1e', borderRadius: 4, display: 'block' }}>
        {sparkline(accVals, '#22c55e', false)}
        {sparkline(brierVals, '#f59e0b', true)}
        {xs.map((x, i) => (
          <line key={i} x1={x} y1={0} x2={x} y2={H} stroke="rgba(255,255,255,0.04)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
        ))}
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--muted)', marginTop: 3 }}>
        <span>{agentHistory[0].fitted_at?.slice(0, 10)}</span>
        <div style={{ display: 'flex', gap: 10 }}>
          <span style={{ color: '#22c55e' }}>— acc (↑)</span>
          <span style={{ color: '#f59e0b' }}>— brier (↓)</span>
          <span>latest: acc {latest.accuracy?.toFixed(3)}  brier {latest.brier_score?.toFixed(3)}  real {latest.n_real}</span>
        </div>
        <span>{latest.fitted_at?.slice(0, 10)}</span>
      </div>
    </div>
  );
}

function ClassifierModelsPanel({ clfHistory }) {
  const AGENTS = ['tire', 'battery', 'weather', 'telemetry', 'safety_car', 'fuel'];
  const [selectedAgent, setSelectedAgent] = useState('tire');
  const [snapshots, setSnapshots]         = useState([]);
  const [testResult, setTestResult]       = useState(null);
  const [promoting, setPromoting]         = useState(null);
  const [promoteResult, setPromoteResult] = useState(null);
  const [testing, setTesting]             = useState(null);
  const [error, setError]                 = useState('');

  async function loadSnapshots(agent) {
    const r = await fetch(`/api/v1/model/snapshots/${agent}`).catch(() => null);
    if (r?.ok) setSnapshots(await r.json());
    else setSnapshots([]);
    setTestResult(null);
    setPromoteResult(null);
    setError('');
  }

  useEffect(() => { loadSnapshots(selectedAgent); }, [selectedAgent]);

  async function runTest(snapshot) {
    setTesting(snapshot.filename); setTestResult(null); setError('');
    const r = await fetch('/api/v1/model/test', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ agent: selectedAgent, snapshot_path: snapshot.path }),
    }).catch(() => null);
    if (r?.ok) setTestResult(await r.json());
    else setError(`Test failed: ${r?.status}`);
    setTesting(null);
  }

  async function runPromote(snapshot) {
    if (!window.confirm(`Promote ${snapshot.filename} to live ${selectedAgent} classifier?`)) return;
    setPromoting(snapshot.filename); setPromoteResult(null); setError('');
    const r = await fetch('/api/v1/model/promote', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ agent: selectedAgent, snapshot_path: snapshot.path }),
    }).catch(() => null);
    if (r?.ok) setPromoteResult(await r.json());
    else setError(`Promote failed: ${r?.status}`);
    setPromoting(null);
    loadSnapshots(selectedAgent);
  }

  const liveName = snapshots.length > 0
    ? `${selectedAgent}_classifier.pkl`
    : null;

  return (
    <div>
      {/* Agent tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 12, flexWrap: 'wrap' }}>
        {AGENTS.map(a => (
          <button key={a} onClick={() => setSelectedAgent(a)} style={{
            fontSize: 10, padding: '3px 10px', borderRadius: 10, cursor: 'pointer',
            background: selectedAgent === a ? '#1e3a5f' : '#0d1b2e',
            color: selectedAgent === a ? '#93c5fd' : '#64748b',
            border: `1px solid ${selectedAgent === a ? '#3b82f6' : '#334155'}`,
            textTransform: 'capitalize',
          }}>{a}</button>
        ))}
      </div>

      {/* Trend sparkline for selected agent */}
      <ClassifierHistoryChart history={clfHistory} agent={selectedAgent} />

      {/* Snapshots */}
      <div style={{ marginTop: 12 }}>
        <p style={{ fontSize: 10, color: 'var(--muted)', margin: '0 0 6px', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>
          Snapshots — {selectedAgent}
        </p>
        {snapshots.length === 0 && (
          <p className="muted" style={{ fontSize: 11 }}>No snapshots found. Run <code style={{ fontSize: 10 }}>make fit-{selectedAgent}</code>.</p>
        )}
        {snapshots.map((snap, i) => {
          const isLive = i === 0;
          const isTesting   = testing === snap.filename;
          const isPromoting = promoting === snap.filename;
          const tested = testResult?.snapshot === snap.filename ? testResult : null;
          const promoted = promoteResult?.snapshot === snap.filename ? promoteResult : null;
          return (
            <div key={snap.filename} style={{
              padding: '7px 9px', borderRadius: 6, marginBottom: 5,
              background: isLive ? '#0a1e0f' : '#0a0f1e',
              border: `1px solid ${isLive ? '#166534' : '#1e293b'}`,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  {isLive && <span style={{ fontSize: 8, padding: '1px 5px', borderRadius: 3, background: '#0f2418', color: '#4ade80', border: '1px solid #166534', fontFamily: 'monospace' }}>LIVE</span>}
                  {snap.model_version && <span style={{ fontSize: 8, padding: '1px 5px', borderRadius: 3, background: '#0d1b2e', color: '#93c5fd', border: '1px solid #1e3a5f', fontFamily: 'monospace' }}>{snap.model_version}</span>}
                  <span style={{ fontSize: 10, color: '#94a3b8', fontFamily: 'monospace' }}>{snap.fitted_at?.replace('T', ' ').replace('Z', '')}</span>
                </div>
                <div style={{ display: 'flex', gap: 4 }}>
                  <button className="kb-btn" onClick={() => runTest(snap)} disabled={isTesting || isPromoting}
                    style={{ fontSize: 10, padding: '2px 8px' }}>
                    {isTesting ? '…' : 'Test'}
                  </button>
                  {!isLive && (
                    <button className="kb-btn" onClick={() => runPromote(snap)} disabled={isTesting || isPromoting}
                      style={{ fontSize: 10, padding: '2px 8px', color: '#4ade80', borderColor: '#166534' }}>
                      {isPromoting ? '…' : 'Promote'}
                    </button>
                  )}
                </div>
              </div>
              <div style={{ display: 'flex', gap: 14, marginTop: 4, fontSize: 10, color: '#64748b', fontFamily: 'monospace' }}>
                {snap.accuracy != null && <span>acc {snap.accuracy.toFixed(3)}</span>}
                {snap.brier_score != null && <span>brier {snap.brier_score.toFixed(3)}</span>}
                {snap.n_real != null && <span>real {snap.n_real}</span>}
                {snap.n_train != null && <span>train {snap.n_train}</span>}
              </div>
              {tested && (
                <div style={{ marginTop: 5, padding: '4px 8px', borderRadius: 4, background: '#0d1b2e', border: '1px solid #1e3a5f', fontSize: 10, fontFamily: 'monospace', color: '#93c5fd' }}>
                  held-out test (n={tested.test_n}): acc {tested.test_accuracy.toFixed(3)}  brier {tested.test_brier.toFixed(3)}
                  {Math.abs(tested.test_accuracy - tested.train_accuracy) > 0.05 && (
                    <span style={{ color: '#f59e0b', marginLeft: 8 }}>⚠ train/test gap {((tested.train_accuracy - tested.test_accuracy) * 100).toFixed(1)}pp</span>
                  )}
                </div>
              )}
              {promoted && (
                <div style={{ marginTop: 5, padding: '4px 8px', borderRadius: 4, background: '#0a1e0f', border: '1px solid #166534', fontSize: 10, color: '#4ade80' }}>
                  ✓ Promoted — prev acc {promoted.prev_accuracy?.toFixed(3) ?? '—'} → {promoted.accuracy.toFixed(3)}
                </div>
              )}
            </div>
          );
        })}
        {error && <p style={{ fontSize: 10, color: '#ef4444', marginTop: 4 }}>{error}</p>}
      </div>
    </div>
  );
}

function QualityTrendChart({ history }) {
  if (!history || history.length < 2) return (
    <p className="muted" style={{ fontSize: 11 }}>
      Need ≥2 snapshots to show trend. Record snapshots via the flywheel or "Record now".
    </p>
  );

  const W = 1000, H = 80;
  const n = history.length;
  const xs = history.map((_, i) => (i / (n - 1)) * W);

  const eceVals  = history.map(h => h.calibration?.ece);
  const mrrVals  = history.map(h => h.retrieval?.mrr);

  function sparkline(vals, color, invert = false) {
    const finite = vals.filter(v => v != null && isFinite(v));
    if (finite.length < 2) return null;
    const lo = Math.min(...finite), hi = Math.max(...finite);
    const range = hi - lo || 0.001;
    const pts = vals.map((v, i) => {
      if (v == null) return null;
      const y = invert
        ? 4 + ((v - lo) / range) * (H - 8)      // higher = worse (ECE)
        : H - 4 - ((v - lo) / range) * (H - 8); // higher = better (MRR)
      return `${xs[i]},${y}`;
    }).filter(Boolean).join(' ');
    return <polyline points={pts} fill="none" stroke={color} strokeWidth="2.5" vectorEffect="non-scaling-stroke" />;
  }

  const labels = history.map(h => h.recorded_at?.slice(0, 10) ?? '');

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none"
        style={{ width: '100%', height: 72, background: '#0a0f1e', borderRadius: 4, display: 'block' }}>
        <line x1="0" y1={H / 2} x2={W} y2={H / 2} stroke="rgba(255,255,255,0.06)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
        {sparkline(eceVals, '#f59e0b', true)}
        {sparkline(mrrVals, '#22c55e', false)}
        {xs.map((x, i) => (
          <line key={i} x1={x} y1={0} x2={x} y2={H} stroke="rgba(255,255,255,0.04)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
        ))}
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.62rem', color: 'var(--muted)', marginTop: 2 }}>
        <span>{labels[0]}</span>
        <div style={{ display: 'flex', gap: 12 }}>
          <span style={{ color: '#f59e0b' }}>— ECE (↓ better)</span>
          <span style={{ color: '#22c55e' }}>— MRR (↑ better)</span>
        </div>
        <span>{labels[labels.length - 1]}</span>
      </div>
      <div style={{ display: 'flex', gap: 16, marginTop: 8, flexWrap: 'wrap' }}>
        {history.slice(-3).reverse().map((h, i) => (
          <div key={i} style={{ fontSize: 10, color: 'var(--muted)', background: '#0d1b2e', borderRadius: 4, padding: '4px 8px', border: '1px solid var(--card-border)' }}>
            <span style={{ fontWeight: 700 }}>{h.recorded_at?.slice(0, 10)}</span>
            {h.calibration?.ece != null && <span style={{ color: '#f59e0b', marginLeft: 6 }}>ECE {h.calibration.ece.toFixed(4)}</span>}
            {h.retrieval?.mrr != null && <span style={{ color: '#22c55e', marginLeft: 6 }}>MRR {(h.retrieval.mrr * 100).toFixed(0)}%</span>}
          </div>
        ))}
      </div>
    </div>
  );
}

function ModelLabPanel({ version }) {
  const [stats, setStats]                   = useState(DEFAULT_STATS);
  const [challengerVersion, setChallengerVersion] = useState('challenger');
  const [shadowResult, setShadowResult]     = useState(null);
  const [compareData, setCompareData]       = useState(null);
  const [evalData, setEvalData]             = useState(null);
  const [retrievalData, setRetrievalData]   = useState(null);
  const [qualityHistory, setQualityHistory] = useState([]);
  const [clfHistory, setClfHistory]         = useState([]);
  const [loading, setLoading]               = useState(false);
  const [promoting, setPromoting]           = useState(false);
  const [runningRetrieval, setRunningRetrieval] = useState(false);
  const [recordingSnapshot, setRecordingSnapshot] = useState(false);
  const [promoteResult, setPromoteResult]   = useState(null);
  const [error, setError]                   = useState('');

  async function refreshCompare(ver) {
    const v = ver ?? challengerVersion;
    const r = await fetch(`/api/v1/shadow/compare?challenger_version=${encodeURIComponent(v)}`).catch(() => null);
    if (r?.ok) setCompareData(await r.json());
  }

  async function refreshEval(ver) {
    const v = ver ?? challengerVersion;
    const r = await fetch(`/api/v1/shadow/evaluate?challenger_version=${encodeURIComponent(v)}`).catch(() => null);
    if (r?.ok) setEvalData(await r.json());
  }

  async function loadQualityHistory() {
    const r = await fetch('/api/v1/quality/history?limit=30').catch(() => null);
    if (r?.ok) setQualityHistory(await r.json());
  }

  async function loadClfHistory() {
    const r = await fetch('/api/v1/model/history?limit=100').catch(() => null);
    if (r?.ok) setClfHistory(await r.json());
  }

  async function recordSnapshot() {
    setRecordingSnapshot(true);
    await fetch('/api/v1/quality/record?trigger=manual', { method: 'POST' }).catch(() => null);
    await loadQualityHistory();
    setRecordingSnapshot(false);
  }

  useEffect(() => {
    refreshCompare();
    refreshEval();
    loadQualityHistory();
    loadClfHistory();
    fetch('/api/v1/eval/retrieval').then(r => r.ok ? r.json() : null).then(d => { if (d) setRetrievalData(d); }).catch(() => {});
  }, []);

  async function runShadow() {
    setLoading(true); setError(''); setShadowResult(null);
    try {
      const win = buildWindow(stats);
      const res = await fetch(`/api/v1/shadow/analyze?challenger_version=${encodeURIComponent(challengerVersion)}`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(win),
      });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      setShadowResult(await res.json());
      await Promise.all([refreshCompare(), refreshEval()]);
    } catch (e) {
      setError(String(e.message ?? e));
    }
    setLoading(false);
  }

  async function promote(force = false) {
    setPromoting(true); setPromoteResult(null);
    const r = await fetch(
      `/api/v1/shadow/promote?challenger_version=${encodeURIComponent(challengerVersion)}&force=${force}`,
      { method: 'POST' },
    );
    const d = await r.json();
    setPromoteResult(d);
    setPromoting(false);
    if (d.promoted) refreshEval();
  }

  async function runRetrievalEval() {
    setRunningRetrieval(true);
    setError('');
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 90000);
    try {
      const r = await fetch('/api/v1/eval/retrieval?save=true', { signal: ctrl.signal });
      if (r.ok) setRetrievalData(await r.json());
      else setError(`Eval failed: ${r.status} ${r.statusText}`);
    } catch (e) {
      setError(e.name === 'AbortError' ? 'Eval timed out (90s) — retrieval may be slow on first run' : `Eval error: ${e.message}`);
    } finally {
      clearTimeout(timer);
      setRunningRetrieval(false);
    }
  }

  const canPromote = evalData?.promote === true && !promoteResult?.promoted;
  const evalRec = evalData?.recommendation;
  const recColor = canPromote ? '#22c55e' : evalRec === 'insufficient_data' ? '#64748b' : '#f59e0b';

  return (
    <div className="grid">
      {/* Left: shadow analyze form */}
      <div className="card input-card">
        <div className="input-header">
          <h2><Radio size={14} /> Shadow Analyze</h2>
        </div>
        <div className="stats-form">
          <Section title="Challenger">
            <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span style={{ fontSize: 11, color: 'var(--muted)' }}>Version tag</span>
              <input className="text-input" value={challengerVersion}
                onChange={e => setChallengerVersion(e.target.value)}
                placeholder="challenger" />
            </label>
          </Section>
        </div>
        <StatsForm stats={stats} onChange={setStats} />
        <button className="analyze-btn" onClick={runShadow} disabled={loading}>
          {loading ? <><Activity size={13} /> Analyzing…</> : 'Run Shadow Analyze'}
        </button>
        {error && <pre className="error">{error}</pre>}
        {shadowResult && (
          <div style={{ marginTop: 10, padding: '8px 10px', background: '#0a1628', borderRadius: 6, border: '1px solid var(--card-border)', fontSize: 12 }}>
            <span style={{ color: RISK_META[shadowResult.risk]?.color ?? '#64748b', fontWeight: 700 }}>{shadowResult.risk}</span>
            <span className="muted" style={{ marginLeft: 8 }}>conf {(shadowResult.confidence * 100).toFixed(0)}% · stored as shadow</span>
          </div>
        )}
      </div>

      {/* Right: compare, evaluate, retrieval */}
      <div className="card insight-card" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

        {/* Compare */}
        <div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
            <h2 style={{ margin: 0, fontSize: 14 }}><BarChart2 size={13} /> Risk Distribution</h2>
            <button className="kb-btn" onClick={() => { refreshCompare(); refreshEval(); }}>
              <RefreshCw size={11} /> Refresh
            </button>
          </div>
          {compareData ? (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div>
                <p style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 6, fontWeight: 700 }}>
                  PRODUCTION · n={compareData.production?.n ?? 0}
                </p>
                <RiskDistBar distribution={compareData.production?.risk_distribution} total={compareData.production?.n ?? 0} />
                {compareData.production?.n > 0 && (
                  <p style={{ fontSize: 10, color: 'var(--muted)', marginTop: 6 }}>
                    avg conf {(compareData.production.avg_confidence * 100).toFixed(1)}%
                    · unc {(compareData.production.avg_uncertainty * 100).toFixed(1)}%
                  </p>
                )}
              </div>
              <div>
                <p style={{ fontSize: 10, color: '#38bdf8', marginBottom: 6, fontWeight: 700 }}>
                  CHALLENGER · {compareData.challenger_version} · n={compareData.shadow?.n ?? 0}
                </p>
                <RiskDistBar distribution={compareData.shadow?.risk_distribution} total={compareData.shadow?.n ?? 0} />
                {compareData.shadow?.n > 0 && (
                  <p style={{ fontSize: 10, color: 'var(--muted)', marginTop: 6 }}>
                    avg conf {(compareData.shadow.avg_confidence * 100).toFixed(1)}%
                    · unc {(compareData.shadow.avg_uncertainty * 100).toFixed(1)}%
                  </p>
                )}
              </div>
            </div>
          ) : (
            <p className="muted" style={{ fontSize: 12 }}>Run shadow analyses to populate comparison.</p>
          )}
        </div>

        <div style={{ borderTop: '1px solid var(--card-border)' }} />

        {/* Evaluate */}
        <div>
          <h2 style={{ margin: '0 0 10px', fontSize: 14 }}><TrendingUp size={13} /> Statistical Evaluation</h2>
          {evalData ? (
            evalData.recommendation === 'insufficient_data' ? (
              <div style={{ padding: '8px 12px', background: '#0d1b2e', borderRadius: 6, fontSize: 12 }}>
                <p style={{ color: '#64748b', margin: '0 0 4px' }}>Insufficient data — need ≥{evalData.min_n} shadow runs</p>
                <p className="muted" style={{ fontSize: 10, margin: 0 }}>
                  {evalData.n_shadow} shadow · {evalData.n_prod} production records
                </p>
              </div>
            ) : (
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                  <span style={{ fontSize: 12, fontWeight: 700, color: recColor, padding: '3px 10px', borderRadius: 12, background: recColor + '22', border: `1px solid ${recColor}55` }}>
                    {evalRec?.replace(/_/g, ' ').toUpperCase()}
                  </span>
                </div>
                <div className="feat-table" style={{ marginBottom: 12 }}>
                  {[
                    ['Shadow n',      evalData.n_shadow],
                    ['Prod n',        evalData.n_prod],
                    ['Shadow conf',   evalData.shadow_mean_confidence != null ? `${(evalData.shadow_mean_confidence * 100).toFixed(1)}%` : '—'],
                    ['Prod conf',     evalData.prod_mean_confidence != null ? `${(evalData.prod_mean_confidence * 100).toFixed(1)}%` : '—'],
                    ['p-value',       evalData.p_value != null ? evalData.p_value.toFixed(4) : '—'],
                    ['Effect (RBC)',  evalData.rank_biserial_correlation != null ? evalData.rank_biserial_correlation.toFixed(4) : '—'],
                    ['Escalation Δ', evalData.shadow_escalation_rate != null ? `${((evalData.shadow_escalation_rate - evalData.prod_escalation_rate) * 100).toFixed(1)}pp` : '—'],
                  ].map(([label, val]) => (
                    <div key={label} className="feat-row">
                      <span className="feat-key">{label}</span>
                      <span className="feat-val">{val}</span>
                    </div>
                  ))}
                </div>
                <button className="analyze-btn" style={{ padding: '6px 14px', fontSize: 12 }}
                  onClick={() => promote(false)} disabled={promoting || !canPromote}
                  title={!canPromote ? (promoteResult?.promoted ? 'Already promoted' : 'Evaluation does not recommend promotion') : 'Promote challenger to production'}>
                  {promoting ? <><Activity size={12} className="spin" /> Promoting…</> : canPromote ? 'Promote Challenger' : 'Promote (not recommended)'}
                </button>
                {!canPromote && !promoteResult && (
                  <p style={{ fontSize: 10, color: 'var(--muted)', marginTop: 4 }}>
                    Force-promote via <code style={{ fontSize: 10 }}>POST /v1/shadow/promote?force=true</code>
                  </p>
                )}
                {promoteResult && (
                  <div style={{ marginTop: 8, padding: '6px 10px', borderRadius: 6, fontSize: 11,
                    background: promoteResult.promoted ? 'rgba(34,197,94,0.1)' : 'rgba(245,158,11,0.1)',
                    border: `1px solid ${promoteResult.promoted ? 'rgba(34,197,94,0.3)' : 'rgba(245,158,11,0.3)'}`,
                    color: promoteResult.promoted ? '#22c55e' : '#f59e0b' }}>
                    {promoteResult.promoted
                      ? `✓ Promoted at ${promoteResult.promoted_at?.slice(0, 19).replace('T', ' ')}`
                      : `Not promoted — ${promoteResult.reason?.replace(/_/g, ' ')}`}
                  </div>
                )}
              </div>
            )
          ) : (
            <p className="muted" style={{ fontSize: 12 }}>Loading evaluation…</p>
          )}
        </div>

        <div style={{ borderTop: '1px solid var(--card-border)' }} />

        {/* Retrieval quality */}
        <div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
            <h2 style={{ margin: 0, fontSize: 14 }}><Search size={13} /> Retrieval Quality</h2>
            <button className="kb-btn" onClick={runRetrievalEval} disabled={runningRetrieval}>
              <RefreshCw size={11} className={runningRetrieval ? 'spin' : ''} />
              {runningRetrieval ? 'Running…' : 'Run eval'}
            </button>
          </div>
          {error && <p style={{ fontSize: 11, color: '#ef4444', margin: '0 0 8px' }}>{error}</p>}
          {retrievalData ? (
            <div>
              <div className="feat-table" style={{ marginBottom: 10 }}>
                {[
                  ['P@1',    retrievalData.precision_at_1],
                  ['P@3',    retrievalData.precision_at_3],
                  ['P@5',    retrievalData.precision_at_5],
                  ['R@3',    retrievalData.recall_at_3],
                  ['MRR',    retrievalData.mrr],
                  ['NDCG@5', retrievalData.ndcg_at_5],
                ].map(([label, val]) => {
                  const pct = val != null ? val * 100 : null;
                  const color = pct == null ? '#64748b' : pct >= 70 ? '#22c55e' : pct >= 45 ? '#f59e0b' : '#ef4444';
                  return (
                    <div key={label} className="feat-row">
                      <span className="feat-key">{label}</span>
                      <span className="feat-val" style={{ color, fontFamily: 'monospace' }}>
                        {pct != null ? `${pct.toFixed(1)}%` : '—'}
                      </span>
                    </div>
                  );
                })}
              </div>
              {retrievalData.per_topic && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 10 }}>
                  {Object.entries(retrievalData.per_topic).map(([topic, m]) => {
                    const mrr = m.mrr ?? 0;
                    const c = mrr >= 0.7 ? '#22c55e' : mrr >= 0.45 ? '#f59e0b' : '#ef4444';
                    return (
                      <span key={topic} style={{ fontSize: 10, padding: '2px 7px', borderRadius: 10,
                        background: c + '22', border: `1px solid ${c}55`, color: c }}>
                        {topic} MRR {(mrr * 100).toFixed(0)}%
                      </span>
                    );
                  })}
                </div>
              )}
              {retrievalData.query_results?.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 6, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Per-query results</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {retrievalData.query_results.map((qr, i) => {
                      const hit = qr['p@1'] >= 1.0;
                      const partial = !hit && qr['mrr'] > 0;
                      const borderColor = hit ? '#22c55e' : partial ? '#f59e0b' : '#ef4444';
                      const relevantSet = new Set(qr.relevant);
                      return (
                        <div key={i} style={{ fontSize: 11, padding: '7px 9px', borderRadius: 6,
                          background: 'var(--card-bg)', border: `1px solid ${borderColor}44` }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                            <span style={{ color: 'var(--fg)', fontStyle: 'italic' }}>"{qr.query}"</span>
                            <span style={{ color: borderColor, fontFamily: 'monospace', marginLeft: 8, whiteSpace: 'nowrap' }}>
                              P@1 {(qr['p@1'] * 100).toFixed(0)}%  MRR {(qr['mrr'] * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                            {qr.retrieved_top3.map((docId, rank) => {
                              const isRelevant = relevantSet.has(docId);
                              return (
                                <div key={rank} style={{ display: 'flex', gap: 6, alignItems: 'baseline' }}>
                                  <span style={{ color: 'var(--muted)', minWidth: 14 }}>#{rank + 1}</span>
                                  <span style={{ fontFamily: 'monospace', color: isRelevant ? '#22c55e' : 'var(--muted)',
                                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                    {isRelevant ? '✓ ' : '✗ '}{docId}
                                  </span>
                                </div>
                              );
                            })}
                            {qr.retrieved_top3.length === 0 && (
                              <span style={{ color: '#ef4444' }}>no results returned</span>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <p className="muted" style={{ fontSize: 12 }}>Click "Run eval" to measure retrieval quality against the gold QA set.</p>
          )}
        </div>

        <div style={{ borderTop: '1px solid var(--card-border)' }} />

        {/* Quality trend */}
        <div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
            <h2 style={{ margin: 0, fontSize: 14 }}><LineChart size={13} /> Quality Trend</h2>
            <button className="kb-btn" onClick={recordSnapshot} disabled={recordingSnapshot}>
              <RefreshCw size={11} className={recordingSnapshot ? 'spin' : ''} />
              {recordingSnapshot ? 'Recording…' : 'Record now'}
            </button>
          </div>
          <QualityTrendChart history={qualityHistory} />
        </div>

        <div style={{ borderTop: '1px solid var(--card-border)' }} />

        {/* Classifier models */}
        <div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
            <h2 style={{ margin: 0, fontSize: 14 }}><Activity size={13} /> Classifier Models</h2>
            <button className="kb-btn" onClick={loadClfHistory}>
              <RefreshCw size={11} /> Refresh
            </button>
          </div>
          <ClassifierModelsPanel clfHistory={clfHistory} />
        </div>

      </div>
    </div>
  );
}

// ── System panel ───────────────────────────────────────────────────────────

const RISK_OPTIONS = ['INFO', 'WATCH', 'WARNING', 'CRITICAL'];

function DeliverySection() {
  const [status, setStatus]       = useState(null);
  const [newEmail, setNewEmail]   = useState('');
  const [testResult, setTestResult] = useState(null);
  const [loading, setLoading]     = useState(false);

  function load() {
    fetch('/api/v1/delivery/status').then(r => r.ok ? r.json() : null).then(d => { if (d) setStatus(d); });
  }
  useEffect(load, []);

  async function addRecipient() {
    if (!newEmail.trim()) return;
    const next = [...(status?.email_recipients ?? []), newEmail.trim()];
    await fetch('/api/v1/delivery/recipients', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ recipients: next }),
    });
    setNewEmail('');
    load();
  }

  async function removeRecipient(addr) {
    const next = (status?.email_recipients ?? []).filter(r => r !== addr);
    await fetch('/api/v1/delivery/recipients', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ recipients: next }),
    });
    load();
  }

  async function setMinRisk(risk) {
    await fetch('/api/v1/delivery/min-risk', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ risk }),
    });
    load();
  }

  async function sendTest() {
    setLoading(true); setTestResult(null);
    const r = await fetch('/api/v1/delivery/test', { method: 'POST' });
    const d = await r.json();
    setTestResult(d.result);
    setLoading(false);
  }

  const emailOk = status?.email;

  return (
    <div className="card">
      <div className="input-header"><h2><Bell size={14} /> Push Delivery</h2></div>

      {/* Channel status row */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
        <span style={{
          padding: '3px 10px', borderRadius: 12, fontSize: 11, fontWeight: 700,
          background: emailOk ? '#14532d' : '#1e293b',
          color: emailOk ? '#4ade80' : '#64748b',
          border: `1px solid ${emailOk ? '#16a34a' : '#334155'}`,
        }}>
          <Mail size={10} style={{ marginRight: 4, verticalAlign: 'middle' }} />
          Email {emailOk ? 'ready' : 'not configured'}
        </span>
        <span style={{
          padding: '3px 10px', borderRadius: 12, fontSize: 11, fontWeight: 700,
          background: status?.telegram ? '#1e1b4b' : '#1e293b',
          color: status?.telegram ? '#a5b4fc' : '#64748b',
          border: `1px solid ${status?.telegram ? '#4f46e5' : '#334155'}`,
        }}>Telegram {status?.telegram ? 'ready' : 'not configured'}</span>
        <span style={{
          padding: '3px 10px', borderRadius: 12, fontSize: 11, fontWeight: 700,
          background: status?.slack ? '#1c1917' : '#1e293b',
          color: status?.slack ? '#fb923c' : '#64748b',
          border: `1px solid ${status?.slack ? '#ea580c' : '#334155'}`,
        }}>Slack {status?.slack ? 'ready' : 'not configured'}</span>
      </div>

      {!emailOk && (
        <p className="muted" style={{ fontSize: 11, marginBottom: 12 }}>
          Set <code>F1DI_SMTP_USERNAME</code> and <code>F1DI_SMTP_PASSWORD</code> (Gmail app password) in your .env to enable email delivery.
        </p>
      )}

      {/* Recipients */}
      <h3 style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 8, marginTop: 0 }}>
        <Mail size={11} style={{ marginRight: 4, verticalAlign: 'middle' }} />
        Email recipients
      </h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginBottom: 12 }}>
        {(status?.email_recipients ?? []).map(addr => (
          <div key={addr} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 12, flex: 1, color: '#e2e8f0' }}>{addr}</span>
            <button onClick={() => removeRecipient(addr)}
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#ef4444', padding: 2 }}>
              <X size={12} />
            </button>
          </div>
        ))}
        {!(status?.email_recipients?.length) && (
          <p className="muted" style={{ fontSize: 11 }}>No recipients configured.</p>
        )}
      </div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <input className="chat-input" style={{ fontSize: 12 }}
          placeholder="add email address…"
          value={newEmail}
          onChange={e => setNewEmail(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && addRecipient()}
        />
        <button className="send-btn" onClick={addRecipient} title="Add recipient">
          <Plus size={13} />
        </button>
      </div>

      {/* Min risk threshold */}
      <h3 style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 8, marginTop: 0 }}>
        <ShieldAlert size={11} style={{ marginRight: 4, verticalAlign: 'middle' }} />
        Alert threshold
      </h3>
      <div style={{ display: 'flex', gap: 6, marginBottom: 16, flexWrap: 'wrap' }}>
        {RISK_OPTIONS.map(r => (
          <button key={r}
            onClick={() => setMinRisk(r)}
            style={{
              padding: '4px 12px', borderRadius: 8, fontSize: 11, fontWeight: 700, cursor: 'pointer',
              border: '1px solid',
              background: status?.notify_min_risk === r ? RISK_META[r]?.bg ?? '#1e293b' : '#1e293b',
              color: status?.notify_min_risk === r ? '#e2e8f0' : '#64748b',
              borderColor: status?.notify_min_risk === r ? (RISK_META[r]?.color ?? '#64748b') : '#334155',
            }}>
            {r}
          </button>
        ))}
      </div>
      <p className="muted" style={{ fontSize: 11, marginBottom: 16 }}>
        Alerts fire for this risk level and above. Applies to all channels.
      </p>

      {/* Test button */}
      <button className="send-btn" style={{ width: '100%', justifyContent: 'center', gap: 6 }}
        onClick={sendTest} disabled={loading}>
        {loading ? <Activity size={13} className="spin" /> : <Bell size={13} />}
        {loading ? 'Sending…' : 'Send test notification'}
      </button>
      {testResult && (
        <div style={{ marginTop: 10, fontSize: 11 }}>
          {Object.entries(testResult).map(([ch, ok]) => (
            <p key={ch} style={{ margin: '2px 0', color: ok ? '#4ade80' : '#f87171' }}>
              {ch}: {ok ? 'delivered' : 'failed'}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

function SchedulerSection() {
  const [status, setStatus]   = useState(null);
  const [source, setSource]   = useState('fastf1');
  const [years, setYears]     = useState('');
  const [n, setN]             = useState(5);
  const [triggering, setTriggering] = useState(false);
  const [triggerMsg, setTriggerMsg] = useState(null);

  function load() {
    fetch('/api/v1/ingestion/status').then(r => r.ok ? r.json() : null).then(d => { if (d) setStatus(d); });
  }
  useEffect(load, []);

  async function trigger() {
    setTriggering(true); setTriggerMsg(null);
    const qs = `source=${source}&n=${n}${years ? `&years=${years}` : ''}`;
    const r = await fetch(`/api/v1/ingestion/trigger?${qs}`, { method: 'POST' });
    const d = await r.json();
    setTriggerMsg(d.status === 'ingestion_triggered'
      ? `Triggered ${source} ingestion (${d.years ? `years: ${d.years.join(',')}` : 'auto'}, n=${n})`
      : d.error ?? 'Unknown error');
    setTriggering(false);
    setTimeout(load, 3000);
  }

  const runs = status?.latest ?? [];

  return (
    <div className="card">
      <div className="input-header"><h2><Clock size={14} /> Data Ingestion</h2></div>

      {/* Auto-schedule status */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <span style={{
          padding: '3px 10px', borderRadius: 12, fontSize: 11, fontWeight: 700,
          background: '#1e293b', color: '#64748b', border: '1px solid #334155',
        }}>
          Auto-schedule: enable via <code style={{ fontSize: 10 }}>F1DI_INGESTION_AUTO_ENABLED=true</code>
        </span>
        {status?.total_runs != null && (
          <span style={{
            padding: '3px 10px', borderRadius: 12, fontSize: 11, fontWeight: 700,
            background: '#14532d', color: '#4ade80', border: '1px solid #16a34a',
          }}>
            {status.total_runs} runs total
          </span>
        )}
      </div>

      {/* Manual trigger */}
      <h3 style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 8, marginTop: 0 }}>
        <Play size={11} style={{ marginRight: 4, verticalAlign: 'middle' }} />
        Manual trigger
      </h3>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <select value={source} onChange={e => setSource(e.target.value)}
          style={{ background: '#1e293b', color: '#e2e8f0', border: '1px solid #334155', borderRadius: 6, padding: '4px 8px', fontSize: 12 }}>
          <option value="fastf1">FastF1</option>
          <option value="openf1">OpenF1</option>
          <option value="jolpica">Jolpica</option>
        </select>
        <input className="chat-input" style={{ fontSize: 12, width: 120 }}
          placeholder="years e.g. 2024"
          value={years}
          onChange={e => setYears(e.target.value)}
        />
        <input type="number" min={1} max={20} value={n} onChange={e => setN(Number(e.target.value))}
          style={{ background: '#1e293b', color: '#e2e8f0', border: '1px solid #334155', borderRadius: 6, padding: '4px 8px', fontSize: 12, width: 60 }}
          title="Sessions per year"
        />
        <button className="send-btn" onClick={trigger} disabled={triggering}>
          {triggering ? <Activity size={13} className="spin" /> : <Play size={13} />}
          {triggering ? 'Triggering…' : 'Run'}
        </button>
        <button className="kb-btn" onClick={load}><RefreshCw size={11} /></button>
      </div>
      {triggerMsg && <p style={{ fontSize: 11, color: '#4ade80', marginBottom: 12 }}>{triggerMsg}</p>}

      {/* Recent runs */}
      <h3 style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 8, marginTop: 0 }}>Recent runs</h3>
      {runs.length === 0
        ? <p className="muted" style={{ fontSize: 11 }}>No ingestion runs recorded yet.</p>
        : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
              <thead>
                <tr>{['Source', 'Year', 'Round', 'Track', 'Docs added', 'Completed'].map(h => (
                  <th key={h} style={{ textAlign: 'left', padding: '4px 8px', color: 'var(--muted)', fontWeight: 600, borderBottom: '1px solid var(--card-border)' }}>{h}</th>
                ))}</tr>
              </thead>
              <tbody>
                {runs.map((r, i) => (
                  <tr key={i} style={{ borderTop: '1px solid var(--card-border)' }}>
                    <td style={{ padding: '4px 8px' }}>{r.source}</td>
                    <td style={{ padding: '4px 8px' }}>{r.year ?? '—'}</td>
                    <td style={{ padding: '4px 8px' }}>{r.round_num ?? '—'}</td>
                    <td style={{ padding: '4px 8px' }}>{r.track_id ?? '—'}</td>
                    <td style={{ padding: '4px 8px', fontFamily: 'monospace' }}>{r.documents_added}</td>
                    <td style={{ padding: '4px 8px', color: 'var(--muted)' }}>{r.completed_at?.replace('T', ' ').slice(0, 16)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      }
    </div>
  );
}

function FlywheelStatusCard() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  function load() {
    setLoading(true);
    fetch('/api/v1/flywheel/status')
      .then(r => r.ok ? r.json() : null)
      .then(d => { setStatus(d); setLoading(false); })
      .catch(() => setLoading(false));
  }
  useEffect(load, []);

  function Chip({ ok, label, detail }) {
    const on  = ok === true;
    const off = ok === false;
    return (
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '6px 10px', borderRadius: 8, marginBottom: 6,
        background: on ? '#052e16' : off ? '#1f0a0a' : '#1e293b',
        border: `1px solid ${on ? '#166534' : off ? '#7f1d1d' : '#334155'}`,
      }}>
        <span style={{ fontSize: 12, color: on ? '#4ade80' : off ? '#f87171' : '#94a3b8', fontWeight: 600 }}>
          {on ? '✓' : off ? '✗' : '–'} {label}
        </span>
        {detail != null && (
          <span style={{ fontSize: 11, color: '#64748b', fontFamily: 'monospace' }}>{detail}</span>
        )}
      </div>
    );
  }

  const overall = status?.overall_ok;

  return (
    <div className="card">
      <div className="input-header" style={{ justifyContent: 'space-between' }}>
        <h2><Workflow size={14} /> Flywheel Status</h2>
        <button className="kb-btn" onClick={load} disabled={loading} title="Refresh">
          <RefreshCw size={11} className={loading ? 'spin' : ''} />
        </button>
      </div>

      {/* Overall badge */}
      <div style={{ marginBottom: 14 }}>
        <span style={{
          padding: '3px 12px', borderRadius: 12, fontSize: 11, fontWeight: 700,
          background: overall === true ? '#14532d' : overall === false ? '#450a0a' : '#1e293b',
          color:      overall === true ? '#4ade80' : overall === false ? '#f87171' : '#64748b',
          border:     `1px solid ${overall === true ? '#16a34a' : overall === false ? '#991b1b' : '#334155'}`,
        }}>
          {loading ? 'Checking…' : overall ? 'Pipeline ready' : 'Action required'}
        </span>
      </div>

      {status && (
        <>
          <Chip ok={status.ingestion_enabled}  label="Auto-ingestion enabled"
            detail={status.ingestion_enabled ? null : 'F1DI_INGESTION_AUTO_ENABLED=true'} />
          <Chip ok={status.db_ok}              label="Database reachable" />
          <Chip ok={status.calibrator_exists}  label="Calibrator artifact" />
          <Chip
            ok={status.ece_ok}
            label="Calibration ECE ≤ 0.15"
            detail={status.calibrator_ece != null ? status.calibrator_ece.toFixed(4) : 'n/a'}
          />
          <Chip
            ok={status.outcome_cache_exists}
            label="Outcome labels cache"
            detail={status.rounds_labeled > 0 ? `${status.rounds_labeled} rounds` : 'none yet'}
          />

          {/* Agent classifiers */}
          <div style={{ marginTop: 12, paddingTop: 10, borderTop: '1px solid var(--card-border)' }}>
            <h3 style={{ fontSize: 11, color: 'var(--muted)', margin: '0 0 8px' }}>Agent classifiers</h3>
            {['tire', 'battery', 'weather', 'telemetry', 'safety_car', 'fuel'].map(agent => {
              const c = status.classifiers?.[agent];
              const rt = status.auto_retrain?.agents?.[agent];
              const exists = c?.exists;
              const acc = c?.accuracy != null ? c.accuracy.toFixed(3) : null;
              const brier = c?.brier_score != null ? c.brier_score.toFixed(3) : null;
              const real = c?.n_real ?? 0;
              const brierOk = c?.brier_score != null && c.brier_score < 0.20;
              const ver = c?.model_version ?? null;
              const retraining = rt?.retrain_in_progress ?? false;
              const delta = rt != null ? (real - (rt.pkl_n_real ?? real)) : 0;
              const threshold = status.auto_retrain?.threshold ?? 5;
              return (
                <div key={agent} style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  padding: '4px 8px', borderRadius: 6, marginBottom: 4,
                  background: retraining ? '#0f1e10' : exists ? '#0a1628' : '#1e293b',
                  border: `1px solid ${retraining ? '#166534' : exists ? (brierOk ? '#1e3a5f' : '#7c3a1e') : '#334155'}`,
                }}>
                  <span style={{ fontSize: 11, color: retraining ? '#4ade80' : exists ? '#93c5fd' : '#475569', fontWeight: 600, textTransform: 'capitalize', display: 'flex', alignItems: 'center', gap: 5 }}>
                    {retraining ? '⟳' : exists ? '●' : '○'} {agent}
                    {ver && <span style={{ fontSize: 8, padding: '1px 4px', borderRadius: 3, background: '#0f2418', color: '#4ade80', border: '1px solid #166534', fontFamily: 'monospace' }}>{ver}</span>}
                  </span>
                  <span style={{ fontSize: 10, color: '#64748b', fontFamily: 'monospace' }}>
                    {retraining
                      ? 'retraining…'
                      : exists
                        ? `acc ${acc}  brier ${brier}  real ${real}${delta > 0 ? `  +${delta} new` : ''}`
                        : 'run make fit-' + agent}
                    {!retraining && delta > 0 && delta < threshold && (
                      <span style={{ color: '#f59e0b', marginLeft: 4 }}>({threshold - delta} until auto-retrain)</span>
                    )}
                  </span>
                </div>
              );
            })}
            {/* Meta-learner row */}
            {(() => {
              const m = status.classifiers?.meta;
              const exists = m?.exists;
              const active = m?.active_in_inference;
              const real = m?.n_real ?? 0;
              const brier = m?.brier_score != null ? m.brier_score.toFixed(3) : null;
              return (
                <div style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  padding: '4px 8px', borderRadius: 6, marginBottom: 4,
                  background: active ? '#0f2418' : exists ? '#0a1628' : '#1e293b',
                  border: `1px solid ${active ? '#166534' : exists ? '#1e3a5f' : '#334155'}`,
                }}>
                  <span style={{ fontSize: 11, color: active ? '#4ade80' : exists ? '#93c5fd' : '#475569', fontWeight: 600 }}>
                    {active ? '●' : exists ? '○' : '○'} meta-learner
                  </span>
                  <span style={{ fontSize: 10, color: '#64748b', fontFamily: 'monospace' }}>
                    {exists
                      ? (active
                          ? `active · real ${real}${brier ? '  brier ' + brier : ''}`
                          : `inactive · need ${Math.max(0, 20 - real)} more labels${brier ? '  brier ' + brier : ''}`)
                      : 'run make fit-meta'}
                  </span>
                </div>
              );
            })()}
          </div>

          {/* Active settings */}
          <div style={{ marginTop: 10, paddingTop: 10, borderTop: '1px solid var(--card-border)' }}>
            <h3 style={{ fontSize: 11, color: 'var(--muted)', margin: '0 0 8px' }}>Active settings</h3>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <span style={{
                padding: '2px 9px', borderRadius: 10, fontSize: 10, fontWeight: 700,
                background: status.shadow_challenger_enabled ? '#1e1b4b' : '#1e293b',
                color: status.shadow_challenger_enabled ? '#a5b4fc' : '#64748b',
                border: `1px solid ${status.shadow_challenger_enabled ? '#4f46e5' : '#334155'}`,
              }}>
                Shadow challenger {status.shadow_challenger_enabled ? 'ON' : 'OFF'}
              </span>
              <span style={{
                padding: '2px 9px', borderRadius: 10, fontSize: 10, fontWeight: 700,
                background: '#1e293b', color: '#94a3b8', border: '1px solid #334155',
              }}>
                Cooldown {status.alert_cooldown_laps} laps
              </span>
            </div>
          </div>

          {!status.overall_ok && (
            <p className="muted" style={{ fontSize: 11, marginTop: 12, lineHeight: 1.5 }}>
              Run <code>make smoketest</code> for a full pre-race check including FastF1 dry-run.
            </p>
          )}
        </>
      )}
    </div>
  );
}

function SystemPanel() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="grid" style={{ gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <DeliverySection />
        <SchedulerSection />
      </div>
      <FlywheelStatusCard />
    </div>
  );
}

// ── App ────────────────────────────────────────────────────────────────────

export default function App() {
  const [mode, setMode]         = useState('telemetry');
  const [calibEce, setCalibEce] = useState(null);
  const [version, setVersion]   = useState(null);

  useEffect(() => {
    fetch('/api/ready').then(r => r.ok ? r.json() : null)
      .then(d => { const e = d?.checks?.calibration_quality?.ece; if (e != null) setCalibEce(e); })
      .catch(() => {});
    fetch('/api/version').then(r => r.ok ? r.json() : null)
      .then(d => { if (d) setVersion(d); })
      .catch(() => {});
  }, []);

  return (
    <main className="shell">
      <header className="hero">
        <div className="hero-left">
          <h1>F1 Driver Intelligence</h1>
          <p>Telemetry analysis · RAG evidence · Calibrated confidence · LLM-backed advice</p>
        </div>
        <div className="hero-center">
          <div className="mode-tabs">
            <button className={`mode-tab${mode === 'telemetry' ? ' active' : ''}`}
              onClick={() => setMode('telemetry')}>
              <BarChart2 size={14} /> Telemetry
            </button>
            <button className={`mode-tab${mode === 'live' ? ' active' : ''}`}
              onClick={() => setMode('live')}>
              <History size={14} /> Session
            </button>
            <button className={`mode-tab${mode === 'chat' ? ' active' : ''}`}
              onClick={() => setMode('chat')}>
              <MessageSquare size={14} /> Chat Analysis
            </button>
            <button className={`mode-tab${mode === 'history' ? ' active' : ''}`}
              onClick={() => setMode('history')}>
              <History size={14} /> History
            </button>
            <button className={`mode-tab${mode === 'analytics' ? ' active' : ''}`}
              onClick={() => setMode('analytics')}>
              <Database size={14} /> Analytics
            </button>
            <button className={`mode-tab${mode === 'predictions' ? ' active' : ''}`}
              onClick={() => setMode('predictions')}>
              <LineChart size={14} /> Predictions
            </button>
            <button className={`mode-tab${mode === 'regression' ? ' active' : ''}`}
              onClick={() => setMode('regression')}>
              <FlaskConical size={14} /> Regression
            </button>
            <button className={`mode-tab${mode === 'modellab' ? ' active' : ''}`}
              onClick={() => setMode('modellab')}>
              <Radio size={14} /> Model Lab
            </button>
            <button className={`mode-tab${mode === 'system' ? ' active' : ''}`}
              onClick={() => setMode('system')}>
              <Settings size={14} /> System
            </button>
          </div>
        </div>
        <div className="hero-right">
          <BackendBadges version={version} />
          {calibEce != null && (
            <div className={`ece-badge ${calibEce <= 0.15 ? 'ece-pass' : 'ece-fail'}`}>
              <span className="ece-label">Calibration ECE</span>
              <span className="ece-value">{calibEce.toFixed(4)}</span>
            </div>
          )}
          <Gauge size={34} strokeWidth={1.5} />
        </div>
      </header>

      {mode === 'telemetry'  && <TelemetryPanel version={version} />}
      {mode === 'live'       && <LivePanel version={version} />}
      {mode === 'chat'       && <ChatPanel version={version} />}
      {mode === 'history'    && <HistoryPanel />}
      {mode === 'analytics'  && <AnalyticsPanel />}
      {mode === 'predictions' && <PredictionsPanel version={version} />}
      {mode === 'regression' && <RegressionPanel />}
      {mode === 'modellab'   && <ModelLabPanel version={version} />}
      {mode === 'system'     && <SystemPanel />}
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
