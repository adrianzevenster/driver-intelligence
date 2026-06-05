import React, { useState, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity, ChevronDown, ChevronRight, Gauge, ShieldAlert,
  Cpu, Database, MessageSquare, BarChart2, Send, User, Bot,
  RefreshCw, BookOpen, Radio, History,
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
  telemetry: 'Telemetry', tire_strategy: 'Tire Strategy',
  weather: 'Weather', battery: 'Battery / ERS',
};
const COMPOUNDS     = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'];
const COMPOUND_COLOR = { SOFT:'#ef4444', MEDIUM:'#eab308', HARD:'#cbd5e1', INTERMEDIATE:'#22c55e', WET:'#3b82f6' };
const TRACKS        = [
  'bahrain', 'jeddah', 'melbourne', 'shanghai', 'miami',
  'imola', 'monaco', 'barcelona', 'montreal', 'spielberg',
  'silverstone', 'budapest', 'spa', 'zandvoort', 'monza',
  'baku', 'singapore', 'austin', 'mexico_city', 'interlagos',
  'las_vegas', 'lusail', 'abu_dhabi', 'suzuka',
];
const SKIP_KEYS     = new Set(['lap', 'sector', 'lockup_count']);
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
            <input className="text-input" value={stats.driverId}
              onChange={e => set('driverId', e.target.value.toUpperCase().slice(0, 4))} />
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
  return [...findings].sort((a, b) => rm(b.risk).order - rm(a.risk).order)[0];
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

function AgentCard({ finding, isTop, expanded, onToggle }) {
  const m = rm(finding.risk);
  const confPct = (finding.confidence * 100).toFixed(0);
  const entries = Object.entries(finding.features ?? {}).filter(([k, v]) => !SKIP_KEYS.has(k) && typeof v === 'number');
  const preview = entries.slice(0, 3);
  return (
    <div className={`agent-card${isTop ? ' agent-top' : ''}`} style={{ borderColor: m.color+(isTop ? 'aa' : '44') }}>
      <button className="agent-header" onClick={onToggle}>
        <div className="agent-name-row">
          {isTop && <span className="top-dot" style={{ background: m.color }} />}
          <span className="agent-name">{AGENT_LABELS[finding.agent] ?? finding.agent}</span>
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
        {insight.findings.map(f => (
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
      <p className="footer-line">{insight.latency_ms.toFixed(1)} ms · session {insight.session_id}</p>
    </>
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
  const [status, setStatus]     = useState(null);
  const [loading, setLoading]   = useState({});   // { openf1: bool, fastf1: bool, jolpica: bool }
  const [result, setResult]     = useState('');
  const [years, setYears]       = useState(new Date().getFullYear().toString());
  const [n, setN]               = useState(8);

  function refreshStatus() {
    fetch('/api/v1/knowledge/status').then(r => r.ok ? r.json() : null).then(d => { if (d) setStatus(d); }).catch(() => {});
  }

  useEffect(() => { refreshStatus(); }, []);

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

function ChatPanel({ version }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput]       = useState('');
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState('');
  const bottomRef = useRef(null);
  const llm = LLM_META[version?.model_backend] ?? LLM_META.rules;

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

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
        {messages.map((m, i) => <ChatBubble key={i} msg={m} />)}
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

function LapBadge({ lap }) {
  if (!lap) return null;
  const colours = { SOFT: '#e53e3e', MEDIUM: '#d69e2e', HARD: '#e2e8f0', INTERMEDIATE: '#38a169', WET: '#3182ce' };
  const c = lap.compound?.toUpperCase() ?? 'MEDIUM';
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: '0.7rem', color: 'var(--muted)' }}>
      <span style={{ width: 8, height: 8, borderRadius: '50%', background: colours[c] ?? '#888', flexShrink: 0 }} />
      {fmtLapTime(lap.lap_time_s)}
      {lap.tyre_life != null && <span style={{ opacity: 0.6 }}>· {c[0]} ×{lap.tyre_life}</span>}
    </span>
  );
}

function LivePanel({ version }) {
  const [races, setRaces]           = useState([]);
  const [drivers, setDrivers]       = useState([]);
  const [laps, setLaps]             = useState([]);
  const [roundNum, setRoundNum]     = useState('');
  const [driver, setDriver]         = useState('');
  const [selectedLap, setSelectedLap] = useState(null);
  const [replayMode, setReplayMode] = useState(false);
  const [audience, setAudience]     = useState('DRIVER');
  const [insight, setInsight]       = useState(null);
  const [error, setError]           = useState('');
  const [loading, setLoading]       = useState(false);
  const [loadingDrivers, setLoadingDrivers] = useState(false);
  const [year, setYear]             = useState(2024);

  useEffect(() => {
    setError(''); setRaces([]); setRoundNum(''); setDrivers([]); setDriver(''); setLaps([]);
    fetch(`/api/v1/session/races?year=${year}`)
      .then(r => r.ok ? r.json() : [])
      .then(setRaces)
      .catch(() => {});
  }, [year]);

  useEffect(() => {
    if (!roundNum) return;
    setDrivers([]); setDriver(''); setLaps([]); setSelectedLap(null);
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
    setLaps([]); setSelectedLap(null);
    fetch(`/api/v1/session/laps/${year}/${roundNum}/${driver}`)
      .then(r => r.ok ? r.json() : [])
      .then(rows => {
        setLaps(rows);
        if (rows.length) setSelectedLap(rows[rows.length - 1].lap_number);
      })
      .catch(() => {});
  }, [year, roundNum, driver]);

  async function fetchInsight(lapOverride) {
    if (!roundNum || !driver) return;
    setError(''); setLoading(true);
    const lap = lapOverride ?? (replayMode ? selectedLap : null);
    const lapParam = lap != null ? `&lap_number=${lap}` : '';
    try {
      const res = await fetch(
        `/api/v1/session/insight?year=${year}&round_num=${roundNum}&driver=${driver}&audience=${audience}${lapParam}`,
        { method: 'POST' }
      );
      if (!res.ok) throw new Error(await res.text());
      setInsight(await res.json());
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
            <label style={{ display: 'block' }}><span>Driver</span>
              <select className="text-input" value={driver}
                onChange={e => setDriver(e.target.value)}
                disabled={!drivers.length}>
                {loadingDrivers
                  ? <option>Loading…</option>
                  : !drivers.length
                    ? <option>— pick race first —</option>
                    : drivers.map(d => (
                        <option key={d.code} value={d.code}>{d.code}</option>
                      ))}
              </select>
            </label>
          </Section>

          {laps.length > 0 && (
            <Section title="Lap">
              <div className="context-row" style={{ alignItems: 'center', gap: 8 }}>
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
              : 'Analyse Session'}
        </button>

        <p className="muted" style={{ fontSize: '0.70rem', marginTop: 8 }}>
          Uses FastF1 — works any time, including during live race weekends.
          Analyses a 5-lap window for trend detection. First load per session
          downloads ~50 MB and may take 20–30 s.
        </p>
        {error && <pre className="error">{error}</pre>}
      </div>

      <div className="card insight-card">
        <h2><ShieldAlert size={15} /> {replayMode ? `Lap ${selectedLap} Insight` : 'Session Insight'}</h2>
        {!insight
          ? <p className="muted empty-hint">Pick a race and driver, then click Analyse Session.</p>
          : <InsightPanel insight={insight} modelBackend={version?.model_backend} />}
      </div>
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

      {mode === 'telemetry' && <TelemetryPanel version={version} />}
      {mode === 'live'      && <LivePanel version={version} />}
      {mode === 'chat'      && <ChatPanel version={version} />}
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
