import { useState, useEffect, useRef } from 'react';
import { useSignals } from '../hooks/useSignals';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell } from 'recharts';

function MetricCard({ label, value, unit, color }) {
  return (
    <div style={{
      background: 'var(--bg-surface)',
      borderRadius: 'var(--radius-sm)',
      padding: '0.6rem 0.75rem',
      flex: '1 1 120px',
      minWidth: 120,
    }}>
      <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        {label}
      </div>
      <div style={{ fontSize: '1.25rem', fontWeight: 700, color: color || 'var(--text-primary)', marginTop: 2 }}>
        {value}
        {unit && <span style={{ fontSize: '0.7rem', fontWeight: 400, color: 'var(--text-muted)', marginLeft: 3 }}>{unit}</span>}
      </div>
    </div>
  );
}

const STATE_COLORS = {
  IDLE: '#4466cc',
  LISTENING: '#44cc88',
  PROCESSING: '#aa66dd',
  SPEAKING: '#66aaff',
  INTERRUPTED: '#ff4466',
  DISABLED: '#555570',
};

export default function Dashboard({ signals: parentSignals, state, events }) {
  const localSignals = useSignals([
    'end_to_end_latency',
    'vram_usage',
    'tokens_per_sec',
    'audio.input_level',
    'mood_change',
  ]);
  const sig = { ...localSignals, ...parentSignals };

  const [latency, setLatency] = useState({ vad: 0, asr: 0, llm_first: 0, tts: 0, total: 0 });
  const [vram, setVram] = useState({ used: 0, total: 0 });
  const [tps, setTps] = useState(0);
  const [mood, setMood] = useState('neutral');
  const [lastTranscript, setLastTranscript] = useState('');
  const [lastResponse, setLastResponse] = useState('');
  const [inputLevel, setInputLevel] = useState(0);

  useEffect(() => {
    if (sig.end_to_end_latency) {
      const d = sig.end_to_end_latency;
      setLatency({
        vad: d.vad_ms || 0,
        asr: d.asr_ms || 0,
        llm_first: d.llm_first_token_ms || 0,
        tts: d.tts_ms || 0,
        total: d.total_ms || 0,
      });
    }
    if (sig.vram_usage) {
      setVram({ used: sig.vram_usage.used_gb || 0, total: sig.vram_usage.total_gb || 0 });
    }
    if (sig.tokens_per_sec) setTps(sig.tokens_per_sec.tps || 0);
    if (sig.mood_change) setMood(sig.mood_change.mood || 'neutral');
    if (sig['audio.input_level']) setInputLevel(sig['audio.input_level'].level || 0);
  }, [sig]);

  // Track last transcript/response from events
  const eventsRef = useRef(events);
  eventsRef.current = events;
  useEffect(() => {
    if (!events || events.length === 0) return;
    const last = events[events.length - 1];
    if (last.event === 'transcript' && last.role === 'user') setLastTranscript(last.text || '');
    if (last.event === 'transcript' && last.role === 'assistant') setLastResponse(last.text || '');
  }, [events]);

  const latencyData = [
    { name: 'VAD', ms: latency.vad, fill: '#44cc88' },
    { name: 'ASR', ms: latency.asr, fill: '#66aaff' },
    { name: 'LLM', ms: latency.llm_first, fill: '#aa66dd' },
    { name: 'TTS', ms: latency.tts, fill: '#cc8844' },
  ];

  const vramPct = vram.total > 0 ? ((vram.used / vram.total) * 100).toFixed(0) : 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      {/* Top row — state + mood + metrics */}
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
        <MetricCard label="State" value={state || 'IDLE'} color={STATE_COLORS[state] || STATE_COLORS.IDLE} />
        <MetricCard label="Mood" value={mood} />
        <MetricCard label="Input Level" value={(inputLevel * 100).toFixed(0)} unit="%" />
        <MetricCard label="Tokens/s" value={tps.toFixed(1)} />
      </div>

      {/* VRAM */}
      <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 4 }}>
          <span>VRAM</span>
          <span>{vram.used.toFixed(1)} / {vram.total.toFixed(1)} GB ({vramPct}%)</span>
        </div>
        <div style={{ height: 8, background: 'var(--bg-secondary)', borderRadius: 4, overflow: 'hidden' }}>
          <div style={{
            height: '100%',
            width: `${vramPct}%`,
            background: vramPct > 90 ? '#ff4466' : vramPct > 70 ? '#ccaa44' : '#44cc88',
            borderRadius: 4,
            transition: 'width 0.5s ease',
          }} />
        </div>
      </div>

      {/* Latency breakdown */}
      <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 4 }}>
          <span>Latency Breakdown</span>
          <span>Total: {latency.total.toFixed(0)} ms</span>
        </div>
        <ResponsiveContainer width="100%" height={80}>
          <BarChart data={latencyData} layout="vertical" barCategoryGap={4}>
            <XAxis type="number" hide />
            <YAxis type="category" dataKey="name" width={30} tick={{ fontSize: 10, fill: '#8888a0' }} />
            <Bar dataKey="ms" radius={[0, 3, 3, 0]}>
              {latencyData.map((entry, i) => (
                <Cell key={i} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Last transcript + response */}
      <div style={{ display: 'flex', gap: '0.5rem' }}>
        <div style={{ flex: 1, background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.75rem' }}>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 4 }}>Last Transcript</div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-primary)', wordBreak: 'break-word' }}>
            {lastTranscript || '\u2014'}
          </div>
        </div>
        <div style={{ flex: 1, background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.75rem' }}>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 4 }}>Last Response</div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-primary)', wordBreak: 'break-word' }}>
            {lastResponse?.slice(0, 200) || '\u2014'}
          </div>
        </div>
      </div>
    </div>
  );
}
