import { useState, useEffect, useRef, useCallback } from 'react';

const SIGNAL_COLORS = {
  speech_start: '#44cc88',
  speech_end: '#44cc88',
  barge_in: '#ff4466',
  'audio.input_level': '#555570',
  transcript: '#66aaff',
  detected_language: '#66aaff',
  language_switch: '#66aaff',
  request_start: '#aa66dd',
  first_token: '#aa66dd',
  complete: '#aa66dd',
  tokens_per_sec: '#aa66dd',
  synth_start: '#cc8844',
  synth_end: '#cc8844',
  state_change: '#ccaa44',
  mood_change: '#ccaa44',
  filler_played: '#ccaa44',
  end_to_end_latency: '#8888a0',
  vram_usage: '#8888a0',
  'system.vram_optimized': '#8888a0',
  'system.device_change': '#8888a0',
  voice_clone_progress: '#cc6666',
  process_state_change: '#ccaa44',
  error: '#ff4466',
};

const CATEGORIES = {
  all: 'All',
  audio: 'Audio/VAD',
  asr: 'ASR',
  llm: 'LLM',
  tts: 'TTS',
  state: 'State',
  system: 'System',
};

const CATEGORY_TYPES = {
  audio: ['speech_start', 'speech_end', 'barge_in', 'audio.input_level'],
  asr: ['transcript', 'detected_language', 'language_switch'],
  llm: ['request_start', 'first_token', 'complete', 'tokens_per_sec'],
  tts: ['synth_start', 'synth_end'],
  state: ['state_change', 'mood_change', 'filler_played'],
  system: ['end_to_end_latency', 'vram_usage', 'system.vram_optimized', 'system.device_change', 'voice_clone_progress', 'process_state_change', 'error'],
};

export default function EventLogPage({ events }) {
  const [filter, setFilter] = useState('all');
  const [paused, setPaused] = useState(false);
  const [logEntries, setLogEntries] = useState([]);
  const bottomRef = useRef(null);
  const containerRef = useRef(null);

  // Accumulate events
  useEffect(() => {
    if (paused || !events || events.length === 0) return;
    const last = events[events.length - 1];
    if (!last) return;

    const entry = {
      time: Date.now(),
      type: last.type || last.event || 'unknown',
      event: last.event,
      data: last,
    };

    setLogEntries(prev => [...prev.slice(-500), entry]);
  }, [events, paused]);

  // Auto-scroll
  useEffect(() => {
    if (!paused && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logEntries, paused]);

  const filtered = filter === 'all'
    ? logEntries
    : logEntries.filter(e => CATEGORY_TYPES[filter]?.includes(e.type));

  const handleClear = useCallback(() => setLogEntries([]), []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', height: '100%' }}>
      {/* Toolbar */}
      <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center', flexWrap: 'wrap' }}>
        {Object.entries(CATEGORIES).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setFilter(key)}
            style={{
              background: filter === key ? 'var(--bg-surface)' : 'transparent',
              border: filter === key ? '1px solid var(--text-muted)' : '1px solid transparent',
              borderRadius: 'var(--radius-sm)',
              color: filter === key ? 'var(--text-primary)' : 'var(--text-muted)',
              fontSize: '0.65rem',
              padding: '0.25rem 0.5rem',
              cursor: 'pointer',
            }}
          >
            {label}
          </button>
        ))}

        <div style={{ marginLeft: 'auto', display: 'flex', gap: '0.3rem' }}>
          <button
            onClick={() => setPaused(!paused)}
            style={{
              background: paused ? '#ccaa44' : 'var(--bg-surface)',
              border: 'none',
              borderRadius: 'var(--radius-sm)',
              color: paused ? '#0a0a0f' : 'var(--text-secondary)',
              fontSize: '0.65rem',
              padding: '0.25rem 0.5rem',
              cursor: 'pointer',
              fontWeight: 600,
            }}
          >
            {paused ? '\u25B6 Resume' : '\u23F8 Pause'}
          </button>
          <button
            onClick={handleClear}
            style={{
              background: 'transparent',
              border: '1px solid var(--text-muted)',
              borderRadius: 'var(--radius-sm)',
              color: 'var(--text-muted)',
              fontSize: '0.65rem',
              padding: '0.25rem 0.5rem',
              cursor: 'pointer',
            }}
          >
            Clear
          </button>
          <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', alignSelf: 'center', marginLeft: 4 }}>
            {filtered.length} events
          </span>
        </div>
      </div>

      {/* Event stream */}
      <div
        ref={containerRef}
        style={{
          flex: 1,
          overflow: 'auto',
          background: 'var(--bg-surface)',
          borderRadius: 'var(--radius-sm)',
          fontFamily: 'monospace',
          fontSize: '0.7rem',
          lineHeight: 1.6,
        }}
      >
        {filtered.length === 0 && (
          <div style={{ padding: '1rem', color: 'var(--text-muted)', textAlign: 'center' }}>
            {paused ? 'Paused \u2014 events are being buffered' : 'Waiting for events...'}
          </div>
        )}
        {filtered.map((entry, i) => (
          <div
            key={i}
            style={{
              display: 'flex',
              gap: '0.5rem',
              padding: '0.15rem 0.5rem',
              borderBottom: '1px solid var(--bg-secondary)',
              alignItems: 'baseline',
            }}
          >
            <span style={{ color: 'var(--text-muted)', minWidth: 70, flexShrink: 0, fontSize: '0.6rem' }}>
              {new Date(entry.time).toLocaleTimeString(undefined, { hour12: false, fractionalSecondDigits: 1 })}
            </span>
            <span style={{
              color: SIGNAL_COLORS[entry.type] || 'var(--text-secondary)',
              minWidth: 120,
              flexShrink: 0,
              fontWeight: 600,
            }}>
              {entry.type}
            </span>
            <span style={{ color: 'var(--text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {formatPayload(entry.data)}
            </span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

function formatPayload(data) {
  if (!data) return '';
  const copy = { ...data };
  delete copy.event;
  delete copy.type;
  delete copy.timestamp;
  const keys = Object.keys(copy);
  if (keys.length === 0) return '';
  if (keys.length === 1 && typeof copy[keys[0]] !== 'object') return String(copy[keys[0]]);
  try {
    return JSON.stringify(copy);
  } catch {
    return '';
  }
}
