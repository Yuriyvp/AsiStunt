import { useState, useEffect } from 'react';
import { useSignals } from '../hooks/useSignals';

const STATES = ['IDLE', 'LISTENING', 'PROCESSING', 'SPEAKING', 'INTERRUPTED'];

const STATE_META = {
  IDLE: { color: '#4466cc', label: 'Idle' },
  LISTENING: { color: '#44cc88', label: 'Listening' },
  PROCESSING: { color: '#aa66dd', label: 'Processing' },
  SPEAKING: { color: '#66aaff', label: 'Speaking' },
  INTERRUPTED: { color: '#ff4466', label: 'Interrupted' },
};

const TRANSITIONS = [
  { from: 'IDLE', to: 'LISTENING', label: 'speech_start' },
  { from: 'LISTENING', to: 'PROCESSING', label: 'speech_end' },
  { from: 'PROCESSING', to: 'SPEAKING', label: 'first_chunk' },
  { from: 'SPEAKING', to: 'IDLE', label: 'playback_done' },
  { from: 'SPEAKING', to: 'INTERRUPTED', label: 'barge_in' },
  { from: 'INTERRUPTED', to: 'LISTENING', label: 'resume' },
  { from: 'LISTENING', to: 'IDLE', label: 'timeout' },
];

function StateNode({ name, active, x, y }) {
  const meta = STATE_META[name];
  return (
    <g>
      <circle
        cx={x}
        cy={y}
        r={28}
        fill={active ? meta.color : 'var(--bg-surface)'}
        stroke={meta.color}
        strokeWidth={active ? 3 : 1.5}
        opacity={active ? 1 : 0.5}
        style={{ transition: 'all 0.3s' }}
      />
      {active && (
        <circle
          cx={x}
          cy={y}
          r={34}
          fill="none"
          stroke={meta.color}
          strokeWidth={1}
          opacity={0.3}
          style={{ animation: 'pulse 2s ease-in-out infinite' }}
        />
      )}
      <text
        x={x}
        y={y + 1}
        textAnchor="middle"
        dominantBaseline="middle"
        fill={active ? '#0a0a0f' : '#8888a0'}
        fontSize={9}
        fontWeight={active ? 700 : 400}
      >
        {meta.label}
      </text>
    </g>
  );
}

export default function AudioPipelinePage({ signals: parentSignals, state }) {
  const localSignals = useSignals([
    'state_change',
    'filler_played',
    'barge_in',
  ]);
  const sig = { ...localSignals, ...parentSignals };

  const [stateHistory, setStateHistory] = useState([]);
  const [playlistItems, setPlaylistItems] = useState([]);

  useEffect(() => {
    if (sig.state_change) {
      setStateHistory(prev => [...prev.slice(-30), {
        from: sig.state_change.from,
        to: sig.state_change.to || sig.state_change.state,
        time: Date.now(),
      }]);
    }
  }, [sig.state_change]);

  // Simulate playlist from events
  useEffect(() => {
    if (sig.filler_played) {
      setPlaylistItems(prev => [...prev.slice(-20), { type: 'filler', text: 'filler', time: Date.now(), status: 'played' }]);
    }
  }, [sig.filler_played]);

  // Node positions in a circular layout
  const positions = {
    IDLE: { x: 120, y: 40 },
    LISTENING: { x: 220, y: 80 },
    PROCESSING: { x: 220, y: 160 },
    SPEAKING: { x: 120, y: 200 },
    INTERRUPTED: { x: 40, y: 120 },
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      {/* State diagram */}
      <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.75rem' }}>
        <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 4 }}>
          State Machine
        </div>
        <svg width="280" height="240" viewBox="0 0 280 240" style={{ display: 'block', margin: '0 auto' }}>
          {/* Transition arrows */}
          {TRANSITIONS.map((t, i) => {
            const from = positions[t.from];
            const to = positions[t.to];
            if (!from || !to) return null;
            const midX = (from.x + to.x) / 2;
            const midY = (from.y + to.y) / 2;
            return (
              <g key={i}>
                <line
                  x1={from.x}
                  y1={from.y}
                  x2={to.x}
                  y2={to.y}
                  stroke="#333350"
                  strokeWidth={1}
                  markerEnd="url(#arrowhead)"
                />
                <text x={midX} y={midY - 5} textAnchor="middle" fill="#555570" fontSize={7}>
                  {t.label}
                </text>
              </g>
            );
          })}
          {/* Arrow marker */}
          <defs>
            <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <polygon points="0 0, 8 3, 0 6" fill="#555570" />
            </marker>
          </defs>
          {/* State nodes */}
          {STATES.map(s => (
            <StateNode key={s} name={s} active={state === s} x={positions[s].x} y={positions[s].y} />
          ))}
        </svg>
      </div>

      {/* State transition history */}
      <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.75rem' }}>
        <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 4 }}>
          State Transitions
        </div>
        <div style={{ maxHeight: 120, overflow: 'auto', fontSize: '0.75rem' }}>
          {stateHistory.length === 0 && (
            <span style={{ color: 'var(--text-muted)' }}>No transitions yet</span>
          )}
          {stateHistory.slice().reverse().map((entry, i) => (
            <div key={i} style={{ color: 'var(--text-secondary)', marginBottom: 2 }}>
              <span style={{ color: STATE_META[entry.from]?.color || '#888' }}>{entry.from}</span>
              {' \u2192 '}
              <span style={{ color: STATE_META[entry.to]?.color || '#888' }}>{entry.to}</span>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.65rem', marginLeft: 6 }}>
                {new Date(entry.time).toLocaleTimeString()}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Playlist timeline */}
      <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.75rem' }}>
        <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 4 }}>
          Playlist Timeline
        </div>
        <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap', minHeight: 24 }}>
          {playlistItems.length === 0 && (
            <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>Empty</span>
          )}
          {playlistItems.map((item, i) => (
            <div
              key={i}
              title={`${item.type} - ${item.status}`}
              style={{
                width: 20,
                height: 20,
                borderRadius: 3,
                background: item.status === 'played' ? '#44cc88' : item.status === 'current' ? '#66aaff' : '#333350',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '0.55rem',
                color: '#0a0a0f',
                fontWeight: 600,
              }}
            >
              {item.type === 'filler' ? 'F' : 'C'}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
