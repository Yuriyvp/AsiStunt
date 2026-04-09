import { useState, useEffect, useRef, useCallback } from 'react';
import { useSignals } from '../hooks/useSignals';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, ReferenceLine } from 'recharts';

const MAX_POINTS = 200;

export default function VADPage({ signals: parentSignals }) {
  const localSignals = useSignals([
    'audio.input_level',
    'speech_start',
    'speech_end',
  ]);
  const sig = { ...localSignals, ...parentSignals };

  const [energyHistory, setEnergyHistory] = useState([]);
  const [threshold, setThreshold] = useState(0.5);
  const [isSpeech, setIsSpeech] = useState(false);
  const [vadEvents, setVadEvents] = useState([]);
  const tickRef = useRef(0);

  // Accumulate energy readings
  useEffect(() => {
    if (sig['audio.input_level']) {
      const level = sig['audio.input_level'].level || 0;
      tickRef.current++;
      setEnergyHistory(prev => {
        const next = [...prev, { t: tickRef.current, energy: level, threshold }];
        return next.length > MAX_POINTS ? next.slice(-MAX_POINTS) : next;
      });
    }
  }, [sig['audio.input_level'], threshold]);

  useEffect(() => {
    if (sig.speech_start) {
      setIsSpeech(true);
      setVadEvents(prev => [...prev.slice(-50), { type: 'start', time: Date.now() }]);
    }
  }, [sig.speech_start]);

  useEffect(() => {
    if (sig.speech_end) {
      setIsSpeech(false);
      setVadEvents(prev => [...prev.slice(-50), { type: 'end', time: Date.now(), duration_ms: sig.speech_end.duration_ms }]);
    }
  }, [sig.speech_end]);

  const handleThresholdChange = useCallback((e) => {
    setThreshold(parseFloat(e.target.value));
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      {/* Speech indicator */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <div style={{
          width: 12,
          height: 12,
          borderRadius: '50%',
          background: isSpeech ? '#44cc88' : '#555570',
          boxShadow: isSpeech ? '0 0 8px #44cc88' : 'none',
          transition: 'all 0.2s',
        }} />
        <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
          {isSpeech ? 'Speech Detected' : 'Silence'}
        </span>
      </div>

      {/* Energy waveform */}
      <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.75rem' }}>
        <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 4 }}>
          Energy / Speech Probability
        </div>
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={energyHistory}>
            <XAxis dataKey="t" hide />
            <YAxis domain={[0, 1]} hide />
            <ReferenceLine y={threshold} stroke="#ccaa44" strokeDasharray="3 3" label={{ value: 'Threshold', fill: '#ccaa44', fontSize: 10, position: 'right' }} />
            <Line type="monotone" dataKey="energy" stroke="#44cc88" dot={false} strokeWidth={1.5} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Threshold slider */}
      <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 4 }}>
          <span>VAD Threshold</span>
          <span>{threshold.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={threshold}
          onChange={handleThresholdChange}
          style={{ width: '100%', accentColor: '#ccaa44' }}
        />
      </div>

      {/* Recent VAD events */}
      <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.75rem' }}>
        <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 4 }}>
          Recent VAD Events
        </div>
        <div style={{ maxHeight: 120, overflow: 'auto', fontSize: '0.75rem' }}>
          {vadEvents.length === 0 && (
            <span style={{ color: 'var(--text-muted)' }}>No events yet</span>
          )}
          {vadEvents.slice().reverse().map((evt, i) => (
            <div key={i} style={{ color: 'var(--text-secondary)', marginBottom: 2 }}>
              <span style={{ color: evt.type === 'start' ? '#44cc88' : '#ff4466' }}>
                {evt.type === 'start' ? '\u25B6' : '\u25A0'}
              </span>
              {' '}
              {evt.type === 'start' ? 'Speech Start' : `Speech End (${evt.duration_ms || '?'}ms)`}
              {' '}
              <span style={{ color: 'var(--text-muted)', fontSize: '0.65rem' }}>
                {new Date(evt.time).toLocaleTimeString()}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
