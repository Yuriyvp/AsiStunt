import { useState, useEffect, useRef, useCallback } from 'react';
import { listen } from '@tauri-apps/api/event';
import { SettingsPage, SectionLabel } from './SettingsUI';

const MAX_POINTS = 200;

export default function VADSettings({ sendCommand, onBack }) {
  const [energyHistory, setEnergyHistory] = useState([]);
  const [threshold, setThreshold] = useState(0.5);
  const [isSpeech, setIsSpeech] = useState(false);
  const [vadEvents, setVadEvents] = useState([]);
  const tickRef = useRef(0);
  const canvasRef = useRef(null);

  useEffect(() => {
    const unlisten = listen('python_event', (event) => {
      const data = event.payload;
      if (data.event !== 'signal') return;

      if (data.type === 'audio.input_level' || data.type === 'input_level') {
        const level = data.level || 0;
        tickRef.current++;
        setEnergyHistory(prev => {
          const next = [...prev, level];
          return next.length > MAX_POINTS ? next.slice(-MAX_POINTS) : next;
        });
      }
      if (data.type === 'speech_start') {
        setIsSpeech(true);
        setVadEvents(prev => [...prev.slice(-50), { type: 'start', time: Date.now() }]);
      }
      if (data.type === 'speech_end') {
        setIsSpeech(false);
        setVadEvents(prev => [...prev.slice(-50), { type: 'end', time: Date.now(), duration_ms: data.duration_ms }]);
      }
    });
    return () => { unlisten.then(fn => fn()); };
  }, []);

  // Draw energy waveform on canvas (no recharts dependency needed)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || energyHistory.length === 0) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Background grid
    ctx.strokeStyle = '#333350';
    ctx.lineWidth = 0.5;
    for (let y = 0; y <= 1; y += 0.25) {
      const py = h - y * h;
      ctx.beginPath();
      ctx.moveTo(0, py);
      ctx.lineTo(w, py);
      ctx.stroke();
    }

    // Threshold line
    const ty = h - threshold * h;
    ctx.strokeStyle = '#ccaa44';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(0, ty);
    ctx.lineTo(w, ty);
    ctx.stroke();
    ctx.setLineDash([]);

    // Energy line
    ctx.strokeStyle = '#44cc88';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const step = w / MAX_POINTS;
    energyHistory.forEach((val, i) => {
      const x = i * step;
      const y = h - val * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }, [energyHistory, threshold]);

  const handleThresholdChange = useCallback((e) => {
    const val = parseFloat(e.target.value);
    setThreshold(val);
    sendCommand?.({ cmd: 'set_vad_threshold', threshold: val });
  }, [sendCommand]);

  return (
    <SettingsPage title="VAD" onBack={onBack}>
      {/* Speech indicator */}
      <SectionLabel first>Status</SectionLabel>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.3rem 0' }}>
        <div style={{
          width: 12, height: 12, borderRadius: '50%',
          background: isSpeech ? '#44cc88' : '#555570',
          boxShadow: isSpeech ? '0 0 8px #44cc88' : 'none',
          transition: 'all 0.2s',
        }} />
        <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
          {isSpeech ? 'Speech Detected' : 'Silence'}
        </span>
      </div>

      {/* Threshold slider */}
      <SectionLabel>Threshold</SectionLabel>
      <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 4 }}>
          <span>VAD Sensitivity</span>
          <span>{threshold.toFixed(2)}</span>
        </div>
        <input
          type="range" min="0" max="1" step="0.01"
          value={threshold} onChange={handleThresholdChange}
          style={{ width: '100%', accentColor: '#ccaa44' }}
        />
      </div>

      {/* Energy waveform */}
      <SectionLabel>Diagnostics</SectionLabel>
      <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.75rem' }}>
        <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 4 }}>
          Energy / Speech Probability
        </div>
        <canvas
          ref={canvasRef}
          width={400}
          height={120}
          style={{ width: '100%', height: 120, borderRadius: 'var(--radius-sm)' }}
        />
        <div style={{ display: 'flex', gap: '1rem', fontSize: '0.6rem', color: 'var(--text-muted)', marginTop: 4 }}>
          <span><span style={{ color: '#44cc88' }}>{'\u2014'}</span> Energy</span>
          <span><span style={{ color: '#ccaa44' }}>- -</span> Threshold</span>
        </div>
      </div>

      {/* Recent VAD events */}
      <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.75rem', marginTop: '0.5rem' }}>
        <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 4 }}>
          Recent Events
        </div>
        <div style={{ maxHeight: 150, overflow: 'auto', fontSize: '0.75rem' }}>
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
    </SettingsPage>
  );
}
