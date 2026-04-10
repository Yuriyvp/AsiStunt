import { useState, useEffect } from 'react';
import { listen } from '@tauri-apps/api/event';

export default function TTSPage({ signals: parentSignals, events }) {
  const [isSynthesizing, setIsSynthesizing] = useState(false);
  const [synthHistory, setSynthHistory] = useState([]);

  useEffect(() => {
    const unlisten = listen('python_event', (event) => {
      const data = event.payload;
      if (data.event !== 'signal') return;

      if (data.type === 'synth_start') setIsSynthesizing(true);
      if (data.type === 'synth_end') {
        setIsSynthesizing(false);
        setSynthHistory(prev => [...prev.slice(-30), {
          text: data.text || '',
          lang: data.lang || '',
          rtf: data.rtf || 0,
          duration_ms: data.duration_ms || 0,
          error: data.error || null,
          time: Date.now(),
        }]);
      }
    });
    return () => { unlisten.then(fn => fn()); };
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      {/* Status */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <div style={{
          width: 12, height: 12, borderRadius: '50%',
          background: isSynthesizing ? '#cc8844' : '#555570',
          boxShadow: isSynthesizing ? '0 0 8px #cc8844' : 'none',
          transition: 'all 0.2s',
        }} />
        <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
          {isSynthesizing ? 'Synthesizing...' : 'Idle'}
        </span>
      </div>

      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', padding: '0.3rem 0' }}>
        Use Settings &rarr; TTS Settings to configure voices and test synthesis.
      </div>

      {/* Synthesis Log */}
      <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.75rem' }}>
        <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 4 }}>
          Synthesis Log
        </div>
        <div style={{ maxHeight: 350, overflow: 'auto' }}>
          {synthHistory.length === 0 && (
            <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>No synthesis yet</span>
          )}
          {synthHistory.slice().reverse().map((entry, i) => (
            <div key={i} style={{
              padding: '0.4rem 0',
              borderBottom: '1px solid var(--bg-secondary)',
              fontSize: '0.75rem',
            }}>
              <div style={{ color: 'var(--text-primary)', marginBottom: 2, wordBreak: 'break-word' }}>
                {entry.text.slice(0, 100)}{entry.text.length > 100 ? '...' : ''}
                {entry.lang && <span style={{ color: 'var(--text-muted)', marginLeft: 6 }}>[{entry.lang}]</span>}
              </div>
              <div style={{ display: 'flex', gap: '1rem', color: 'var(--text-muted)', fontSize: '0.65rem' }}>
                {entry.error ? (
                  <span style={{ color: '#ff4466' }}>Error: {entry.error}</span>
                ) : (
                  <>
                    <span>RTF: <span style={{ color: entry.rtf < 1 ? '#44cc88' : '#ff4466' }}>{entry.rtf.toFixed(2)}</span></span>
                    <span>{entry.duration_ms}ms</span>
                  </>
                )}
                <span>{new Date(entry.time).toLocaleTimeString()}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
