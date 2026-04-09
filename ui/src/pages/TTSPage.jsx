import { useState, useEffect } from 'react';
import { useSignals } from '../hooks/useSignals';

export default function TTSPage({ signals: parentSignals, events, sendCommand }) {
  const localSignals = useSignals([
    'synth_start',
    'synth_end',
  ]);
  const sig = { ...localSignals, ...parentSignals };

  const [isSynthesizing, setIsSynthesizing] = useState(false);
  const [synthHistory, setSynthHistory] = useState([]);
  const [testText, setTestText] = useState('');

  useEffect(() => {
    if (!events || events.length === 0) return;
    const last = events[events.length - 1];
    if (last.event === 'signal') {
      if (last.type === 'synth_start') {
        setIsSynthesizing(true);
      }
      if (last.type === 'synth_end') {
        setIsSynthesizing(false);
        setSynthHistory(prev => [...prev.slice(-20), {
          text: last.text || '',
          rtf: last.rtf || 0,
          duration_ms: last.duration_ms || 0,
          time: Date.now(),
        }]);
      }
    }
  }, [events]);

  const handleTestSynth = () => {
    if (!testText.trim()) return;
    sendCommand?.({ cmd: 'text_input', text: `[tts_test] ${testText}` });
    setTestText('');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      {/* Status */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <div style={{
          width: 12,
          height: 12,
          borderRadius: '50%',
          background: isSynthesizing ? '#cc8844' : '#555570',
          boxShadow: isSynthesizing ? '0 0 8px #cc8844' : 'none',
          transition: 'all 0.2s',
        }} />
        <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
          {isSynthesizing ? 'Synthesizing...' : 'Idle'}
        </span>
      </div>

      {/* Test bench */}
      <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.75rem' }}>
        <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 4 }}>
          Test Bench
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <input
            type="text"
            value={testText}
            onChange={(e) => setTestText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleTestSynth()}
            placeholder="Type text to synthesize..."
            style={{
              flex: 1,
              background: 'var(--bg-secondary)',
              border: '1px solid var(--text-muted)',
              borderRadius: 'var(--radius-sm)',
              padding: '0.4rem 0.6rem',
              color: 'var(--text-primary)',
              fontSize: '0.8rem',
              outline: 'none',
            }}
          />
          <button
            onClick={handleTestSynth}
            disabled={!testText.trim()}
            style={{
              background: testText.trim() ? '#cc8844' : 'var(--bg-surface)',
              border: 'none',
              borderRadius: 'var(--radius-sm)',
              color: testText.trim() ? 'var(--bg-primary)' : 'var(--text-muted)',
              fontSize: '0.8rem',
              fontWeight: 600,
              padding: '0.4rem 0.75rem',
              cursor: testText.trim() ? 'pointer' : 'default',
            }}
          >
            Speak
          </button>
        </div>
      </div>

      {/* Synthesis history */}
      <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.75rem' }}>
        <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 4 }}>
          Synthesis History
        </div>
        <div style={{ maxHeight: 250, overflow: 'auto' }}>
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
              </div>
              <div style={{ display: 'flex', gap: '1rem', color: 'var(--text-muted)', fontSize: '0.65rem' }}>
                <span>RTF: <span style={{ color: entry.rtf < 1 ? '#44cc88' : '#ff4466' }}>{entry.rtf.toFixed(2)}</span></span>
                <span>{entry.duration_ms}ms</span>
                <span>{new Date(entry.time).toLocaleTimeString()}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
