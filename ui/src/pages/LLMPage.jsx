import { useState, useEffect, useRef } from 'react';
import { useSignals } from '../hooks/useSignals';

export default function LLMPage({ signals: parentSignals, events, sendCommand }) {
  const localSignals = useSignals([
    'request_start',
    'first_token',
    'complete',
    'tokens_per_sec',
  ]);
  const sig = { ...localSignals, ...parentSignals };

  const [tokenStream, setTokenStream] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [stats, setStats] = useState({ tps: 0, promptTokens: 0, completionTokens: 0, ttft: 0 });
  const [promptText, setPromptText] = useState('');
  const [showPrompt, setShowPrompt] = useState(false);
  const streamRef = useRef(null);

  // Process events for token stream
  useEffect(() => {
    if (!events || events.length === 0) return;
    const last = events[events.length - 1];
    if (last.event === 'signal') {
      if (last.type === 'request_start') {
        setIsGenerating(true);
        setTokenStream('');
        setPromptText(last.prompt || '');
      }
      if (last.type === 'first_token') {
        setStats(prev => ({ ...prev, ttft: last.latency_ms || 0 }));
      }
      if (last.type === 'complete') {
        setIsGenerating(false);
        setStats(prev => ({
          ...prev,
          promptTokens: last.prompt_tokens || 0,
          completionTokens: last.completion_tokens || 0,
        }));
      }
      if (last.type === 'tokens_per_sec') {
        setStats(prev => ({ ...prev, tps: last.tps || 0 }));
      }
    }
    // Accumulate token content from events
    if (last.event === 'token') {
      setTokenStream(prev => prev + (last.text || ''));
    }
  }, [events]);

  useEffect(() => {
    if (streamRef.current) {
      streamRef.current.scrollTop = streamRef.current.scrollHeight;
    }
  }, [tokenStream]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      {/* Stats row */}
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
        <StatBox label="Status" value={isGenerating ? 'Generating...' : 'Idle'} color={isGenerating ? '#aa66dd' : 'var(--text-muted)'} />
        <StatBox label="Tokens/s" value={stats.tps.toFixed(1)} />
        <StatBox label="TTFT" value={`${stats.ttft.toFixed(0)} ms`} />
        <StatBox label="Prompt" value={stats.promptTokens} unit="tok" />
        <StatBox label="Completion" value={stats.completionTokens} unit="tok" />
      </div>

      {/* Live token stream */}
      <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 4 }}>
          <span>Token Stream</span>
          {isGenerating && (
            <span style={{ color: '#aa66dd', animation: 'pulse 1s ease-in-out infinite' }}>
              {'\u25CF'} Live
            </span>
          )}
        </div>
        <div
          ref={streamRef}
          style={{
            maxHeight: 200,
            overflow: 'auto',
            fontSize: '0.8rem',
            fontFamily: 'monospace',
            color: 'var(--text-primary)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            lineHeight: 1.5,
          }}
        >
          {tokenStream || <span style={{ color: 'var(--text-muted)' }}>Waiting for generation...</span>}
          {isGenerating && <span style={{ color: '#aa66dd', animation: 'pulse 0.8s ease-in-out infinite' }}>{'\u2588'}</span>}
        </div>
      </div>

      {/* Prompt inspector */}
      <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.75rem' }}>
        <div
          onClick={() => setShowPrompt(!showPrompt)}
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: '0.7rem',
            color: 'var(--text-muted)',
            cursor: 'pointer',
          }}
        >
          <span>Prompt Inspector</span>
          <span>{showPrompt ? '\u25BC' : '\u25B6'}</span>
        </div>
        {showPrompt && (
          <div style={{
            marginTop: 6,
            maxHeight: 200,
            overflow: 'auto',
            fontSize: '0.75rem',
            fontFamily: 'monospace',
            color: 'var(--text-secondary)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            lineHeight: 1.4,
            borderTop: '1px solid var(--bg-secondary)',
            paddingTop: 6,
          }}>
            {promptText || <span style={{ color: 'var(--text-muted)' }}>No prompt captured yet</span>}
          </div>
        )}
      </div>
    </div>
  );
}

function StatBox({ label, value, unit, color }) {
  return (
    <div style={{
      background: 'var(--bg-surface)',
      borderRadius: 'var(--radius-sm)',
      padding: '0.5rem 0.6rem',
      flex: '1 1 80px',
      minWidth: 80,
    }}>
      <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>{label}</div>
      <div style={{ fontSize: '1rem', fontWeight: 600, color: color || 'var(--text-primary)', marginTop: 1 }}>
        {value}
        {unit && <span style={{ fontSize: '0.6rem', fontWeight: 400, color: 'var(--text-muted)', marginLeft: 2 }}>{unit}</span>}
      </div>
    </div>
  );
}
