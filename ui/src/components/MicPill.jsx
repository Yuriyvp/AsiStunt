import { useState, useCallback } from 'react';

const MIC_STATES = [
  { key: 'listening', icon: '🟢', label: 'Listening', color: 'var(--mic-listening)' },
  { key: 'muted',     icon: '🔴', label: 'Muted',     color: 'var(--mic-muted)' },
  { key: 'ptt',       icon: '🟡', label: 'Push to Talk', color: 'var(--mic-ptt)' },
  { key: 'text',      icon: '⌨️', label: 'Text Only', color: 'var(--mic-text)' },
];

export default function MicPill({ micState = 'listening', onStateChange }) {
  const currentIdx = MIC_STATES.findIndex(s => s.key === micState);
  const current = MIC_STATES[currentIdx >= 0 ? currentIdx : 0];

  const cycle = useCallback(() => {
    const nextIdx = (currentIdx + 1) % MIC_STATES.length;
    onStateChange?.(MIC_STATES[nextIdx].key);
  }, [currentIdx, onStateChange]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      cycle();
    }
  }, [cycle]);

  return (
    <button
      onClick={cycle}
      onKeyDown={handleKeyDown}
      aria-label={`Microphone: ${current.label}. Click to change.`}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 'var(--spacing-sm)',
        padding: '0.375rem 1rem',
        margin: '0 auto',
        border: `1px solid ${current.color}33`,
        borderRadius: 'var(--radius-pill)',
        background: `${current.color}15`,
        color: current.color,
        fontSize: '0.85rem',
        cursor: 'pointer',
        transition: 'var(--transition-state)',
      }}
    >
      <span style={{ fontSize: '0.75rem' }}>{current.icon}</span>
      <span>{current.label}</span>
    </button>
  );
}
