import { useState, useEffect, useRef } from 'react';

export default function AudioCheckStep({ energy = 0 }) {
  const [tested, setTested] = useState(false);
  const [level, setLevel] = useState(0);
  const peakRef = useRef(0);

  useEffect(() => {
    setLevel(energy);
    if (energy > 0.1) {
      setTested(true);
      if (energy > peakRef.current) peakRef.current = energy;
    }
  }, [energy]);

  return (
    <div style={{ padding: '1rem 0', textAlign: 'center' }}>
      <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.75rem' }}>
        Audio Check
      </h2>
      <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
        Say something to check your microphone is working.
      </p>

      {/* Level meter */}
      <div style={{
        width: '100%',
        maxWidth: 260,
        height: 12,
        background: 'var(--bg-surface)',
        borderRadius: 6,
        margin: '0 auto 1rem',
        overflow: 'hidden',
      }}>
        <div style={{
          width: `${Math.min(100, level * 100)}%`,
          height: '100%',
          background: level > 0.1
            ? 'var(--mic-listening)'
            : 'var(--text-muted)',
          borderRadius: 6,
          transition: 'width 0.1s',
        }} />
      </div>

      <div style={{
        fontSize: '0.8rem',
        color: tested ? 'var(--mic-listening)' : 'var(--text-muted)',
        marginBottom: '0.5rem',
      }}>
        {tested ? '\u2713 Microphone is working' : 'Waiting for audio\u2026'}
      </div>

      <p style={{
        fontSize: '0.7rem',
        color: 'var(--text-muted)',
        marginTop: '1rem',
      }}>
        You can skip this step if you plan to use text-only mode.
      </p>
    </div>
  );
}
