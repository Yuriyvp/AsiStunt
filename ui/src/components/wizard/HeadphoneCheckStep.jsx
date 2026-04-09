import { useState } from 'react';

export default function HeadphoneCheckStep() {
  const [answer, setAnswer] = useState(null);

  return (
    <div style={{ padding: '1rem 0', textAlign: 'center' }}>
      <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.75rem' }}>
        Headphone Check
      </h2>
      <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
        Are you wearing headphones?
      </p>

      <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'center', marginBottom: '1rem' }}>
        {[
          { key: 'yes', label: 'Yes', icon: '\uD83C\uDFA7' },
          { key: 'no', label: 'No', icon: '\uD83D\uDD07' },
        ].map(opt => (
          <button
            key={opt.key}
            onClick={() => setAnswer(opt.key)}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '1rem 1.5rem',
              background: answer === opt.key ? 'var(--bg-surface)' : 'transparent',
              border: `1px solid ${answer === opt.key ? 'var(--orb-listening)' : 'var(--text-muted)'}`,
              borderRadius: 'var(--radius-md)',
              color: 'var(--text-primary)',
              fontSize: '0.85rem',
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
          >
            <span style={{ fontSize: '1.5rem' }}>{opt.icon}</span>
            <span>{opt.label}</span>
          </button>
        ))}
      </div>

      {answer === 'no' && (
        <div style={{
          background: 'rgba(204, 170, 68, 0.1)',
          border: '1px solid var(--mood-concerned)',
          borderRadius: 'var(--radius-sm)',
          padding: '0.75rem',
          fontSize: '0.8rem',
          color: 'var(--mood-concerned)',
          maxWidth: 300,
          margin: '0 auto',
        }}>
          Without headphones, the assistant may hear its own voice and create feedback loops.
          Consider using headphones for the best experience.
        </div>
      )}
    </div>
  );
}
