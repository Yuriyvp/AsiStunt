import { useState } from 'react';

export default function ConsentStep({ onConsent }) {
  const [acknowledged, setAcknowledged] = useState(false);

  return (
    <div style={{ padding: '1rem 0' }}>
      <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.75rem', textAlign: 'center' }}>
        Privacy & Microphone Access
      </h2>

      <div style={{
        background: 'var(--bg-surface)',
        borderRadius: 'var(--radius-md)',
        padding: '1rem',
        fontSize: '0.8rem',
        color: 'var(--text-secondary)',
        lineHeight: 1.6,
        marginBottom: '1rem',
      }}>
        <p style={{ marginBottom: '0.75rem' }}>
          This assistant runs <strong style={{ color: 'var(--text-primary)' }}>entirely on your device</strong>.
          No audio, text, or personal data leaves your computer.
        </p>
        <p style={{ marginBottom: '0.75rem' }}>
          Voice mode requires microphone access. Your audio is processed locally
          for speech recognition and is never stored or transmitted.
        </p>
        <p>
          You can use <strong style={{ color: 'var(--text-primary)' }}>text-only mode</strong> if
          you prefer not to use the microphone.
        </p>
      </div>

      <label style={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: '0.5rem',
        cursor: 'pointer',
        fontSize: '0.8rem',
        color: 'var(--text-primary)',
        padding: '0.5rem',
        borderRadius: 'var(--radius-sm)',
        background: acknowledged ? 'var(--bg-surface)' : 'transparent',
        transition: 'background 0.2s',
      }}>
        <input
          type="checkbox"
          checked={acknowledged}
          onChange={(e) => {
            setAcknowledged(e.target.checked);
            onConsent?.(e.target.checked);
          }}
          style={{ marginTop: 2, accentColor: 'var(--orb-listening)' }}
        />
        <span>I understand that voice mode uses my microphone for local speech recognition</span>
      </label>
    </div>
  );
}
