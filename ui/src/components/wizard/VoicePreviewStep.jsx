import { useState, useCallback } from 'react';

export default function VoicePreviewStep({ sendCommand, cloneProgress = null }) {
  const [playing, setPlaying] = useState(false);
  const [played, setPlayed] = useState(false);

  const handlePreview = useCallback(() => {
    setPlaying(true);
    sendCommand?.({ cmd: 'voice_preview' });
    // The sidecar will play the preview audio through the speaker
    // Simulate completion after a reasonable time
    setTimeout(() => {
      setPlaying(false);
      setPlayed(true);
    }, 3000);
  }, [sendCommand]);

  return (
    <div style={{ padding: '1rem 0', textAlign: 'center' }}>
      <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.75rem' }}>
        Voice Preview
      </h2>
      <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
        Listen to a short greeting to preview the voice.
      </p>

      {/* Clone progress */}
      {cloneProgress !== null && cloneProgress < 100 && (
        <div style={{ marginBottom: '1rem', maxWidth: 260, margin: '0 auto 1rem' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
            Cloning voice\u2026 {Math.round(cloneProgress)}%
          </div>
          <div style={{
            width: '100%',
            height: 6,
            background: 'var(--bg-surface)',
            borderRadius: 3,
            overflow: 'hidden',
          }}>
            <div style={{
              width: `${cloneProgress}%`,
              height: '100%',
              background: 'var(--orb-processing)',
              borderRadius: 3,
              transition: 'width 0.3s',
            }} />
          </div>
        </div>
      )}

      <button
        onClick={handlePreview}
        disabled={playing || (cloneProgress !== null && cloneProgress < 100)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          margin: '0 auto',
          padding: '0.75rem 1.5rem',
          background: playing ? 'var(--bg-surface)' : 'var(--orb-speaking)',
          border: 'none',
          borderRadius: 'var(--radius-pill)',
          color: playing ? 'var(--text-secondary)' : 'var(--bg-primary)',
          fontSize: '0.9rem',
          fontWeight: 600,
          cursor: playing ? 'default' : 'pointer',
          transition: 'all 0.2s',
        }}
      >
        {playing ? '\u23F8\uFE0F Playing\u2026' : played ? '\uD83D\uDD01 Play Again' : '\u25B6\uFE0F Play Preview'}
      </button>

      {played && (
        <div style={{
          marginTop: '1rem',
          fontSize: '0.8rem',
          color: 'var(--mic-listening)',
        }}>
          {'\u2713'} Voice preview complete
        </div>
      )}
    </div>
  );
}
