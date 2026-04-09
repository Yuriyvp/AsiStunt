export default function WelcomeStep({ componentStatus = {} }) {
  const components = [
    { key: 'llm', label: 'Language Model', icon: '\uD83E\uDDE0' },
    { key: 'tts', label: 'Text-to-Speech', icon: '\uD83D\uDD0A' },
    { key: 'asr', label: 'Speech Recognition', icon: '\uD83C\uDF99\uFE0F' },
    { key: 'vad', label: 'Voice Detection', icon: '\uD83D\uDC42' },
  ];

  return (
    <div style={{ textAlign: 'center', padding: '1rem 0' }}>
      <div style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>{'\uD83D\uDC4B'}</div>
      <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '0.5rem' }}>
        Welcome to Voice Assistant
      </h2>
      <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
        Let's get everything set up. This will only take a minute.
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxWidth: 280, margin: '0 auto' }}>
        {components.map(comp => {
          const status = componentStatus[comp.key] || 'loading';
          return (
            <div
              key={comp.key}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
                padding: '0.5rem 0.75rem',
                background: 'var(--bg-surface)',
                borderRadius: 'var(--radius-sm)',
              }}
            >
              <span>{comp.icon}</span>
              <span style={{ flex: 1, fontSize: '0.85rem', textAlign: 'left' }}>{comp.label}</span>
              <StatusDot status={status} />
            </div>
          );
        })}
      </div>
    </div>
  );
}

function StatusDot({ status }) {
  const colors = {
    ready: 'var(--mic-listening)',
    loading: 'var(--mic-ptt)',
    error: 'var(--mic-muted)',
  };
  const labels = {
    ready: 'Ready',
    loading: 'Loading\u2026',
    error: 'Error',
  };

  return (
    <span style={{
      display: 'flex',
      alignItems: 'center',
      gap: '0.25rem',
      fontSize: '0.7rem',
      color: colors[status] || 'var(--text-muted)',
    }}>
      <span style={{
        width: 8,
        height: 8,
        borderRadius: '50%',
        background: colors[status] || 'var(--text-muted)',
        animation: status === 'loading' ? 'pulse 1.5s infinite' : 'none',
      }} />
      {labels[status] || status}
    </span>
  );
}
