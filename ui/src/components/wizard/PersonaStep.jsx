export default function PersonaStep({ personas = [], selected = 'Aria', onSelect }) {
  const available = personas.length > 0
    ? personas
    : [{ name: 'Aria', description: 'Warm and thoughtful companion', mood: 'warm' }];

  return (
    <div style={{ padding: '1rem 0', textAlign: 'center' }}>
      <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.25rem' }}>
        Choose a Persona
      </h2>
      <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
        Select who you'd like to talk to
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxWidth: 300, margin: '0 auto' }}>
        {available.map(persona => {
          const active = persona.name === selected;
          return (
            <button
              key={persona.name}
              onClick={() => onSelect?.(persona.name)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
                padding: '0.75rem',
                background: active ? 'var(--bg-surface)' : 'transparent',
                border: `1px solid ${active ? 'var(--orb-listening)' : 'var(--text-muted)'}`,
                borderRadius: 'var(--radius-md)',
                color: 'var(--text-primary)',
                fontSize: '0.85rem',
                cursor: 'pointer',
                transition: 'all 0.2s',
                textAlign: 'left',
              }}
            >
              <span style={{
                width: 36,
                height: 36,
                borderRadius: '50%',
                background: `var(--mood-${persona.mood || 'neutral'})`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '1.1rem',
                flexShrink: 0,
                opacity: 0.8,
              }}>
                {persona.name[0]}
              </span>
              <div>
                <div style={{ fontWeight: 600, marginBottom: 2 }}>{persona.name}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                  {persona.description}
                </div>
              </div>
              {active && (
                <span style={{ marginLeft: 'auto', color: 'var(--orb-listening)', fontSize: '1rem' }}>
                  {'\u2713'}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
