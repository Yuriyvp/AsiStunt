import { useState } from 'react';

const LANGUAGES = [
  { code: 'hr', name: 'Croatian' },
  { code: 'en', name: 'English' },
  { code: 'de', name: 'German' },
  { code: 'fr', name: 'French' },
  { code: 'es', name: 'Spanish' },
  { code: 'it', name: 'Italian' },
  { code: 'pt', name: 'Portuguese' },
  { code: 'nl', name: 'Dutch' },
  { code: 'pl', name: 'Polish' },
  { code: 'cs', name: 'Czech' },
  { code: 'sk', name: 'Slovak' },
  { code: 'sl', name: 'Slovenian' },
  { code: 'hu', name: 'Hungarian' },
  { code: 'ro', name: 'Romanian' },
  { code: 'bg', name: 'Bulgarian' },
  { code: 'sr', name: 'Serbian' },
  { code: 'uk', name: 'Ukrainian' },
  { code: 'ru', name: 'Russian' },
  { code: 'ja', name: 'Japanese' },
  { code: 'ko', name: 'Korean' },
  { code: 'zh', name: 'Chinese' },
  { code: 'ar', name: 'Arabic' },
  { code: 'tr', name: 'Turkish' },
  { code: 'sv', name: 'Swedish' },
  { code: 'da', name: 'Danish' },
];

export default function LanguageStep({ defaultLanguages = ['hr', 'en'], onSelect }) {
  const [selected, setSelected] = useState(new Set(defaultLanguages));

  const toggle = (code) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(code)) {
        if (next.size > 1) next.delete(code);
      } else if (next.size < 5) {
        next.add(code);
      }
      onSelect?.(Array.from(next));
      return next;
    });
  };

  return (
    <div style={{ padding: '1rem 0' }}>
      <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.25rem', textAlign: 'center' }}>
        Languages
      </h2>
      <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '1rem', textAlign: 'center' }}>
        Select 1\u20135 languages you'll use ({selected.size}/5)
      </p>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2, 1fr)',
        gap: '0.35rem',
        maxHeight: 280,
        overflowY: 'auto',
        padding: '0 0.25rem',
        scrollbarWidth: 'thin',
        scrollbarColor: 'var(--bg-surface) transparent',
      }}>
        {LANGUAGES.map(lang => {
          const active = selected.has(lang.code);
          return (
            <button
              key={lang.code}
              onClick={() => toggle(lang.code)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.4rem 0.6rem',
                background: active ? 'var(--bg-surface)' : 'transparent',
                border: `1px solid ${active ? 'var(--orb-listening)' : 'var(--text-muted)'}`,
                borderRadius: 'var(--radius-sm)',
                color: active ? 'var(--text-primary)' : 'var(--text-secondary)',
                fontSize: '0.75rem',
                cursor: 'pointer',
                transition: 'all 0.15s',
                textAlign: 'left',
              }}
            >
              <span style={{
                width: 14,
                height: 14,
                borderRadius: 3,
                border: `1px solid ${active ? 'var(--orb-listening)' : 'var(--text-muted)'}`,
                background: active ? 'var(--orb-listening)' : 'transparent',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '0.6rem',
                color: 'var(--bg-primary)',
                flexShrink: 0,
              }}>
                {active ? '\u2713' : ''}
              </span>
              <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', width: 20 }}>{lang.code}</span>
              <span>{lang.name}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
