/**
 * Shared UI primitives for all Settings module pages.
 */

export function SettingRow({ label, children, column }) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: column ? 'column' : 'row',
      justifyContent: column ? 'flex-start' : 'space-between',
      alignItems: column ? 'stretch' : 'center',
      gap: column ? '0.35rem' : 0,
      padding: '0.5rem 0',
      borderBottom: '1px solid var(--bg-surface)',
    }}>
      <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{label}</span>
      {children}
    </div>
  );
}

export function Toggle({ value, onChange, label }) {
  return (
    <button
      role="switch"
      aria-checked={value}
      aria-label={label}
      onClick={() => onChange(!value)}
      style={{
        width: 36, height: 20, borderRadius: 10, border: 'none',
        background: value ? 'var(--orb-listening)' : 'var(--text-muted)',
        position: 'relative', cursor: 'pointer', transition: 'background 0.2s', padding: 0,
      }}
    >
      <span style={{
        position: 'absolute', top: 2, left: value ? 18 : 2,
        width: 16, height: 16, borderRadius: '50%',
        background: 'var(--text-primary)', transition: 'left 0.2s',
      }} />
    </button>
  );
}

export function Select({ value, options, onChange }) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        background: 'var(--bg-surface)',
        border: '1px solid var(--text-muted)',
        borderRadius: 'var(--radius-sm)',
        padding: '0.3rem 0.5rem',
        color: 'var(--text-primary)',
        fontSize: '0.8rem',
        cursor: 'pointer',
        outline: 'none',
        colorScheme: 'dark',
      }}
    >
      {options.map(opt => (
        <option key={opt.value} value={opt.value}>{opt.label}</option>
      ))}
    </select>
  );
}

export function SectionLabel({ children, first }) {
  return (
    <div style={{
      fontSize: '0.7rem',
      color: 'var(--text-muted)',
      textTransform: 'uppercase',
      letterSpacing: '0.05em',
      marginBottom: '0.25rem',
      marginTop: first ? '0.25rem' : '0.75rem',
    }}>
      {children}
    </div>
  );
}

export function SmallButton({ onClick, children, color, disabled }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        background: 'none',
        border: `1px solid ${color || 'var(--text-muted)'}`,
        borderRadius: 'var(--radius-sm)',
        color: color || 'var(--text-secondary)',
        fontSize: '0.7rem',
        padding: '3px 10px',
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.5 : 1,
        transition: 'all 0.15s',
        whiteSpace: 'nowrap',
      }}
    >
      {children}
    </button>
  );
}

export function LevelBar({ level }) {
  return (
    <div style={{
      flex: 1, height: 6, background: 'var(--bg-surface)',
      borderRadius: 3, overflow: 'hidden',
    }}>
      <div style={{
        width: `${Math.round(level * 100)}%`,
        height: '100%',
        background: level > 0.8 ? 'var(--mic-muted)' : level > 0.4 ? 'var(--mic-ptt)' : 'var(--mic-listening)',
        borderRadius: 3,
        transition: 'width 0.1s, background 0.2s',
      }} />
    </div>
  );
}

/**
 * Standard sub-page wrapper with back arrow and title.
 */
export function SettingsPage({ title, onBack, children }) {
  return (
    <div style={{
      position: 'absolute', inset: 0,
      background: 'var(--bg-primary)',
      zIndex: 101,
      display: 'flex', flexDirection: 'column',
      overflow: 'hidden',
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: '0.5rem',
        padding: '0.75rem', borderBottom: '1px solid var(--bg-surface)',
      }}>
        <button
          onClick={onBack}
          style={{
            background: 'none', border: 'none',
            color: 'var(--text-secondary)', fontSize: '1rem',
            cursor: 'pointer', padding: '0.25rem',
          }}
        >
          {'\u2190'}
        </button>
        <span style={{ fontSize: '0.95rem', fontWeight: 600 }}>{title}</span>
      </div>
      <div style={{
        flex: 1, overflowY: 'auto', padding: '0.5rem 0.75rem',
        scrollbarWidth: 'thin', scrollbarColor: 'var(--bg-surface) transparent',
      }}>
        {children}
      </div>
    </div>
  );
}
