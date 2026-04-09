import { useState, useCallback, useEffect } from 'react';

export default function SOULPage({ sendCommand, events }) {
  const [yaml, setYaml] = useState('');
  const [validation, setValidation] = useState({ valid: true, errors: [] });
  const [lastReload, setLastReload] = useState(null);
  const [loading, setLoading] = useState(false);

  // Load initial SOUL yaml (request from backend)
  useEffect(() => {
    sendCommand?.({ cmd: 'get_soul_yaml' });
  }, [sendCommand]);

  // Listen for soul_yaml response
  useEffect(() => {
    if (!events || events.length === 0) return;
    const last = events[events.length - 1];
    if (last.event === 'soul_yaml') {
      setYaml(last.content || '');
    }
    if (last.event === 'soul_validation') {
      setValidation({
        valid: last.valid ?? true,
        errors: last.errors || [],
      });
    }
    if (last.event === 'signal' && last.type === 'process_state_change') {
      setLoading(false);
    }
  }, [events]);

  const handleValidate = useCallback(() => {
    sendCommand?.({ cmd: 'validate_soul', content: yaml });
  }, [sendCommand, yaml]);

  const handleReload = useCallback(() => {
    setLoading(true);
    sendCommand?.({ cmd: 'reload_soul', content: yaml });
    setLastReload(new Date().toLocaleTimeString());
  }, [sendCommand, yaml]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', height: '100%' }}>
      {/* Toolbar */}
      <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
        <button
          onClick={handleValidate}
          style={{
            background: 'var(--bg-surface)',
            border: '1px solid var(--text-muted)',
            borderRadius: 'var(--radius-sm)',
            color: 'var(--text-secondary)',
            fontSize: '0.75rem',
            padding: '0.35rem 0.75rem',
            cursor: 'pointer',
          }}
        >
          Validate
        </button>
        <button
          onClick={handleReload}
          disabled={loading}
          style={{
            background: validation.valid ? '#44cc88' : 'var(--bg-surface)',
            border: 'none',
            borderRadius: 'var(--radius-sm)',
            color: validation.valid ? '#0a0a0f' : 'var(--text-muted)',
            fontSize: '0.75rem',
            fontWeight: 600,
            padding: '0.35rem 0.75rem',
            cursor: validation.valid ? 'pointer' : 'default',
            opacity: loading ? 0.6 : 1,
          }}
        >
          {loading ? 'Reloading...' : 'Reload'}
        </button>

        {/* Validation status */}
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
          <div style={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            background: validation.valid ? '#44cc88' : '#ff4466',
          }} />
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            {validation.valid ? 'Valid' : `${validation.errors.length} error(s)`}
          </span>
          {lastReload && (
            <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginLeft: 8 }}>
              Last reload: {lastReload}
            </span>
          )}
        </div>
      </div>

      {/* Validation errors */}
      {validation.errors.length > 0 && (
        <div style={{
          background: '#331111',
          border: '1px solid #ff4466',
          borderRadius: 'var(--radius-sm)',
          padding: '0.5rem 0.75rem',
          fontSize: '0.75rem',
        }}>
          {validation.errors.map((err, i) => (
            <div key={i} style={{ color: '#ff8888', marginBottom: 2 }}>
              {'\u2022'} {err}
            </div>
          ))}
        </div>
      )}

      {/* YAML editor */}
      <textarea
        value={yaml}
        onChange={(e) => setYaml(e.target.value)}
        spellCheck={false}
        style={{
          flex: 1,
          minHeight: 300,
          background: 'var(--bg-surface)',
          border: '1px solid var(--text-muted)',
          borderRadius: 'var(--radius-sm)',
          padding: '0.75rem',
          color: 'var(--text-primary)',
          fontFamily: 'monospace',
          fontSize: '0.8rem',
          lineHeight: 1.5,
          resize: 'none',
          outline: 'none',
          tabSize: 2,
        }}
        onFocus={(e) => e.target.style.borderColor = 'var(--orb-listening)'}
        onBlur={(e) => e.target.style.borderColor = 'var(--text-muted)'}
      />
    </div>
  );
}
