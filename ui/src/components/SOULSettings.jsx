import { useState, useCallback, useEffect, useMemo } from 'react';
import { listen } from '@tauri-apps/api/event';
import { SettingsPage, SectionLabel } from './SettingsUI';

function extractName(yamlStr) {
  const m = yamlStr.match(/^name:\s*(.+)$/m);
  return m ? m[1].trim().replace(/^['"]|['"]$/g, '') : '';
}

export default function SOULSettings({ sendCommand, onBack }) {
  const [yaml, setYaml] = useState('');
  const [validation, setValidation] = useState({ valid: true, errors: [] });
  const [lastReload, setLastReload] = useState(null);
  const [loading, setLoading] = useState(false);

  const name = useMemo(() => extractName(yaml), [yaml]);

  const handleNameChange = useCallback((newName) => {
    setYaml(prev => {
      if (prev.match(/^name:\s*.+$/m)) {
        return prev.replace(/^name:\s*.+$/m, `name: ${newName}`);
      }
      return `name: ${newName}\n${prev}`;
    });
  }, []);

  useEffect(() => {
    const unlisten = listen('python_event', (event) => {
      const data = event.payload;
      if (data.event === 'soul_yaml') {
        setYaml(data.content || '');
      }
      if (data.event === 'soul_validation') {
        setValidation({ valid: data.valid ?? true, errors: data.errors || [] });
      }
      if (data.event === 'signal' && data.type === 'process_state_change') {
        setLoading(false);
      }
    });
    sendCommand?.({ cmd: 'get_soul_yaml' });
    return () => { unlisten.then(fn => fn()); };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleValidate = useCallback(() => {
    sendCommand?.({ cmd: 'validate_soul', content: yaml });
  }, [sendCommand, yaml]);

  const handleReload = useCallback(() => {
    setLoading(true);
    sendCommand?.({ cmd: 'reload_soul', content: yaml });
    setLastReload(new Date().toLocaleTimeString());
  }, [sendCommand, yaml]);

  return (
    <SettingsPage title="SOUL" onBack={onBack}>
      <SectionLabel first>Persona Configuration</SectionLabel>

      {/* Name field */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
        <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', minWidth: 40 }}>Name</label>
        <input
          type="text"
          value={name}
          onChange={(e) => handleNameChange(e.target.value)}
          placeholder="Assistant name"
          style={{
            flex: 1, background: 'var(--bg-surface)',
            border: '1px solid var(--text-muted)', borderRadius: 'var(--radius-sm)',
            padding: '0.4rem 0.6rem', color: 'var(--text-primary)',
            fontSize: '0.85rem', outline: 'none',
          }}
          onFocus={(e) => e.target.style.borderColor = 'var(--orb-listening)'}
          onBlur={(e) => e.target.style.borderColor = 'var(--text-muted)'}
        />
      </div>

      {/* Toolbar */}
      <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginBottom: '0.5rem' }}>
        <button
          onClick={handleValidate}
          style={{
            background: 'var(--bg-surface)', border: '1px solid var(--text-muted)',
            borderRadius: 'var(--radius-sm)', color: 'var(--text-secondary)',
            fontSize: '0.75rem', padding: '0.35rem 0.75rem', cursor: 'pointer',
          }}
        >
          Validate
        </button>
        <button
          onClick={handleReload}
          disabled={loading}
          style={{
            background: validation.valid ? '#44cc88' : 'var(--bg-surface)',
            border: 'none', borderRadius: 'var(--radius-sm)',
            color: validation.valid ? '#0a0a0f' : 'var(--text-muted)',
            fontSize: '0.75rem', fontWeight: 600,
            padding: '0.35rem 0.75rem',
            cursor: validation.valid ? 'pointer' : 'default',
            opacity: loading ? 0.6 : 1,
          }}
        >
          {loading ? 'Reloading...' : 'Reload'}
        </button>

        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
          <div style={{
            width: 8, height: 8, borderRadius: '50%',
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
          background: '#331111', border: '1px solid #ff4466',
          borderRadius: 'var(--radius-sm)', padding: '0.5rem 0.75rem',
          fontSize: '0.75rem', marginBottom: '0.5rem',
        }}>
          {validation.errors.map((err, i) => (
            <div key={i} style={{ color: '#ff8888', marginBottom: 2 }}>{'\u2022'} {err}</div>
          ))}
        </div>
      )}

      {/* YAML editor */}
      <textarea
        value={yaml}
        onChange={(e) => setYaml(e.target.value)}
        spellCheck={false}
        style={{
          minHeight: 300, background: 'var(--bg-surface)',
          border: '1px solid var(--text-muted)', borderRadius: 'var(--radius-sm)',
          padding: '0.75rem', color: 'var(--text-primary)',
          fontFamily: 'monospace', fontSize: '0.8rem', lineHeight: 1.5,
          resize: 'none', outline: 'none', tabSize: 2, width: '100%',
          boxSizing: 'border-box',
        }}
        onFocus={(e) => e.target.style.borderColor = 'var(--orb-listening)'}
        onBlur={(e) => e.target.style.borderColor = 'var(--text-muted)'}
      />
    </SettingsPage>
  );
}
