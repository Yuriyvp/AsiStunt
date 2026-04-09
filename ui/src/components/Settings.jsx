import { useState, useCallback, useEffect, useRef } from 'react';

function SettingRow({ label, children }) {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '0.5rem 0',
      borderBottom: '1px solid var(--bg-surface)',
    }}>
      <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{label}</span>
      {children}
    </div>
  );
}

function Toggle({ value, onChange, label }) {
  return (
    <button
      role="switch"
      aria-checked={value}
      aria-label={label}
      onClick={() => onChange(!value)}
      style={{
        width: 36,
        height: 20,
        borderRadius: 10,
        border: 'none',
        background: value ? 'var(--orb-listening)' : 'var(--text-muted)',
        position: 'relative',
        cursor: 'pointer',
        transition: 'background 0.2s',
        padding: 0,
      }}
    >
      <span style={{
        position: 'absolute',
        top: 2,
        left: value ? 18 : 2,
        width: 16,
        height: 16,
        borderRadius: '50%',
        background: 'var(--text-primary)',
        transition: 'left 0.2s',
      }} />
    </button>
  );
}

function Select({ value, options, onChange }) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        background: 'var(--bg-surface)',
        border: '1px solid var(--text-muted)',
        borderRadius: 'var(--radius-sm)',
        padding: '0.25rem 0.5rem',
        color: 'var(--text-primary)',
        fontSize: '0.8rem',
        cursor: 'pointer',
        outline: 'none',
      }}
    >
      {options.map(opt => (
        <option key={opt.value} value={opt.value}>{opt.label}</option>
      ))}
    </select>
  );
}

export default function Settings({ onClose, sendCommand, currentSettings = {} }) {
  const panelRef = useRef(null);
  const closeRef = useRef(null);
  const [alwaysOnTop, setAlwaysOnTop] = useState(currentSettings.alwaysOnTop ?? true);
  const [debugMode, setDebugMode] = useState(currentSettings.debugMode ?? false);
  const [lockLanguage, setLockLanguage] = useState(currentSettings.lockLanguage ?? false);
  const [voiceInputMode, setVoiceInputMode] = useState(currentSettings.voiceInputMode ?? 'continuous');
  const [responseMode, setResponseMode] = useState(currentSettings.responseMode ?? 'voice');
  const [language, setLanguage] = useState(currentSettings.language ?? 'hr');
  const [showConfirmClear, setShowConfirmClear] = useState(false);

  const handleNewConversation = useCallback(() => {
    sendCommand?.({ cmd: 'new_conversation' });
  }, [sendCommand]);

  const handleClearHistory = useCallback(() => {
    if (!showConfirmClear) {
      setShowConfirmClear(true);
      return;
    }
    sendCommand?.({ cmd: 'new_conversation' });
    setShowConfirmClear(false);
  }, [showConfirmClear, sendCommand]);

  const handleLanguageChange = useCallback((lang) => {
    setLanguage(lang);
    sendCommand?.({ cmd: 'set_language', language: lang });
  }, [sendCommand]);

  const handleVoiceInputChange = useCallback((mode) => {
    setVoiceInputMode(mode);
    sendCommand?.({ cmd: 'set_voice_input_mode', mode });
  }, [sendCommand]);

  const handleResponseModeChange = useCallback((mode) => {
    setResponseMode(mode);
    sendCommand?.({ cmd: 'set_response_mode', mode });
  }, [sendCommand]);

  // Focus trap: focus close button on mount, trap Tab within panel
  useEffect(() => {
    closeRef.current?.focus();

    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key === 'Tab' && panelRef.current) {
        const focusable = panelRef.current.querySelectorAll(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  return (
    <div
      ref={panelRef}
      role="dialog"
      aria-label="Settings"
      aria-modal="true"
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'var(--bg-primary)',
        zIndex: 100,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '0.75rem',
        borderBottom: '1px solid var(--bg-surface)',
      }}>
        <span style={{ fontSize: '0.95rem', fontWeight: 600 }}>Settings</span>
        <button
          ref={closeRef}
          onClick={onClose}
          aria-label="Close settings"
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--text-secondary)',
            fontSize: '1.2rem',
            cursor: 'pointer',
            padding: '0.25rem',
          }}
        >
          {'\u2715'}
        </button>
      </div>

      {/* Settings list */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '0.5rem 0.75rem',
        scrollbarWidth: 'thin',
        scrollbarColor: 'var(--bg-surface) transparent',
      }}>
        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem', marginTop: '0.25rem' }}>
          Language
        </div>
        <SettingRow label="Language">
          <Select
            value={language}
            onChange={handleLanguageChange}
            options={[
              { value: 'hr', label: 'Croatian' },
              { value: 'en', label: 'English' },
              { value: 'de', label: 'German' },
            ]}
          />
        </SettingRow>
        <SettingRow label="Lock language">
          <Toggle value={lockLanguage} onChange={setLockLanguage} label="Lock language detection" />
        </SettingRow>

        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem', marginTop: '0.75rem' }}>
          Input & Output
        </div>
        <SettingRow label="Voice input">
          <Select
            value={voiceInputMode}
            onChange={handleVoiceInputChange}
            options={[
              { value: 'continuous', label: 'Continuous' },
              { value: 'ptt', label: 'Push-to-talk' },
              { value: 'text_only', label: 'Text only' },
            ]}
          />
        </SettingRow>
        <SettingRow label="Response mode">
          <Select
            value={responseMode}
            onChange={handleResponseModeChange}
            options={[
              { value: 'voice', label: 'Voice' },
              { value: 'text', label: 'Text only' },
            ]}
          />
        </SettingRow>

        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem', marginTop: '0.75rem' }}>
          Window
        </div>
        <SettingRow label="Always on top">
          <Toggle value={alwaysOnTop} onChange={setAlwaysOnTop} label="Always on top" />
        </SettingRow>
        <SettingRow label="Debug mode">
          <Toggle value={debugMode} onChange={setDebugMode} label="Debug mode" />
        </SettingRow>

        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem', marginTop: '0.75rem' }}>
          Conversation
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', paddingTop: '0.5rem' }}>
          <button
            onClick={handleNewConversation}
            style={{
              background: 'var(--bg-surface)',
              border: '1px solid var(--text-muted)',
              borderRadius: 'var(--radius-sm)',
              padding: '0.5rem',
              color: 'var(--text-primary)',
              fontSize: '0.8rem',
              cursor: 'pointer',
              transition: 'background 0.2s',
            }}
          >
            New Conversation
          </button>
          <button
            onClick={handleClearHistory}
            style={{
              background: showConfirmClear ? 'var(--mic-muted)' : 'var(--bg-surface)',
              border: `1px solid ${showConfirmClear ? 'var(--mic-muted)' : 'var(--text-muted)'}`,
              borderRadius: 'var(--radius-sm)',
              padding: '0.5rem',
              color: showConfirmClear ? 'var(--bg-primary)' : 'var(--mic-muted)',
              fontSize: '0.8rem',
              fontWeight: showConfirmClear ? 600 : 400,
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
          >
            {showConfirmClear ? 'Confirm Clear History' : 'Clear History'}
          </button>
        </div>
      </div>
    </div>
  );
}
