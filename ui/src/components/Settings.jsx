import { useState, useCallback, useEffect, useRef } from 'react';
import { SectionLabel, Toggle, SettingRow } from './SettingsUI';
import AudioSettings from './AudioSettings';
import VADSettings from './VADSettings';
import LLMSettings from './LLMSettings';
import TTSSettings from './TTSSettings';
import SOULSettings from './SOULSettings';

const MODULES = [
  { key: 'audio', label: 'Audio', icon: '\uD83C\uDFA7', desc: 'Devices, input modes, mic & speaker test' },
  { key: 'vad',   label: 'VAD',   icon: '\uD83D\uDC42', desc: 'Voice activity detection, threshold, events' },
  { key: 'llm',   label: 'LLM',   icon: '\uD83E\uDDE0', desc: 'Token stream, stats, prompt inspector' },
  { key: 'tts',   label: 'TTS',   icon: '\uD83D\uDD0A', desc: 'Languages, voice cloning, test bench' },
  { key: 'soul',  label: 'SOUL',  icon: '\uD83D\uDC64', desc: 'Persona YAML editor, validate & reload' },
];

function ModuleButton({ icon, label, desc, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        background: 'var(--bg-surface)',
        border: '1px solid var(--bg-surface)',
        borderRadius: 'var(--radius-sm)',
        padding: '0.6rem 0.75rem',
        color: 'var(--text-primary)',
        cursor: 'pointer',
        width: '100%',
        transition: 'border-color 0.2s, background 0.2s',
        display: 'flex',
        alignItems: 'center',
        gap: '0.6rem',
        textAlign: 'left',
      }}
      onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--text-muted)'}
      onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--bg-surface)'}
    >
      <span style={{ fontSize: '1.2rem', width: 28, textAlign: 'center', flexShrink: 0 }}>{icon}</span>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: '0.85rem', fontWeight: 500 }}>{label}</div>
        <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: 1 }}>{desc}</div>
      </div>
      <span style={{ color: 'var(--text-muted)', fontSize: '1rem' }}>{'\u203A'}</span>
    </button>
  );
}

export default function Settings({ onClose, sendCommand, onNewConversation, currentSettings = {} }) {
  const panelRef = useRef(null);
  const closeRef = useRef(null);
  const [activeModule, setActiveModule] = useState(null);
  const [alwaysOnTop, setAlwaysOnTop] = useState(currentSettings.alwaysOnTop ?? true);
  const [debugMode, setDebugMode] = useState(currentSettings.debugMode ?? false);
  const [showConfirmClear, setShowConfirmClear] = useState(false);

  const handleNewConversation = useCallback(() => {
    onNewConversation?.();
  }, [onNewConversation]);

  const handleClearHistory = useCallback(() => {
    if (!showConfirmClear) {
      setShowConfirmClear(true);
      return;
    }
    onNewConversation?.();
    setShowConfirmClear(false);
  }, [showConfirmClear, onNewConversation]);

  // Focus trap + Escape
  useEffect(() => {
    closeRef.current?.focus();
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        if (activeModule) setActiveModule(null);
        else onClose();
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
  }, [onClose, activeModule]);

  // Render active module sub-page
  if (activeModule) {
    const backHandler = () => setActiveModule(null);
    switch (activeModule) {
      case 'audio': return <AudioSettings sendCommand={sendCommand} onBack={backHandler} currentSettings={currentSettings} />;
      case 'vad':   return <VADSettings sendCommand={sendCommand} onBack={backHandler} />;
      case 'llm':   return <LLMSettings sendCommand={sendCommand} onBack={backHandler} />;
      case 'tts':   return <TTSSettings sendCommand={sendCommand} onBack={backHandler} />;
      case 'soul':  return <SOULSettings sendCommand={sendCommand} onBack={backHandler} />;
      default:      return null;
    }
  }

  return (
    <div
      ref={panelRef}
      role="dialog"
      aria-label="Settings"
      aria-modal="true"
      style={{
        position: 'absolute', inset: 0,
        background: 'var(--bg-primary)',
        zIndex: 100,
        display: 'flex', flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '0.75rem', borderBottom: '1px solid var(--bg-surface)',
      }}>
        <span style={{ fontSize: '0.95rem', fontWeight: 600 }}>Settings</span>
        <button
          ref={closeRef}
          onClick={onClose}
          aria-label="Close settings"
          style={{
            background: 'none', border: 'none',
            color: 'var(--text-secondary)', fontSize: '1.2rem',
            cursor: 'pointer', padding: '0.25rem',
          }}
        >
          {'\u2715'}
        </button>
      </div>

      {/* Module menu + General settings */}
      <div style={{
        flex: 1, overflowY: 'auto', padding: '0.5rem 0.75rem',
        scrollbarWidth: 'thin', scrollbarColor: 'var(--bg-surface) transparent',
      }}>
        <SectionLabel first>Modules</SectionLabel>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
          {MODULES.map(m => (
            <ModuleButton
              key={m.key}
              icon={m.icon}
              label={m.label}
              desc={m.desc}
              onClick={() => setActiveModule(m.key)}
            />
          ))}
        </div>

        <SectionLabel>Window</SectionLabel>
        <SettingRow label="Always on top">
          <Toggle value={alwaysOnTop} onChange={setAlwaysOnTop} label="Always on top" />
        </SettingRow>
        <SettingRow label="Debug mode">
          <Toggle value={debugMode} onChange={setDebugMode} label="Debug mode" />
        </SettingRow>

        <SectionLabel>Conversation</SectionLabel>
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
