import { useState, useEffect, useCallback, useRef } from 'react';
import { listen } from '@tauri-apps/api/event';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { LogicalSize } from '@tauri-apps/api/dpi';
import { useSidecar } from './hooks/useSidecar';
import { useSignals } from './hooks/useSignals';
import Orb from './components/Orb';
import MicPill from './components/MicPill';
import StatusLine from './components/StatusLine';
import MoodGlow from './components/MoodGlow';
import Transcript from './components/Transcript';
import TextInput from './components/TextInput';
import Settings from './components/Settings';
import Wizard from './components/Wizard';
import DebugWindow from './components/DebugWindow';
import './styles/theme.css';

const COMPACT_SIZE = { width: 300, height: 400 };
const EXPANDED_SIZE = { width: 500, height: 700 };

function LanguageBadge({ language }) {
  const [visible, setVisible] = useState(false);
  const [display, setDisplay] = useState(language);

  useEffect(() => {
    if (!language) return;
    setDisplay(language);
    setVisible(true);
    const timer = setTimeout(() => setVisible(false), 3000);
    return () => clearTimeout(timer);
  }, [language]);

  if (!display) return null;

  return (
    <div
      role="status"
      aria-live="polite"
      aria-label={visible ? `Language: ${display}` : undefined}
      style={{
        position: 'absolute',
        top: '0.75rem',
        right: '0.75rem',
        fontSize: '0.7rem',
        color: 'var(--text-secondary)',
        background: 'var(--bg-surface)',
        padding: '0.125rem 0.5rem',
        borderRadius: 'var(--radius-sm)',
        opacity: visible ? 0.8 : 0,
        transition: 'opacity 0.5s ease',
        zIndex: 10,
      }}
    >
      {display}
    </div>
  );
}

function LastExchange({ lastUser, lastAssistant }) {
  const [hovered, setHovered] = useState(false);

  if (!lastUser && !lastAssistant) return null;

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        padding: 'var(--spacing-md)',
        background: 'var(--bg-secondary)',
        borderTop: '1px solid var(--bg-surface)',
        opacity: hovered ? 1 : 0,
        transition: 'opacity 0.3s ease',
        pointerEvents: hovered ? 'auto' : 'none',
        maxHeight: 120,
        overflow: 'auto',
      }}
    >
      {lastUser && (
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 4 }}>
          <span style={{ color: 'var(--text-secondary)' }}>You: </span>
          {lastUser.slice(0, 120)}
        </div>
      )}
      {lastAssistant && (
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          <span style={{ color: 'var(--text-secondary)' }}>AI: </span>
          {lastAssistant.slice(0, 120)}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const { state, events, sendCommand } = useSidecar();
  const signals = useSignals([
    'input_level', 'mood_change', 'language_detected',
    'audio.input_level', 'speech_start', 'speech_end', 'barge_in',
    'end_to_end_latency', 'vram_usage', 'tokens_per_sec',
    'state_change', 'filler_played',
    'request_start', 'first_token', 'complete',
    'synth_start', 'synth_end',
  ]);

  const [showWizard, setShowWizard] = useState(() => {
    try { return !localStorage.getItem('wizard_complete'); } catch { return true; }
  });
  const [expanded, setExpanded] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [micState, setMicState] = useState('listening');
  const [mood, setMood] = useState('neutral');
  const [language, setLanguage] = useState(null);
  const [lastUser, setLastUser] = useState('');
  const [lastAssistant, setLastAssistant] = useState('');
  const [energy, setEnergy] = useState(0);
  const [contextMenu, setContextMenu] = useState(null);
  const [turns, setTurns] = useState([]);
  const turnIdRef = useRef(0);

  // Process incoming signals
  useEffect(() => {
    if (signals.input_level) setEnergy(signals.input_level.level || 0);
    if (signals.mood_change) setMood(signals.mood_change.mood || 'neutral');
    if (signals.language_detected) setLanguage(signals.language_detected.language || null);
  }, [signals]);

  // Track turns from events
  useEffect(() => {
    if (events.length === 0) return;
    const last = events[events.length - 1];

    if (last.event === 'transcript') {
      if (last.role === 'user') {
        setLastUser(last.text || '');
        setTurns(prev => [...prev, {
          id: ++turnIdRef.current,
          role: 'user',
          text: last.text || '',
          source: last.source || 'voice',
        }]);
      }
      if (last.role === 'assistant') {
        setLastAssistant(last.text || '');
        setTurns(prev => [...prev, {
          id: ++turnIdRef.current,
          role: 'assistant',
          text: last.text || '',
          source: 'voice',
          interrupted: last.interrupted || false,
          spokenText: last.spoken_text || null,
        }]);
      }
    }
  }, [events]);

  // Window resize on mode toggle
  const toggleExpanded = useCallback(async () => {
    const next = !expanded;
    setExpanded(next);
    try {
      const win = getCurrentWindow();
      const size = next ? EXPANDED_SIZE : COMPACT_SIZE;
      await win.setSize(new LogicalSize(size.width, size.height));
    } catch (e) {
      console.error('Window resize failed:', e);
    }
  }, [expanded]);

  // Handle global shortcuts
  useEffect(() => {
    const unlisten = listen('shortcut', (event) => {
      const action = event.payload;
      if (action === 'mute_toggle') {
        setMicState(prev => prev === 'muted' ? 'listening' : 'muted');
        sendCommand({ cmd: 'mute_toggle' });
      }
      if (action === 'compact_toggle') {
        toggleExpanded();
      }
      if (action === 'debug_toggle') {
        setShowDebug(prev => !prev);
      }
    });
    return () => { unlisten.then(fn => fn()); };
  }, [sendCommand, toggleExpanded]);

  // Handle tray actions
  useEffect(() => {
    const unlisten = listen('tray_action', (event) => {
      const action = event.payload;
      if (action === 'new_conversation') {
        sendCommand({ cmd: 'new_conversation' });
        setLastUser('');
        setLastAssistant('');
        setTurns([]);
      }
      if (action === 'settings') {
        if (!expanded) toggleExpanded();
        setShowSettings(true);
      }
      if (action === 'debug') {
        setShowDebug(prev => !prev);
      }
    });
    return () => { unlisten.then(fn => fn()); };
  }, [sendCommand, expanded, toggleExpanded]);

  // Mic state change
  const handleMicChange = useCallback((newState) => {
    setMicState(newState);
    if (newState === 'muted') {
      sendCommand({ cmd: 'mute_toggle' });
    } else if (newState === 'text') {
      sendCommand({ cmd: 'set_mode', mode: 'TEXT_ONLY' });
    } else {
      sendCommand({ cmd: 'set_mode', mode: 'FULL' });
    }
  }, [sendCommand]);

  // Text submission
  const handleTextSubmit = useCallback((text) => {
    sendCommand({ cmd: 'text_input', text });
    setTurns(prev => [...prev, {
      id: ++turnIdRef.current,
      role: 'user',
      text,
      source: 'text',
    }]);
    setLastUser(text);
  }, [sendCommand]);

  // Right-click context menu
  const handleContextMenu = useCallback((e) => {
    e.preventDefault();
    setContextMenu({ x: e.clientX, y: e.clientY });
  }, []);

  const closeContextMenu = useCallback(() => setContextMenu(null), []);

  const handleMenuAction = useCallback((action) => {
    setContextMenu(null);
    switch (action) {
      case 'new_conversation':
        sendCommand({ cmd: 'new_conversation' });
        setLastUser('');
        setLastAssistant('');
        setTurns([]);
        break;
      case 'settings':
        if (!expanded) toggleExpanded();
        setShowSettings(true);
        break;
      case 'expand':
        toggleExpanded();
        break;
      case 'debug':
        setShowDebug(prev => !prev);
        break;
      case 'quit':
        sendCommand({ cmd: 'shutdown' });
        break;
    }
  }, [sendCommand, expanded, toggleExpanded]);

  // ---------- WIZARD ----------
  if (showWizard) {
    const handleWizardComplete = async (prefs) => {
      try { localStorage.setItem('wizard_complete', '1'); } catch {}
      setShowWizard(false);
      if (prefs?.languages) {
        sendCommand({ cmd: 'set_languages', languages: prefs.languages });
      }
      if (prefs?.persona) {
        sendCommand({ cmd: 'reload_soul', persona: prefs.persona });
      }
    };
    return (
      <Wizard
        onComplete={handleWizardComplete}
        sendCommand={sendCommand}
        energy={energy}
        componentStatus={{}}
      />
    );
  }

  // ---------- COMPACT MODE ----------
  if (!expanded) {
    return (
      <div
        data-tauri-drag-region
        onContextMenu={handleContextMenu}
        onClick={contextMenu ? closeContextMenu : undefined}
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 'var(--spacing-lg)',
          position: 'relative',
          background: 'var(--bg-primary)',
        }}
      >
        <LanguageBadge language={language} />
        <MoodGlow mood={mood} />

        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Orb state={state} mood={mood} energy={energy} size={180} />
        </div>

        <StatusLine state={state} />

        <div style={{ marginTop: 'var(--spacing-sm)', marginBottom: 'var(--spacing-sm)' }}>
          <MicPill micState={micState} onStateChange={handleMicChange} />
        </div>

        {/* Expand button */}
        <button
          onClick={toggleExpanded}
          aria-label="Expand to full view"
          style={{
            position: 'absolute',
            bottom: 8,
            right: 8,
            background: 'none',
            border: 'none',
            color: 'var(--text-muted)',
            fontSize: '1rem',
            cursor: 'pointer',
            padding: '4px',
            opacity: 0.5,
            transition: 'opacity 0.2s',
          }}
          onMouseEnter={e => e.target.style.opacity = 1}
          onMouseLeave={e => e.target.style.opacity = 0.5}
          title="Expand"
        >
          \u2922
        </button>

        <LastExchange lastUser={lastUser} lastAssistant={lastAssistant} />

        {contextMenu && (
          <ContextMenu
            x={contextMenu.x}
            y={contextMenu.y}
            expanded={expanded}
            onAction={handleMenuAction}
          />
        )}

        {showDebug && (
          <DebugWindow
            signals={signals}
            events={events}
            state={state}
            sendCommand={sendCommand}
            onClose={() => setShowDebug(false)}
          />
        )}
      </div>
    );
  }

  // ---------- EXPANDED MODE ----------
  return (
    <div
      onContextMenu={handleContextMenu}
      onClick={contextMenu ? closeContextMenu : undefined}
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--bg-primary)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Top bar — drag region + orb + controls */}
      <div
        data-tauri-drag-region
        style={{
          display: 'flex',
          alignItems: 'center',
          padding: '0.5rem 0.75rem',
          gap: '0.75rem',
          borderBottom: '1px solid var(--bg-surface)',
          background: 'var(--bg-secondary)',
          minHeight: 56,
        }}
      >
        <Orb state={state} mood={mood} energy={energy} size={40} />

        <div style={{ flex: 1 }}>
          <StatusLine state={state} />
        </div>

        <LanguageBadge language={language} />

        <MicPill micState={micState} onStateChange={handleMicChange} />

        {/* Collapse button */}
        <button
          onClick={toggleExpanded}
          aria-label="Collapse to compact view"
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--text-secondary)',
            fontSize: '1.1rem',
            cursor: 'pointer',
            padding: '0.25rem',
          }}
          title="Compact"
        >
          \u2923
        </button>
      </div>

      <MoodGlow mood={mood} />

      {/* Transcript */}
      <Transcript turns={turns} />

      {/* Text input */}
      <TextInput onSubmit={handleTextSubmit} disabled={state === 'DISABLED'} />

      {/* Settings overlay */}
      {showSettings && (
        <Settings
          onClose={() => setShowSettings(false)}
          sendCommand={sendCommand}
        />
      )}

      {contextMenu && (
        <ContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          expanded={expanded}
          onAction={handleMenuAction}
        />
      )}

      {showDebug && (
        <DebugWindow
          signals={signals}
          events={events}
          state={state}
          sendCommand={sendCommand}
          onClose={() => setShowDebug(false)}
        />
      )}
    </div>
  );
}

function ContextMenu({ x, y, expanded, onAction }) {
  const items = [
    { key: 'new_conversation', label: 'New Conversation' },
    { key: 'expand', label: expanded ? 'Compact Mode' : 'Expanded Mode' },
    { key: 'settings', label: 'Settings' },
    { key: 'debug', label: 'Debug' },
    { key: 'quit', label: 'Quit' },
  ];

  return (
    <div style={{
      position: 'fixed',
      top: y,
      left: x,
      background: 'var(--bg-surface)',
      border: '1px solid var(--text-muted)',
      borderRadius: 'var(--radius-sm)',
      padding: 'var(--spacing-xs) 0',
      zIndex: 1000,
      minWidth: 160,
      boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
    }}>
      {items.map(item => (
        <div
          key={item.key}
          onClick={() => onAction(item.key)}
          style={{
            padding: '6px 16px',
            cursor: 'pointer',
            fontSize: '0.85rem',
            color: item.key === 'quit' ? 'var(--mic-muted)' : 'var(--text-primary)',
            transition: 'background 0.15s',
          }}
          onMouseEnter={e => e.target.style.background = 'var(--bg-secondary)'}
          onMouseLeave={e => e.target.style.background = 'transparent'}
        >
          {item.label}
        </div>
      ))}
    </div>
  );
}
