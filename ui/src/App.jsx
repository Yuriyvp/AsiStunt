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

const appWindow = getCurrentWindow();

function WindowControls() {
  const [hovered, setHovered] = useState(null);

  const handleMinimize = (e) => {
    e.stopPropagation();
    appWindow.minimize();
  };
  const handleMaximize = async (e) => {
    e.stopPropagation();
    if (await appWindow.isMaximized()) appWindow.unmaximize();
    else appWindow.maximize();
  };
  const handleClose = (e) => {
    e.stopPropagation();
    appWindow.close();
  };

  const dots = [
    { action: handleClose, color: '#ed6a5e', hoverColor: '#f1827a', label: 'Close', key: 'close' },
    { action: handleMinimize, color: '#f5bf4f', hoverColor: '#f7cc6e', label: 'Minimize', key: 'min' },
    { action: handleMaximize, color: '#56c453', hoverColor: '#74d172', label: 'Maximize', key: 'max' },
  ];

  return (
    <div
      style={{ display: 'flex', gap: 7, alignItems: 'center', padding: '0 2px' }}
      onMouseLeave={() => setHovered(null)}
    >
      {dots.map(d => (
        <button
          key={d.key}
          onClick={d.action}
          onMouseEnter={() => setHovered(d.key)}
          aria-label={d.label}
          title={d.label}
          style={{
            width: 13,
            height: 13,
            borderRadius: '50%',
            border: '0.5px solid rgba(0,0,0,0.15)',
            cursor: 'pointer',
            padding: 0,
            background: hovered === d.key ? d.hoverColor : d.color,
            transition: 'background 0.1s, transform 0.1s',
            transform: hovered === d.key ? 'scale(1.15)' : 'scale(1)',
            outline: 'none',
          }}
        />
      ))}
    </div>
  );
}

function TitleBar({ children, onDoubleClick }) {
  const handleMouseDown = async (e) => {
    // Only drag on left-click on the bar itself or drag-region children
    if (e.button !== 0) return;
    const el = e.target;
    const isInteractive = el.closest('button, input, select, a, [role="button"]');
    if (isInteractive) return;

    // If maximized, unmaximize first so user can reposition
    try {
      if (await appWindow.isMaximized()) {
        await appWindow.unmaximize();
      }
    } catch {}

    appWindow.startDragging();
  };

  return (
    <div
      onMouseDown={handleMouseDown}
      onDoubleClick={onDoubleClick}
      style={{
        display: 'flex',
        alignItems: 'center',
        padding: '6px 10px',
        gap: '0.6rem',
        background: 'var(--bg-secondary)',
        borderBottom: '1px solid var(--bg-surface)',
        minHeight: 38,
        cursor: 'grab',
      }}
    >
      {children}
    </div>
  );
}

const titleBtnStyle = {
  background: 'none',
  border: 'none',
  color: 'var(--text-muted)',
  fontSize: '0.85rem',
  cursor: 'pointer',
  padding: '2px 5px',
  opacity: 0.6,
  transition: 'opacity 0.15s, color 0.15s',
  lineHeight: 1,
};

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

const COMPONENTS = [
  { key: 'llm', label: 'LLM', icon: '\uD83E\uDDE0' },
  { key: 'tts', label: 'TTS', icon: '\uD83D\uDD0A' },
  { key: 'asr', label: 'ASR', icon: '\uD83C\uDF99' },
  { key: 'vad', label: 'VAD', icon: '\uD83D\uDC42' },
];

const STATUS_COLORS = {
  idle: 'var(--text-muted)',
  starting: '#f5bf4f',
  stopping: '#f5bf4f',
  ready: '#56c453',
  restarting: '#f5bf4f',
  restarting_optimized: '#cc8844',
  error: '#ed6a5e',
  failed: '#ed6a5e',
};

const STATUS_LABELS = {
  idle: 'Not loaded',
  starting: 'Loading…',
  stopping: 'Stopping…',
  ready: 'Ready',
  restarting: 'Restarting…',
  restarting_optimized: 'Optimizing…',
  error: 'Error',
  failed: 'Failed',
};

function ComponentStatus({ componentStates, sendCommand, compact = false }) {
  const allReady = COMPONENTS.every(c => componentStates[c.key] === 'ready');

  if (compact && allReady) return null;

  const actionBtnStyle = {
    background: 'none',
    border: '1px solid var(--text-muted)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-secondary)',
    fontSize: '0.6rem',
    padding: '1px 6px',
    cursor: 'pointer',
    lineHeight: '1.4',
    transition: 'all 0.15s',
    textTransform: 'uppercase',
    letterSpacing: '0.03em',
  };

  return (
    <div style={{
      display: 'flex',
      gap: compact ? '0.5rem' : '0.5rem',
      alignItems: 'center',
      justifyContent: 'center',
      padding: compact ? '0.25rem 0.5rem' : '0.4rem 0.75rem',
      flexWrap: 'wrap',
      borderBottom: compact ? 'none' : '1px solid var(--bg-surface)',
      background: compact ? 'transparent' : 'var(--bg-secondary)',
    }}>
      {COMPONENTS.map(comp => {
        const s = componentStates[comp.key] || 'idle';
        const color = STATUS_COLORS[s] || 'var(--text-muted)';
        const isLoading = s === 'starting' || s === 'restarting' || s === 'restarting_optimized' || s === 'stopping';
        const isRunning = s === 'ready';
        const isStopped = s === 'idle' || s === 'error' || s === 'failed';

        return (
          <div
            key={comp.key}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.25rem',
              fontSize: compact ? '0.7rem' : '0.75rem',
              color: 'var(--text-secondary)',
              background: compact ? 'none' : 'var(--bg-primary)',
              borderRadius: 'var(--radius-sm)',
              padding: compact ? '0' : '3px 8px',
            }}
          >
            <span style={{ fontSize: compact ? '0.75rem' : '0.85rem' }}>{comp.icon}</span>
            <span style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: color,
              display: 'inline-block',
              flexShrink: 0,
              animation: isLoading ? 'pulse 1.2s ease-in-out infinite' : 'none',
              boxShadow: isRunning ? `0 0 4px ${color}` : 'none',
            }} />
            {!compact && (
              <>
                <span style={{ color, fontSize: '0.65rem', minWidth: 48 }}>
                  {STATUS_LABELS[s] || s}
                </span>
                {isStopped && (
                  <button
                    onClick={() => sendCommand({ cmd: 'start_component', component: comp.key })}
                    style={{
                      ...actionBtnStyle,
                      borderColor: 'var(--mic-listening)',
                      color: 'var(--mic-listening)',
                    }}
                    onMouseEnter={e => { e.target.style.background = 'var(--mic-listening)'; e.target.style.color = 'var(--bg-primary)'; }}
                    onMouseLeave={e => { e.target.style.background = 'none'; e.target.style.color = 'var(--mic-listening)'; }}
                    title={`Load ${comp.label}`}
                  >
                    Load
                  </button>
                )}
                {isRunning && (
                  <button
                    onClick={() => sendCommand({ cmd: 'stop_component', component: comp.key })}
                    style={{
                      ...actionBtnStyle,
                      borderColor: 'var(--mic-muted)',
                      color: 'var(--mic-muted)',
                    }}
                    onMouseEnter={e => { e.target.style.background = 'var(--mic-muted)'; e.target.style.color = 'var(--bg-primary)'; }}
                    onMouseLeave={e => { e.target.style.background = 'none'; e.target.style.color = 'var(--mic-muted)'; }}
                    title={`Stop ${comp.label}`}
                  >
                    Stop
                  </button>
                )}
                {isLoading && (
                  <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>…</span>
                )}
              </>
            )}
          </div>
        );
      })}
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
  const [componentStates, setComponentStates] = useState({
    llm: 'idle', tts: 'idle', asr: 'idle', vad: 'idle',
  });
  const turnIdRef = useRef(0);

  // Direct listener for component state changes — bypasses React batching
  // Each signal is processed individually via functional updater
  useEffect(() => {
    const unlisten = listen('python_event', (event) => {
      const data = event.payload;
      if (data.event === 'signal' && data.type === 'process_state_change') {
        const { component, state } = data;
        if (component && state && component !== 'system') {
          setComponentStates(prev => ({ ...prev, [component]: state }));
        }
      }
    });
    return () => { unlisten.then(fn => fn()); };
  }, []);

  // Request component status on mount (catches up after startup race)
  useEffect(() => {
    const timer = setTimeout(() => {
      sendCommand({ cmd: 'get_status' });
    }, 1000);
    // Also poll periodically in case sidecar was slow to start
    const interval = setInterval(() => {
      const allIdle = Object.values(componentStates).every(s => s === 'idle');
      if (allIdle) sendCommand({ cmd: 'get_status' });
    }, 3000);
    return () => { clearTimeout(timer); clearInterval(interval); };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

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
        onContextMenu={handleContextMenu}
        onClick={contextMenu ? closeContextMenu : undefined}
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
          background: 'var(--bg-primary)',
        }}
      >
        {/* Draggable title bar with window controls */}
        <TitleBar>
          <WindowControls />
          <div style={{ flex: 1, minHeight: 24 }} />
          <button
            onClick={() => setShowDebug(prev => !prev)}
            aria-label="Debug dashboard"
            title="Debug"
            style={titleBtnStyle}
            onMouseEnter={e => { e.target.style.opacity = 1; e.target.style.color = 'var(--orb-processing)'; }}
            onMouseLeave={e => { e.target.style.opacity = 0.6; e.target.style.color = 'var(--text-muted)'; }}
          >
            {'\u2699'}
          </button>
          <button
            onClick={() => { if (!expanded) toggleExpanded(); setShowSettings(true); }}
            aria-label="Settings"
            title="Settings"
            style={titleBtnStyle}
            onMouseEnter={e => { e.target.style.opacity = 1; e.target.style.color = 'var(--text-primary)'; }}
            onMouseLeave={e => { e.target.style.opacity = 0.6; e.target.style.color = 'var(--text-muted)'; }}
          >
            {'\u2630'}
          </button>
          <button
            onClick={toggleExpanded}
            aria-label="Expand to full view"
            title="Expand"
            style={titleBtnStyle}
            onMouseEnter={e => { e.target.style.opacity = 1; }}
            onMouseLeave={e => { e.target.style.opacity = 0.6; }}
          >
            {'\u2922'}
          </button>
        </TitleBar>

        <LanguageBadge language={language} />
        <MoodGlow mood={mood} />

        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 'var(--spacing-md)',
        }}>
          <Orb state={state} mood={mood} energy={energy} size={180} />
        </div>

        <ComponentStatus componentStates={componentStates} sendCommand={sendCommand} compact />
        <StatusLine state={state} />

        <div style={{
          padding: '0 var(--spacing-md) var(--spacing-md)',
          display: 'flex',
          justifyContent: 'center',
        }}>
          <MicPill micState={micState} onStateChange={handleMicChange} />
        </div>

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
      {/* Title bar — draggable with window controls */}
      <TitleBar onDoubleClick={async () => {
        if (await appWindow.isMaximized()) appWindow.unmaximize();
        else appWindow.maximize();
      }}>
        <WindowControls />

        <div style={{ display: 'flex', alignItems: 'center', flex: 1, gap: '0.6rem' }}>
          <Orb state={state} mood={mood} energy={energy} size={32} />
          <div style={{ flex: 1 }}>
            <StatusLine state={state} />
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <LanguageBadge language={language} />
          <MicPill micState={micState} onStateChange={handleMicChange} />
          <button
            onClick={() => setShowDebug(prev => !prev)}
            aria-label="Debug dashboard"
            title="Debug"
            style={titleBtnStyle}
            onMouseEnter={e => { e.target.style.opacity = 1; e.target.style.color = 'var(--orb-processing)'; }}
            onMouseLeave={e => { e.target.style.opacity = 0.6; e.target.style.color = 'var(--text-muted)'; }}
          >
            {'\u2699'}
          </button>
          <button
            onClick={() => setShowSettings(true)}
            aria-label="Settings"
            title="Settings"
            style={titleBtnStyle}
            onMouseEnter={e => { e.target.style.opacity = 1; e.target.style.color = 'var(--text-primary)'; }}
            onMouseLeave={e => { e.target.style.opacity = 0.6; e.target.style.color = 'var(--text-muted)'; }}
          >
            {'\u2630'}
          </button>
          <button
            onClick={toggleExpanded}
            aria-label="Collapse to compact view"
            title="Compact"
            style={titleBtnStyle}
            onMouseEnter={e => { e.target.style.opacity = 1; }}
            onMouseLeave={e => { e.target.style.opacity = 0.6; }}
          >
            {'\u2923'}
          </button>
        </div>
      </TitleBar>

      <ComponentStatus componentStates={componentStates} sendCommand={sendCommand} />
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
