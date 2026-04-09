import { useState, useCallback } from 'react';
import Dashboard from '../pages/Dashboard';
import VADPage from '../pages/VADPage';
import LLMPage from '../pages/LLMPage';
import TTSPage from '../pages/TTSPage';
import AudioPipelinePage from '../pages/AudioPipelinePage';
import SOULPage from '../pages/SOULPage';
import EventLogPage from '../pages/EventLogPage';

const TABS = [
  { key: 'dashboard', label: 'Dashboard' },
  { key: 'vad', label: 'VAD' },
  { key: 'llm', label: 'LLM' },
  { key: 'tts', label: 'TTS' },
  { key: 'pipeline', label: 'Pipeline' },
  { key: 'soul', label: 'SOUL' },
  { key: 'events', label: 'Events' },
];

export default function DebugWindow({ signals, events, state, sendCommand, onClose }) {
  const [activeTab, setActiveTab] = useState('dashboard');

  const renderPage = useCallback(() => {
    switch (activeTab) {
      case 'dashboard': return <Dashboard signals={signals} state={state} events={events} />;
      case 'vad': return <VADPage signals={signals} />;
      case 'llm': return <LLMPage signals={signals} events={events} sendCommand={sendCommand} />;
      case 'tts': return <TTSPage signals={signals} events={events} sendCommand={sendCommand} />;
      case 'pipeline': return <AudioPipelinePage signals={signals} state={state} />;
      case 'soul': return <SOULPage sendCommand={sendCommand} events={events} />;
      case 'events': return <EventLogPage events={events} />;
      default: return null;
    }
  }, [activeTab, signals, events, state, sendCommand]);

  return (
    <div style={{
      position: 'absolute',
      inset: 0,
      background: 'var(--bg-primary)',
      display: 'flex',
      flexDirection: 'column',
      zIndex: 500,
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        padding: '0.4rem 0.75rem',
        background: 'var(--bg-secondary)',
        borderBottom: '1px solid var(--bg-surface)',
        gap: '0.5rem',
        minHeight: 40,
      }}>
        <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginRight: 'auto' }}>
          Debug
        </span>

        {/* Tabs */}
        <div style={{ display: 'flex', gap: 2 }}>
          {TABS.map(tab => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              style={{
                background: activeTab === tab.key ? 'var(--bg-surface)' : 'transparent',
                border: 'none',
                borderRadius: 'var(--radius-sm)',
                color: activeTab === tab.key ? 'var(--text-primary)' : 'var(--text-muted)',
                fontSize: '0.7rem',
                padding: '0.3rem 0.6rem',
                cursor: 'pointer',
                transition: 'all 0.15s',
                fontWeight: activeTab === tab.key ? 600 : 400,
              }}
              onMouseEnter={e => { if (activeTab !== tab.key) e.target.style.color = 'var(--text-secondary)'; }}
              onMouseLeave={e => { if (activeTab !== tab.key) e.target.style.color = 'var(--text-muted)'; }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Close */}
        <button
          onClick={onClose}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--text-muted)',
            fontSize: '1.1rem',
            cursor: 'pointer',
            padding: '0 0.25rem',
            marginLeft: '0.5rem',
          }}
          aria-label="Close debug window"
        >
          {'\u2715'}
        </button>
      </div>

      {/* Page content */}
      <div style={{ flex: 1, overflow: 'auto', padding: '0.75rem' }}>
        {renderPage()}
      </div>
    </div>
  );
}
