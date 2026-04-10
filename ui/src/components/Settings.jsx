import { useState, useCallback, useEffect, useRef } from 'react';
import { listen } from '@tauri-apps/api/event';
import TTSSettings from './TTSSettings';

function SettingRow({ label, children, column }) {
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

function SectionLabel({ children, first }) {
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

function SmallButton({ onClick, children, color, disabled }) {
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

function LevelBar({ level }) {
  return (
    <div style={{
      flex: 1,
      height: 6,
      background: 'var(--bg-surface)',
      borderRadius: 3,
      overflow: 'hidden',
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

export default function Settings({ onClose, sendCommand, onNewConversation, currentSettings = {} }) {
  const panelRef = useRef(null);
  const closeRef = useRef(null);
  const [alwaysOnTop, setAlwaysOnTop] = useState(currentSettings.alwaysOnTop ?? true);
  const [debugMode, setDebugMode] = useState(currentSettings.debugMode ?? false);
  const [voiceInputMode, setVoiceInputMode] = useState(currentSettings.voiceInputMode ?? 'continuous');
  const [responseMode, setResponseMode] = useState(currentSettings.responseMode ?? 'voice');
  const [showConfirmClear, setShowConfirmClear] = useState(false);
  const [audioDevices, setAudioDevices] = useState({ inputs: [], outputs: [] });
  const [defaultInput, setDefaultInput] = useState(-1);
  const [defaultOutput, setDefaultOutput] = useState(-1);
  const [selectedMic, setSelectedMic] = useState(currentSettings.micDeviceId ?? -1);
  const [selectedSpeaker, setSelectedSpeaker] = useState(currentSettings.speakerDeviceId ?? -1);
  const [micTestState, setMicTestState] = useState('idle'); // idle | recording | done | error
  const [micLevel, setMicLevel] = useState(0);
  const [micLevels, setMicLevels] = useState([]);
  const [speakerTestState, setSpeakerTestState] = useState('idle'); // idle | playing | done | error
  const [showTTSSettings, setShowTTSSettings] = useState(false);

  // Listen for audio device list + test results from Python
  useEffect(() => {
    const unlisten = listen('python_event', (event) => {
      const data = event.payload;
      if (data.event !== 'signal') return;

      if (data.type === 'audio_devices') {
        setAudioDevices({ inputs: data.inputs || [], outputs: data.outputs || [] });
        setDefaultInput(data.default_input ?? -1);
        setDefaultOutput(data.default_output ?? -1);
        if (selectedMic === -1 && data.default_input >= 0) setSelectedMic(data.default_input);
        if (selectedSpeaker === -1 && data.default_output >= 0) setSelectedSpeaker(data.default_output);
      }

      if (data.type === 'mic_test') {
        if (data.status === 'recording') {
          setMicTestState('recording');
          setMicLevels([]);
          setMicLevel(0);
        } else if (data.status === 'level') {
          setMicLevel(data.level || 0);
          setMicLevels(prev => [...prev.slice(-50), data.level || 0]);
        } else if (data.status === 'done') {
          if (data.levels && data.levels.length > 0) {
            setMicLevels(data.levels);
          }
          setMicTestState('done');
          setTimeout(() => setMicTestState('idle'), 5000);
        } else if (data.status === 'error') {
          setMicTestState('error');
          setTimeout(() => setMicTestState('idle'), 3000);
        }
      }

      if (data.type === 'speaker_test') {
        if (data.status === 'playing') {
          setSpeakerTestState('playing');
        } else if (data.status === 'done') {
          setSpeakerTestState('done');
          setTimeout(() => setSpeakerTestState('idle'), 2000);
        } else if (data.status === 'error') {
          setSpeakerTestState('error');
          setTimeout(() => setSpeakerTestState('idle'), 3000);
        }
      }

    });
    sendCommand?.({ cmd: 'list_audio_devices' });
    return () => { unlisten.then(fn => fn()); };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

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

  const handleMicChange = useCallback((deviceId) => {
    const id = parseInt(deviceId, 10);
    setSelectedMic(id);
    sendCommand?.({ cmd: 'set_audio_device', type: 'input', device_id: id });
  }, [sendCommand]);

  const handleSpeakerChange = useCallback((deviceId) => {
    const id = parseInt(deviceId, 10);
    setSelectedSpeaker(id);
    sendCommand?.({ cmd: 'set_audio_device', type: 'output', device_id: id });
  }, [sendCommand]);

  const handleTestMic = useCallback(() => {
    setMicTestState('recording');
    setMicLevel(0);
    setMicLevels([]);
    sendCommand?.({ cmd: 'test_mic', device_id: selectedMic });
  }, [sendCommand, selectedMic]);

  const handleTestSpeaker = useCallback(() => {
    setSpeakerTestState('playing');
    sendCommand?.({ cmd: 'test_speaker', device_id: selectedSpeaker });
  }, [sendCommand, selectedSpeaker]);

  const handleVoiceInputChange = useCallback((mode) => {
    setVoiceInputMode(mode);
    sendCommand?.({ cmd: 'set_voice_input_mode', mode });
  }, [sendCommand]);

  const handleResponseModeChange = useCallback((mode) => {
    setResponseMode(mode);
    sendCommand?.({ cmd: 'set_response_mode', mode });
  }, [sendCommand]);

  // Focus trap
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
        <SectionLabel first>Audio Devices</SectionLabel>
        <SettingRow label="Microphone" column>
          <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
            <Select
              value={selectedMic}
              onChange={handleMicChange}
              options={audioDevices.inputs.length > 0
                ? audioDevices.inputs.map(d => ({
                    value: d.id,
                    label: d.id === defaultInput ? `${d.name} (default)` : d.name,
                  }))
                : [{ value: -1, label: 'Loading\u2026' }]
              }
            />
            <SmallButton
              onClick={handleTestMic}
              disabled={micTestState === 'recording'}
              color="var(--mic-listening)"
            >
              {micTestState === 'recording' ? 'Listening 3s\u2026' : micTestState === 'done' ? 'OK' : micTestState === 'error' ? 'Error' : 'Test Mic'}
            </SmallButton>
          </div>
          {(micTestState === 'recording' || micTestState === 'done') && (
            <div style={{
              marginTop: '0.35rem',
              background: 'var(--bg-surface)',
              borderRadius: 'var(--radius-sm)',
              padding: '0.35rem 0.5rem',
            }}>
              {/* Live level bar */}
              {micTestState === 'recording' && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span style={{
                    fontSize: '0.65rem', color: 'var(--mic-listening)',
                    animation: 'pulse 1.2s ease-in-out infinite',
                    minWidth: 10,
                  }}>
                    {'\u25CF'}
                  </span>
                  <LevelBar level={micLevel} />
                  <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', minWidth: 28, textAlign: 'right' }}>
                    {Math.round(micLevel * 100)}%
                  </span>
                </div>
              )}
              {/* Waveform summary after recording */}
              {micTestState === 'done' && micLevels.length > 0 && (
                <div>
                  <div style={{
                    display: 'flex',
                    alignItems: 'flex-end',
                    gap: 1,
                    height: 32,
                  }}>
                    {micLevels.map((l, i) => (
                      <div key={i} style={{
                        flex: 1,
                        height: `${Math.max(4, l * 100)}%`,
                        background: l > 0.8 ? 'var(--mic-muted)' : l > 0.4 ? 'var(--mic-ptt)' : 'var(--mic-listening)',
                        borderRadius: 1,
                        opacity: 0.85,
                      }} />
                    ))}
                  </div>
                  <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)', marginTop: '0.2rem', textAlign: 'center' }}>
                    {Math.max(...micLevels) > 0.1 ? 'Microphone is working' : 'No sound detected \u2014 check your microphone'}
                  </div>
                </div>
              )}
            </div>
          )}
          {micTestState === 'error' && (
            <div style={{ fontSize: '0.65rem', color: 'var(--mic-muted)', marginTop: '0.2rem' }}>
              Could not access microphone
            </div>
          )}
        </SettingRow>

        <SettingRow label="Speaker" column>
          <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
            <Select
              value={selectedSpeaker}
              onChange={handleSpeakerChange}
              options={audioDevices.outputs.length > 0
                ? audioDevices.outputs.map(d => ({
                    value: d.id,
                    label: d.id === defaultOutput ? `${d.name} (default)` : d.name,
                  }))
                : [{ value: -1, label: 'Loading\u2026' }]
              }
            />
            <SmallButton
              onClick={handleTestSpeaker}
              disabled={speakerTestState === 'playing'}
              color="var(--orb-speaking)"
            >
              {speakerTestState === 'playing' ? 'Playing\u2026' : speakerTestState === 'done' ? 'OK' : speakerTestState === 'error' ? 'Error' : 'Test Sound'}
            </SmallButton>
          </div>
          {speakerTestState === 'done' && (
            <div style={{ fontSize: '0.65rem', color: 'var(--mic-listening)', marginTop: '0.2rem' }}>
              Did you hear the chime?
            </div>
          )}
          {speakerTestState === 'error' && (
            <div style={{ fontSize: '0.65rem', color: 'var(--mic-muted)', marginTop: '0.2rem' }}>
              Could not play test sound
            </div>
          )}
        </SettingRow>

        <SectionLabel>Input & Output</SectionLabel>
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

        <SectionLabel>Window</SectionLabel>
        <SettingRow label="Always on top">
          <Toggle value={alwaysOnTop} onChange={setAlwaysOnTop} label="Always on top" />
        </SettingRow>
        <SettingRow label="Debug mode">
          <Toggle value={debugMode} onChange={setDebugMode} label="Debug mode" />
        </SettingRow>

        <SectionLabel>Voice & Language</SectionLabel>
        <div style={{ paddingTop: '0.35rem' }}>
          <button
            onClick={() => setShowTTSSettings(true)}
            style={{
              background: 'var(--bg-surface)',
              border: '1px solid var(--text-muted)',
              borderRadius: 'var(--radius-sm)',
              padding: '0.5rem',
              color: 'var(--text-primary)',
              fontSize: '0.8rem',
              cursor: 'pointer',
              width: '100%',
              transition: 'background 0.2s',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <span>TTS Settings</span>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>{'\u203A'}</span>
          </button>
        </div>

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

      {showTTSSettings && (
        <TTSSettings
          sendCommand={sendCommand}
          onBack={() => setShowTTSSettings(false)}
        />
      )}
    </div>
  );
}
