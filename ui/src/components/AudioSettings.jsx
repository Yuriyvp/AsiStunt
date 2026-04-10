import { useState, useCallback, useEffect } from 'react';
import { listen } from '@tauri-apps/api/event';
import { SettingsPage, SettingRow, Select, SmallButton, SectionLabel, LevelBar } from './SettingsUI';

export default function AudioSettings({ sendCommand, onBack, currentSettings = {} }) {
  const [audioDevices, setAudioDevices] = useState({ inputs: [], outputs: [] });
  const [defaultInput, setDefaultInput] = useState(-1);
  const [defaultOutput, setDefaultOutput] = useState(-1);
  const [selectedMic, setSelectedMic] = useState(currentSettings.micDeviceId ?? -1);
  const [selectedSpeaker, setSelectedSpeaker] = useState(currentSettings.speakerDeviceId ?? -1);
  const [micTestState, setMicTestState] = useState('idle');
  const [micLevel, setMicLevel] = useState(0);
  const [micLevels, setMicLevels] = useState([]);
  const [speakerTestState, setSpeakerTestState] = useState('idle');
  const [voiceInputMode, setVoiceInputMode] = useState(currentSettings.voiceInputMode ?? 'continuous');
  const [responseMode, setResponseMode] = useState(currentSettings.responseMode ?? 'voice');

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
          if (data.levels?.length > 0) setMicLevels(data.levels);
          setMicTestState('done');
          setTimeout(() => setMicTestState('idle'), 5000);
        } else if (data.status === 'error') {
          setMicTestState('error');
          setTimeout(() => setMicTestState('idle'), 3000);
        }
      }

      if (data.type === 'speaker_test') {
        if (data.status === 'playing') setSpeakerTestState('playing');
        else if (data.status === 'done') {
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

  return (
    <SettingsPage title="Audio" onBack={onBack}>
      <SectionLabel first>Devices</SectionLabel>

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
          <SmallButton onClick={handleTestMic} disabled={micTestState === 'recording'} color="var(--mic-listening)">
            {micTestState === 'recording' ? 'Listening 3s\u2026' : micTestState === 'done' ? 'OK' : micTestState === 'error' ? 'Error' : 'Test Mic'}
          </SmallButton>
        </div>
        {(micTestState === 'recording' || micTestState === 'done') && (
          <div style={{
            marginTop: '0.35rem', background: 'var(--bg-surface)',
            borderRadius: 'var(--radius-sm)', padding: '0.35rem 0.5rem',
          }}>
            {micTestState === 'recording' && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.65rem', color: 'var(--mic-listening)', animation: 'pulse 1.2s ease-in-out infinite', minWidth: 10 }}>{'\u25CF'}</span>
                <LevelBar level={micLevel} />
                <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', minWidth: 28, textAlign: 'right' }}>{Math.round(micLevel * 100)}%</span>
              </div>
            )}
            {micTestState === 'done' && micLevels.length > 0 && (
              <div>
                <div style={{ display: 'flex', alignItems: 'flex-end', gap: 1, height: 32 }}>
                  {micLevels.map((l, i) => (
                    <div key={i} style={{
                      flex: 1, height: `${Math.max(4, l * 100)}%`,
                      background: l > 0.8 ? 'var(--mic-muted)' : l > 0.4 ? 'var(--mic-ptt)' : 'var(--mic-listening)',
                      borderRadius: 1, opacity: 0.85,
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
          <div style={{ fontSize: '0.65rem', color: 'var(--mic-muted)', marginTop: '0.2rem' }}>Could not access microphone</div>
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
          <SmallButton onClick={handleTestSpeaker} disabled={speakerTestState === 'playing'} color="var(--orb-speaking)">
            {speakerTestState === 'playing' ? 'Playing\u2026' : speakerTestState === 'done' ? 'OK' : speakerTestState === 'error' ? 'Error' : 'Test Sound'}
          </SmallButton>
        </div>
        {speakerTestState === 'done' && (
          <div style={{ fontSize: '0.65rem', color: 'var(--mic-listening)', marginTop: '0.2rem' }}>Did you hear the chime?</div>
        )}
        {speakerTestState === 'error' && (
          <div style={{ fontSize: '0.65rem', color: 'var(--mic-muted)', marginTop: '0.2rem' }}>Could not play test sound</div>
        )}
      </SettingRow>

      <SectionLabel>Input & Output</SectionLabel>
      <SettingRow label="Voice input">
        <Select
          value={voiceInputMode}
          onChange={(mode) => { setVoiceInputMode(mode); sendCommand?.({ cmd: 'set_voice_input_mode', mode }); }}
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
          onChange={(mode) => { setResponseMode(mode); sendCommand?.({ cmd: 'set_response_mode', mode }); }}
          options={[
            { value: 'voice', label: 'Voice' },
            { value: 'text', label: 'Text only' },
          ]}
        />
      </SettingRow>
    </SettingsPage>
  );
}
