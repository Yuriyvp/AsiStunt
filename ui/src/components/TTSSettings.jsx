import { useState, useCallback, useEffect } from 'react';
import { listen } from '@tauri-apps/api/event';
import { open } from '@tauri-apps/plugin-dialog';
import { SettingsPage, SectionLabel, SmallButton } from './SettingsUI';

const ALL_LANGUAGES = {
  en: 'English', hr: 'Croatian', de: 'German', fr: 'French', es: 'Spanish',
  it: 'Italian', pt: 'Portuguese', nl: 'Dutch', pl: 'Polish', cs: 'Czech',
  ru: 'Russian', uk: 'Ukrainian', ja: 'Japanese', ko: 'Korean', zh: 'Chinese',
  hi: 'Hindi', tr: 'Turkish', sv: 'Swedish', da: 'Danish', fi: 'Finnish',
  no: 'Norwegian', el: 'Greek', he: 'Hebrew', hu: 'Hungarian', ro: 'Romanian',
  bg: 'Bulgarian', sr: 'Serbian', bs: 'Bosnian', sk: 'Slovak', sl: 'Slovenian',
  lt: 'Lithuanian', lv: 'Latvian', et: 'Estonian', ga: 'Irish', cy: 'Welsh',
  ca: 'Catalan', eu: 'Basque', gl: 'Galician', af: 'Afrikaans',
  id: 'Indonesian', ms: 'Malay', vi: 'Vietnamese', th: 'Thai', fil: 'Filipino',
  sw: 'Swahili',
};

function LanguagePicker({ selected, onChange, max }) {
  const [search, setSearch] = useState('');
  const filtered = Object.entries(ALL_LANGUAGES).filter(([id, name]) =>
    name.toLowerCase().includes(search.toLowerCase()) || id.includes(search.toLowerCase())
  );

  return (
    <div>
      <input
        type="text"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder="Search languages..."
        style={{
          background: 'var(--bg-surface)', border: '1px solid var(--text-muted)',
          borderRadius: 'var(--radius-sm)', padding: '0.3rem 0.5rem',
          color: 'var(--text-primary)', fontSize: '0.75rem', outline: 'none',
          width: '100%', marginBottom: '0.4rem',
        }}
      />
      <div style={{
        display: 'flex', flexWrap: 'wrap', gap: 3,
        maxHeight: 120, overflowY: 'auto',
        scrollbarWidth: 'thin', scrollbarColor: 'var(--bg-surface) transparent',
      }}>
        {filtered.map(([id, name]) => {
          const active = selected.includes(id);
          const atMax = selected.length >= max && !active;
          return (
            <button
              key={id}
              onClick={() => {
                if (active) onChange(selected.filter(s => s !== id));
                else if (!atMax) onChange([...selected, id]);
              }}
              disabled={atMax}
              style={{
                background: active ? 'var(--orb-speaking)' : 'var(--bg-surface)',
                border: `1px solid ${active ? 'var(--orb-speaking)' : 'var(--text-muted)'}`,
                borderRadius: 'var(--radius-pill)',
                color: active ? 'var(--bg-primary)' : atMax ? 'var(--text-muted)' : 'var(--text-secondary)',
                fontSize: '0.6rem', padding: '2px 8px',
                cursor: atMax ? 'not-allowed' : 'pointer',
                transition: 'all 0.15s',
                fontWeight: active ? 600 : 400,
                opacity: atMax ? 0.4 : 1,
              }}
            >
              {name}
            </button>
          );
        })}
      </div>
      <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
        {selected.length}/{max} selected
      </div>
    </div>
  );
}

function VoiceCloneRow({ lang, config, cloneState, onClone, onTest, testState }) {
  const [refAudio, setRefAudio] = useState(config.reference_audio || '');
  const name = ALL_LANGUAGES[lang] || lang;
  const hasProfile = config.has_profile;

  const handleBrowse = useCallback(async () => {
    const file = await open({
      multiple: false,
      filters: [{ name: 'Audio', extensions: ['wav', 'mp3', 'flac', 'ogg', 'opus', 'm4a', 'aac', 'wma'] }],
    });
    if (file) setRefAudio(file);
  }, []);

  return (
    <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', padding: '0.5rem 0.6rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.3rem' }}>
        <span style={{ fontSize: '0.8rem', fontWeight: 500 }}>{name}</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
          <span style={{
            width: 7, height: 7, borderRadius: '50%',
            background: hasProfile ? '#56c453' : '#ed6a5e',
            display: 'inline-block',
            boxShadow: hasProfile ? '0 0 4px #56c453' : 'none',
          }} />
          <span style={{ fontSize: '0.6rem', color: hasProfile ? '#56c453' : 'var(--text-muted)' }}>
            {hasProfile ? 'Cloned' : 'No voice'}
          </span>
        </div>
      </div>
      <div style={{ display: 'flex', gap: '0.3rem', alignItems: 'center' }}>
        <input
          type="text"
          value={refAudio}
          onChange={(e) => setRefAudio(e.target.value)}
          placeholder="/path/to/reference.wav"
          style={{
            flex: 1, background: 'var(--bg-secondary)',
            border: '1px solid var(--text-muted)', borderRadius: 'var(--radius-sm)',
            padding: '0.25rem 0.4rem', color: 'var(--text-primary)',
            fontSize: '0.7rem', outline: 'none',
          }}
        />
        <SmallButton onClick={handleBrowse} color="var(--text-secondary)">Browse</SmallButton>
        <SmallButton
          onClick={() => onClone(lang, refAudio)}
          disabled={!refAudio || cloneState === 'cloning'}
          color="var(--orb-processing)"
        >
          {cloneState === 'cloning' ? 'Cloning...' : 'Clone'}
        </SmallButton>
      </div>
      {cloneState === 'error' && (
        <div style={{ fontSize: '0.6rem', color: 'var(--mic-muted)', marginTop: '0.2rem' }}>Clone failed</div>
      )}
      {cloneState === 'complete' && (
        <div style={{ fontSize: '0.6rem', color: 'var(--mic-listening)', marginTop: '0.2rem' }}>Clone complete</div>
      )}
      {hasProfile && (
        <div style={{ display: 'flex', gap: '0.3rem', marginTop: '0.3rem' }}>
          <SmallButton onClick={() => onTest(lang)} disabled={testState === 'playing'} color="var(--mic-listening)">
            {testState === 'playing' ? 'Playing...' : 'Test Voice'}
          </SmallButton>
        </div>
      )}
    </div>
  );
}

export default function TTSSettings({ sendCommand, onBack }) {
  const [selectedLangs, setSelectedLangs] = useState([]);
  const [langConfigs, setLangConfigs] = useState([]);
  const [loadedLangs, setLoadedLangs] = useState([]);
  const [defaultLang, setDefaultLang] = useState('en');
  const [cloneStates, setCloneStates] = useState({});
  const [testStates, setTestStates] = useState({});
  const [testText, setTestText] = useState('');
  const [testLang, setTestLang] = useState('');
  const [synthHistory, setSynthHistory] = useState([]);
  const [isSynthesizing, setIsSynthesizing] = useState(false);

  useEffect(() => {
    const unlisten = listen('python_event', (event) => {
      const data = event.payload;
      if (data.event !== 'signal') return;

      if (data.type === 'tts_settings') {
        const langs = data.languages || [];
        setLangConfigs(langs);
        setSelectedLangs(langs.map(l => l.id));
        setLoadedLangs(data.loaded_languages || []);
        setDefaultLang(data.default_language || 'en');
        if (!testLang && langs.length > 0) setTestLang(langs[0].id);
      }

      if (data.type === 'voice_clone_progress') {
        const lang = data.lang;
        if (lang) {
          if (data.status === 'complete') {
            setCloneStates(prev => ({ ...prev, [lang]: 'complete' }));
            setTimeout(() => setCloneStates(prev => ({ ...prev, [lang]: 'idle' })), 3000);
            sendCommand?.({ cmd: 'get_tts_settings' });
          } else if (data.status === 'error') {
            setCloneStates(prev => ({ ...prev, [lang]: 'error' }));
            setTimeout(() => setCloneStates(prev => ({ ...prev, [lang]: 'idle' })), 5000);
          } else {
            setCloneStates(prev => ({ ...prev, [lang]: 'cloning' }));
          }
        }
      }

      if (data.type === 'synth_start') setIsSynthesizing(true);
      if (data.type === 'synth_end') {
        setIsSynthesizing(false);
        setSynthHistory(prev => [...prev.slice(-30), {
          text: data.text || '', lang: data.lang || '',
          rtf: data.rtf || 0, duration_ms: data.duration_ms || 0,
          error: data.error || null, time: Date.now(),
        }]);
      }
    });
    sendCommand?.({ cmd: 'get_tts_settings' });
    return () => { unlisten.then(fn => fn()); };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleLangChange = useCallback((newLangs) => {
    setSelectedLangs(newLangs);
    sendCommand?.({ cmd: 'update_tts_languages', languages: newLangs });
  }, [sendCommand]);

  const handleDefaultChange = useCallback((lang) => {
    setDefaultLang(lang);
    sendCommand?.({ cmd: 'set_default_language', lang });
  }, [sendCommand]);

  const handleClone = useCallback((lang, refAudio) => {
    setCloneStates(prev => ({ ...prev, [lang]: 'cloning' }));
    sendCommand?.({ cmd: 'clone_voice_for_lang', lang, reference_audio: refAudio });
  }, [sendCommand]);

  const handleTestVoice = useCallback((lang) => {
    const testTexts = {
      en: 'The quick brown fox jumps over the lazy dog. Every morning I wake up and look out the window at the beautiful sunrise.',
      hr: 'Brza smeđa lisica preskače lijenog psa. Svako jutro se probudim i gledam kroz prozor u prekrasan izlazak sunca.',
      ru: 'Быстрая коричневая лиса перепрыгивает через ленивую собаку. Каждое утро я просыпаюсь и смотрю в окно на прекрасный рассвет.',
    };
    const text = testTexts[lang] || testTexts.en;
    setTestStates(prev => ({ ...prev, [lang]: 'playing' }));
    sendCommand?.({ cmd: 'tts_test', text, lang });
    setTimeout(() => setTestStates(prev => ({ ...prev, [lang]: 'idle' })), 8000);
  }, [sendCommand]);

  const handleTestBench = useCallback(() => {
    if (!testText.trim()) return;
    sendCommand?.({ cmd: 'tts_test', text: testText, lang: testLang || null });
    setTestText('');
  }, [sendCommand, testText, testLang]);

  return (
    <SettingsPage title="TTS" onBack={onBack}>
      {/* Language Selection */}
      <SectionLabel first>Languages (max 3)</SectionLabel>
      <LanguagePicker selected={selectedLangs} onChange={handleLangChange} max={3} />

      {/* Default Language */}
      {selectedLangs.length > 0 && (
        <>
          <SectionLabel>Default Language</SectionLabel>
          <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
            {selectedLangs.map(id => (
              <button
                key={id}
                onClick={() => handleDefaultChange(id)}
                style={{
                  background: defaultLang === id ? 'var(--orb-listening)' : 'var(--bg-surface)',
                  border: `1px solid ${defaultLang === id ? 'var(--orb-listening)' : 'var(--text-muted)'}`,
                  borderRadius: 'var(--radius-pill)',
                  color: defaultLang === id ? 'var(--bg-primary)' : 'var(--text-secondary)',
                  fontSize: '0.7rem', padding: '3px 10px', cursor: 'pointer',
                  fontWeight: defaultLang === id ? 600 : 400,
                }}
              >
                {ALL_LANGUAGES[id] || id}
              </button>
            ))}
          </div>
        </>
      )}

      {/* Voice Cloning per Language */}
      {selectedLangs.length > 0 && (
        <>
          <SectionLabel>Voice Cloning</SectionLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
            {selectedLangs.map(id => {
              const config = langConfigs.find(l => l.id === id) || { id, reference_audio: null, has_profile: false };
              return (
                <VoiceCloneRow
                  key={id} lang={id} config={config}
                  cloneState={cloneStates[id] || 'idle'}
                  onClone={handleClone} onTest={handleTestVoice}
                  testState={testStates[id] || 'idle'}
                />
              );
            })}
          </div>
        </>
      )}

      {/* Test Bench */}
      <SectionLabel>Test Bench</SectionLabel>
      <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', padding: '0.5rem 0.6rem' }}>
        <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.3rem' }}>
          <input
            type="text" value={testText}
            onChange={(e) => setTestText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleTestBench()}
            placeholder="Type text to synthesize..."
            style={{
              flex: 1, background: 'var(--bg-secondary)',
              border: '1px solid var(--text-muted)', borderRadius: 'var(--radius-sm)',
              padding: '0.3rem 0.5rem', color: 'var(--text-primary)',
              fontSize: '0.75rem', outline: 'none',
            }}
          />
          <select
            value={testLang} onChange={(e) => setTestLang(e.target.value)}
            style={{
              background: 'var(--bg-secondary)', border: '1px solid var(--text-muted)',
              borderRadius: 'var(--radius-sm)', padding: '0.3rem 0.4rem',
              color: 'var(--text-primary)', fontSize: '0.7rem', outline: 'none', colorScheme: 'dark',
            }}
          >
            {selectedLangs.map(id => (
              <option key={id} value={id}>{ALL_LANGUAGES[id] || id}</option>
            ))}
          </select>
          <SmallButton onClick={handleTestBench} disabled={!testText.trim() || isSynthesizing} color="var(--orb-speaking)">
            {isSynthesizing ? 'Speaking...' : 'Speak'}
          </SmallButton>
        </div>
      </div>

      {/* Synthesis Log (diagnostics) */}
      <SectionLabel>Diagnostics</SectionLabel>
      <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
          <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Synthesis Log</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            {isSynthesizing && (
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.65rem', color: '#cc8844' }}>
                <span style={{
                  width: 8, height: 8, borderRadius: '50%', background: '#cc8844',
                  boxShadow: '0 0 6px #cc8844', animation: 'pulse 1s ease-in-out infinite',
                }} />
                Synthesizing
              </span>
            )}
          </div>
        </div>
        <div style={{ maxHeight: 250, overflowY: 'auto' }}>
          {synthHistory.length === 0 && (
            <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>No synthesis yet</span>
          )}
          {synthHistory.slice().reverse().map((entry, i) => (
            <div key={i} style={{
              padding: '0.3rem 0', borderBottom: '1px solid var(--bg-secondary)', fontSize: '0.7rem',
            }}>
              <div style={{ color: 'var(--text-primary)', marginBottom: 2, wordBreak: 'break-word' }}>
                {entry.text.slice(0, 100)}{entry.text.length > 100 ? '...' : ''}
                {entry.lang && <span style={{ color: 'var(--text-muted)', marginLeft: 6 }}>[{entry.lang}]</span>}
              </div>
              <div style={{ display: 'flex', gap: '0.75rem', color: 'var(--text-muted)', fontSize: '0.6rem' }}>
                {entry.error ? (
                  <span style={{ color: '#ff4466' }}>Error: {entry.error}</span>
                ) : (
                  <>
                    <span>RTF: <span style={{ color: entry.rtf < 1 ? '#44cc88' : '#ff4466' }}>{entry.rtf.toFixed(2)}</span></span>
                    <span>{entry.duration_ms}ms</span>
                  </>
                )}
                <span>{new Date(entry.time).toLocaleTimeString()}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </SettingsPage>
  );
}
