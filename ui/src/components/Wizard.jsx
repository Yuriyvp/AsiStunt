import { useState, useCallback, useEffect } from 'react';
import WelcomeStep from './wizard/WelcomeStep';
import ConsentStep from './wizard/ConsentStep';
import AudioCheckStep from './wizard/AudioCheckStep';
import HeadphoneCheckStep from './wizard/HeadphoneCheckStep';
import LanguageStep from './wizard/LanguageStep';
import PersonaStep from './wizard/PersonaStep';
import VoicePreviewStep from './wizard/VoicePreviewStep';
import DoneStep from './wizard/DoneStep';

const STEPS = [
  { key: 'welcome', label: 'Welcome', skipLabel: null },
  { key: 'consent', label: 'Privacy', skipLabel: null },
  { key: 'audio', label: 'Audio', skipLabel: 'Skip' },
  { key: 'headphone', label: 'Headphones', skipLabel: 'Skip' },
  { key: 'language', label: 'Language', skipLabel: null },
  { key: 'persona', label: 'Persona', skipLabel: null },
  { key: 'preview', label: 'Preview', skipLabel: 'Skip' },
  { key: 'done', label: 'Done', skipLabel: null },
];

export default function Wizard({ onComplete, sendCommand, energy = 0, componentStatus = {} }) {
  const [step, setStep] = useState(0);
  const [consented, setConsented] = useState(false);
  const [selectedLanguages, setSelectedLanguages] = useState(['hr', 'en']);
  const [selectedPersona, setSelectedPersona] = useState('Aria');

  const canAdvance = useCallback(() => {
    if (STEPS[step].key === 'consent') return consented;
    return true;
  }, [step, consented]);

  const next = useCallback(() => {
    if (step < STEPS.length - 1) {
      setStep(step + 1);
    } else {
      onComplete?.({
        languages: selectedLanguages,
        persona: selectedPersona,
        consented,
      });
    }
  }, [step, onComplete, selectedLanguages, selectedPersona, consented]);

  const prev = useCallback(() => {
    if (step > 0) setStep(step - 1);
  }, [step]);

  // Keyboard navigation: left/right arrows, Escape
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'ArrowRight' || e.key === 'Enter') {
        if (canAdvance() && document.activeElement?.tagName !== 'INPUT'
            && document.activeElement?.tagName !== 'TEXTAREA') {
          e.preventDefault();
          next();
        }
      } else if (e.key === 'ArrowLeft') {
        if (step > 0) {
          e.preventDefault();
          prev();
        }
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [canAdvance, next, prev, step]);

  const currentStep = STEPS[step];
  const isLast = step === STEPS.length - 1;

  return (
    <div
      role="dialog"
      aria-label={`Setup wizard: step ${step + 1} of ${STEPS.length}, ${currentStep.label}`}
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--bg-primary)',
        overflow: 'hidden',
      }}
    >
      {/* Progress bar */}
      <div
        role="progressbar"
        aria-valuenow={step + 1}
        aria-valuemin={1}
        aria-valuemax={STEPS.length}
        aria-label={`Step ${step + 1} of ${STEPS.length}`}
        style={{
          display: 'flex',
          gap: 2,
          padding: '0.75rem 1rem 0.5rem',
        }}
      >
        {STEPS.map((s, i) => (
          <div
            key={s.key}
            style={{
              flex: 1,
              height: 3,
              borderRadius: 2,
              background: i <= step ? 'var(--orb-listening)' : 'var(--bg-surface)',
              transition: 'background 0.3s',
            }}
          />
        ))}
      </div>

      {/* Step label */}
      <div style={{
        textAlign: 'center',
        fontSize: '0.65rem',
        color: 'var(--text-muted)',
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        padding: '0 1rem',
      }}>
        Step {step + 1} of {STEPS.length} \u2014 {currentStep.label}
      </div>

      {/* Step content */}
      <div style={{
        flex: 1,
        overflow: 'auto',
        padding: '0 1.5rem',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
      }}>
        {currentStep.key === 'welcome' && <WelcomeStep componentStatus={componentStatus} />}
        {currentStep.key === 'consent' && <ConsentStep onConsent={setConsented} />}
        {currentStep.key === 'audio' && <AudioCheckStep energy={energy} />}
        {currentStep.key === 'headphone' && <HeadphoneCheckStep />}
        {currentStep.key === 'language' && <LanguageStep defaultLanguages={selectedLanguages} onSelect={setSelectedLanguages} />}
        {currentStep.key === 'persona' && <PersonaStep selected={selectedPersona} onSelect={setSelectedPersona} />}
        {currentStep.key === 'preview' && <VoicePreviewStep sendCommand={sendCommand} />}
        {currentStep.key === 'done' && <DoneStep />}
      </div>

      {/* Navigation buttons */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '0.75rem 1.5rem',
        borderTop: '1px solid var(--bg-surface)',
      }}>
        <button
          onClick={prev}
          disabled={step === 0}
          style={{
            background: 'none',
            border: 'none',
            color: step === 0 ? 'var(--text-muted)' : 'var(--text-secondary)',
            fontSize: '0.85rem',
            cursor: step === 0 ? 'default' : 'pointer',
            padding: '0.5rem 1rem',
          }}
        >
          {'\u2190'} Back
        </button>

        <div style={{ display: 'flex', gap: '0.5rem' }}>
          {currentStep.skipLabel && (
            <button
              onClick={next}
              style={{
                background: 'none',
                border: '1px solid var(--text-muted)',
                borderRadius: 'var(--radius-sm)',
                color: 'var(--text-secondary)',
                fontSize: '0.8rem',
                cursor: 'pointer',
                padding: '0.5rem 1rem',
              }}
            >
              {currentStep.skipLabel}
            </button>
          )}
          <button
            onClick={next}
            disabled={!canAdvance()}
            style={{
              background: canAdvance() ? 'var(--orb-listening)' : 'var(--bg-surface)',
              border: 'none',
              borderRadius: 'var(--radius-sm)',
              color: canAdvance() ? 'var(--bg-primary)' : 'var(--text-muted)',
              fontSize: '0.85rem',
              fontWeight: 600,
              cursor: canAdvance() ? 'pointer' : 'default',
              padding: '0.5rem 1.25rem',
              transition: 'all 0.2s',
            }}
          >
            {isLast ? 'Finish' : 'Next \u2192'}
          </button>
        </div>
      </div>
    </div>
  );
}
