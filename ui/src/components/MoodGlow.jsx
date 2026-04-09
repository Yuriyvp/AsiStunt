import { useState, useEffect } from 'react';

const MOOD_LABELS = {
  neutral: null,
  warm: 'Warm',
  playful: 'Playful',
  calm: 'Calm',
  concerned: 'Concerned',
};

export default function MoodGlow({ mood = 'neutral' }) {
  const [visible, setVisible] = useState(false);
  const [displayMood, setDisplayMood] = useState(mood);

  useEffect(() => {
    if (mood === 'neutral') {
      setVisible(false);
      return;
    }
    setDisplayMood(mood);
    setVisible(true);
    const timer = setTimeout(() => setVisible(false), 5000);
    return () => clearTimeout(timer);
  }, [mood]);

  const label = MOOD_LABELS[displayMood];
  if (!label) return null;

  return (
    <div
      role="status"
      aria-live="polite"
      style={{
        textAlign: 'center',
        fontSize: '0.7rem',
        color: `var(--mood-${displayMood})`,
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        opacity: visible ? 0.7 : 0,
        transition: 'opacity 1s ease-in-out',
        height: '1.25rem',
        lineHeight: '1.25rem',
      }}
    >
      {visible ? label : ''}
    </div>
  );
}
