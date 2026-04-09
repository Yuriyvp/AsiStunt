const STATUS_MESSAGES = {
  DISABLED:    '',
  IDLE:        '',
  LISTENING:   'Listening\u2026',
  PROCESSING:  'Thinking\u2026',
  SPEAKING:    '',
  INTERRUPTED: 'Interrupted',
};

export default function StatusLine({ state = 'IDLE', customMessage = '' }) {
  const message = customMessage || STATUS_MESSAGES[state] || '';

  return (
    <div
      role="status"
      aria-live="polite"
      aria-atomic="true"
      style={{
        textAlign: 'center',
        fontSize: '0.8rem',
        color: 'var(--text-secondary)',
        height: '1.5rem',
        lineHeight: '1.5rem',
        transition: 'opacity 0.3s ease',
        opacity: message ? 1 : 0,
      }}
    >
      {message}
    </div>
  );
}
