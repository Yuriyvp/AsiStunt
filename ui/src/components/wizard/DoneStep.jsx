export default function DoneStep() {
  return (
    <div style={{ padding: '2rem 0', textAlign: 'center' }}>
      <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>{'\u2728'}</div>
      <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '0.75rem' }}>
        All Set!
      </h2>
      <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.6, maxWidth: 280, margin: '0 auto' }}>
        Your voice assistant is ready. Click <strong style={{ color: 'var(--text-primary)' }}>Finish</strong> to
        start chatting.
      </p>
      <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '1rem' }}>
        You can change these settings anytime from the right-click menu.
      </p>
    </div>
  );
}
