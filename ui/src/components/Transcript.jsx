import { useRef, useEffect } from 'react';

const SOURCE_ICONS = {
  voice: '\uD83C\uDF99\uFE0F',
  text: '\u2328\uFE0F',
};

function TurnBubble({ turn }) {
  const isUser = turn.role === 'user';
  const icon = SOURCE_ICONS[turn.source] || '';
  const interrupted = turn.interrupted && turn.spokenText != null;
  const sourceLabel = turn.source === 'voice' ? 'via voice' : 'via text';

  return (
    <div
      role="listitem"
      aria-label={`${isUser ? 'You' : 'Assistant'} ${sourceLabel}: ${turn.text}`}
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start',
        marginBottom: '0.5rem',
        maxWidth: '85%',
        alignSelf: isUser ? 'flex-end' : 'flex-start',
      }}
    >
      <div
        style={{
          fontSize: '0.65rem',
          color: 'var(--text-muted)',
          marginBottom: '0.15rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.25rem',
        }}
      >
        {!isUser && <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>AI</span>}
        {isUser && <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>You</span>}
        {icon && <span aria-hidden="true">{icon}</span>}
      </div>
      <div
        style={{
          background: isUser ? 'var(--bg-user-bubble)' : 'var(--bg-ai-bubble)',
          color: 'var(--text-primary)',
          padding: '0.5rem 0.75rem',
          borderRadius: isUser
            ? 'var(--radius-md) var(--radius-md) var(--radius-sm) var(--radius-md)'
            : 'var(--radius-md) var(--radius-md) var(--radius-md) var(--radius-sm)',
          fontSize: '0.875rem',
          lineHeight: 1.5,
          wordBreak: 'break-word',
        }}
      >
        {interrupted ? (
          <>
            <span>{turn.spokenText}</span>
            <span
              style={{
                color: 'var(--text-muted)',
                fontStyle: 'italic',
                opacity: 0.6,
              }}
              title="Interrupted \u2014 this part was not spoken"
            >
              {turn.text.slice(turn.spokenText.length)}
            </span>
          </>
        ) : (
          turn.text
        )}
      </div>
    </div>
  );
}

export default function Transcript({ turns = [] }) {
  const bottomRef = useRef(null);
  const containerRef = useRef(null);
  const userScrolledRef = useRef(false);

  // Auto-scroll unless user has scrolled up
  useEffect(() => {
    if (!userScrolledRef.current && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [turns]);

  const handleScroll = () => {
    const el = containerRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    userScrolledRef.current = !atBottom;
  };

  const displayed = turns.slice(-50);

  return (
    <div
      ref={containerRef}
      onScroll={handleScroll}
      role="log"
      aria-live="polite"
      aria-label="Conversation transcript"
      style={{
        flex: 1,
        overflowY: 'auto',
        padding: '0.75rem',
        display: 'flex',
        flexDirection: 'column',
        gap: 0,
        scrollbarWidth: 'thin',
        scrollbarColor: 'var(--bg-surface) transparent',
      }}
    >
      {displayed.length === 0 && (
        <div style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-muted)',
          fontSize: '0.85rem',
        }}>
          Start a conversation...
        </div>
      )}
      <div role="list">
        {displayed.map((turn, i) => (
          <TurnBubble key={turn.id || i} turn={turn} />
        ))}
      </div>
      <div ref={bottomRef} />
    </div>
  );
}
