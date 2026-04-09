import { useState, useCallback, useRef } from 'react';

export default function TextInput({ onSubmit, disabled = false }) {
  const [value, setValue] = useState('');
  const inputRef = useRef(null);

  const handleSubmit = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSubmit?.(trimmed);
    setValue('');
  }, [value, disabled, onSubmit]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }, [handleSubmit]);

  return (
    <div style={{
      display: 'flex',
      gap: '0.5rem',
      padding: '0.5rem 0.75rem',
      borderTop: '1px solid var(--bg-surface)',
      background: 'var(--bg-secondary)',
    }}>
      <input
        ref={inputRef}
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Type a message…"
        disabled={disabled}
        style={{
          flex: 1,
          background: 'var(--bg-surface)',
          border: '1px solid var(--text-muted)',
          borderRadius: 'var(--radius-sm)',
          padding: '0.5rem 0.75rem',
          color: 'var(--text-primary)',
          fontSize: '0.875rem',
          outline: 'none',
          transition: 'border-color 0.2s',
        }}
        onFocus={(e) => e.target.style.borderColor = 'var(--orb-listening)'}
        onBlur={(e) => e.target.style.borderColor = 'var(--text-muted)'}
      />
      <button
        onClick={handleSubmit}
        disabled={disabled || !value.trim()}
        aria-label="Send message"
        style={{
          background: value.trim() ? 'var(--orb-listening)' : 'var(--bg-surface)',
          border: 'none',
          borderRadius: 'var(--radius-sm)',
          padding: '0.5rem 1rem',
          color: value.trim() ? 'var(--bg-primary)' : 'var(--text-muted)',
          fontSize: '0.85rem',
          fontWeight: 600,
          cursor: value.trim() && !disabled ? 'pointer' : 'default',
          transition: 'background 0.2s, color 0.2s',
        }}
      >
        Send
      </button>
    </div>
  );
}
