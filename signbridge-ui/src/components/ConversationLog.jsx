import React, { useEffect, useRef } from 'react';

/**
 * ConversationLog adhering strictly to Section 4.6 of the design blueprint.
 * Displays time-stamped history of translated utterances with row delete action on hover.
 */
export const ConversationLog = React.memo(function ConversationLog({
  conversationLog,
  deleteLogEntry
}) {
  const containerRef = useRef(null);
  const prevCountRef = useRef(conversationLog.length);

  // Auto-scroll to bottom only when a new entry is appended
  useEffect(() => {
    if (conversationLog.length > prevCountRef.current && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
    prevCountRef.current = conversationLog.length;
  }, [conversationLog.length]);

  return (
    <section className="sb-conversation-panel" aria-labelledby="history-heading">
      <div className="sb-section-label" id="history-heading">
        History
      </div>

      <div
        className="sb-conversation-scroll"
        ref={containerRef}
        tabIndex={0}
        role="region"
        aria-label="Conversation history"
      >
        {conversationLog.length === 0 ? (
          <div className="sb-conversation-empty">
            No conversation yet
          </div>
        ) : (
          <div className="sb-conversation-list" role="list">
            {conversationLog.map((entry, idx) => {
              const isEmergency = Boolean(entry.text && entry.text.includes('EMERGENCY'));
              return (
                <div
                  key={`${entry.time}-${idx}-${entry.text}`}
                  className={`sb-conversation-entry ${isEmergency ? 'emergency' : ''}`}
                  role="listitem"
                >
                <span className="sb-conversation-time">
                  {entry.time || '--:--'}
                </span>
                <span className="sb-conversation-text">
                  {entry.text}
                </span>
                <button
                  type="button"
                  className="sb-conversation-delete"
                  onClick={() => deleteLogEntry(idx)}
                  aria-label={`Delete entry "${entry.text}" from history`}
                  title="Delete entry"
                >
                  ×
                </button>
              </div>
            );
          })}
        </div>
        )}
      </div>
    </section>
  );
});
