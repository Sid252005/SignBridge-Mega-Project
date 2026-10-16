import React from 'react';

/**
 * WordBuilder adhering strictly to Section 4.4 of the design blueprint.
 * Displays spaced monospace word, blinking accent cursor, character count, and compact inline action buttons.
 */
export const WordBuilder = React.memo(function WordBuilder({
  currentWord,
  addLetter,
  backspace,
  clearWord,
  prediction
}) {
  const hasWord = currentWord && currentWord.length > 0;
  const charCount = currentWord ? currentWord.length : 0;

  return (
    <section className="sb-word-builder" aria-labelledby="phrase-heading">
      <div className="sb-section-label" id="phrase-heading">
        Phrase
      </div>

      {/* Word String Display with Blinking Cursor */}
      <div className="sb-phrase-display" tabIndex={0} role="region" aria-label="Current phrase">
        {hasWord ? (
          <span className="sb-phrase-text">
            {currentWord.split('').join(' ')}
            <span className="sb-cursor" aria-hidden="true">|</span>
          </span>
        ) : (
          <span className="sb-phrase-placeholder">
            Start signing…
            <span className="sb-cursor" aria-hidden="true">|</span>
          </span>
        )}
      </div>

      {/* Character Count */}
      <div className="sb-char-count">
        {charCount} {charCount === 1 ? 'character' : 'characters'}
      </div>

      {/* Inline Action Row */}
      <div className="sb-phrase-actions" role="toolbar" aria-label="Phrase controls">
        <button
          type="button"
          className="sb-btn sb-btn-primary sb-btn-compact"
          onClick={() => addLetter()}
          id="btn-add-letter"
          aria-label="Add detected letter to phrase"
        >
          + Add
        </button>

        <button
          type="button"
          className="sb-btn sb-btn-ghost sb-btn-compact"
          onClick={backspace}
          disabled={!hasWord}
          id="btn-backspace"
          aria-label="Backspace last character"
        >
          ⌫ Backspace
        </button>

        <button
          type="button"
          className="sb-btn sb-btn-ghost sb-btn-compact"
          onClick={clearWord}
          disabled={!hasWord}
          id="btn-clear"
          aria-label="Clear phrase"
        >
          ✕ Clear
        </button>
      </div>
    </section>
  );
});
