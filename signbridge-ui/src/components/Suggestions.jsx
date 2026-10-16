import React, { useMemo } from 'react';

// Curated dictionary of common English words for live predictive completion
const COMMON_WORDS = [
  'ABOUT', 'AFTER', 'AGAIN', 'ALL', 'ALSO', 'AND', 'ANY', 'ARE', 'ASL', 'BACK',
  'BECAUSE', 'BEFORE', 'BEST', 'BETTER', 'CALL', 'CAN', 'COME', 'COULD', 'DAY', 'DEAR',
  'DO', 'DOCTOR', 'DOWN', 'DRINK', 'EAT', 'EMERGENCY', 'EVERY', 'FAMILY', 'FEEL', 'FIND',
  'FIRST', 'FOOD', 'FOR', 'FRIEND', 'FROM', 'GET', 'GIVE', 'GO', 'GOOD', 'GREAT',
  'HAVE', 'HEARING', 'HELLO', 'HELP', 'HERE', 'HOME', 'HOSPITAL', 'HOW', 'HURT', 'ISL',
  'JUST', 'KNOW', 'LANGUAGE', 'LEARN', 'LIKE', 'LITTLE', 'LOOK', 'LOVE', 'MAKE', 'ME',
  'MEDICINE', 'MORE', 'NAME', 'NEED', 'NEW', 'NO', 'NOT', 'NOW', 'OF', 'OFFICE',
  'OK', 'ONE', 'ONLY', 'OPEN', 'OUR', 'OUT', 'OVER', 'PAIN', 'PEOPLE', 'PLEASE',
  'RIGHT', 'SAY', 'SEE', 'SIGN', 'SOME', 'SORRY', 'STOP', 'TAKE', 'TELL', 'THANK',
  'THAT', 'THE', 'THEIR', 'THEM', 'THEN', 'THERE', 'THEY', 'THING', 'THINK', 'THIS',
  'TIME', 'TO', 'TODAY', 'UNDERSTAND', 'UP', 'US', 'USE', 'VERY', 'WANT', 'WATER',
  'WAY', 'WE', 'WELL', 'WHAT', 'WHEN', 'WHERE', 'WHICH', 'WHO', 'WHY', 'WILL',
  'WITH', 'WORD', 'WORK', 'WORLD', 'WOULD', 'WRITE', 'YES', 'YOU', 'YOUR'
];

/**
 * Suggestions component adhering strictly to Section 4.5 of the design blueprint.
 * Displays up to 5 matching word prediction pills when typing in letters mode with length >= 2.
 */
export const Suggestions = React.memo(function Suggestions({
  currentWord,
  currentMode,
  setWord
}) {
  const suggestions = useMemo(() => {
    if (currentMode !== 'letters' || !currentWord || currentWord.length < 2) {
      return [];
    }
    const prefix = currentWord.trim().toUpperCase();
    return COMMON_WORDS.filter(
      (word) => word.startsWith(prefix) && word !== prefix
    ).slice(0, 5);
  }, [currentWord, currentMode]);

  if (suggestions.length === 0) {
    return null;
  }

  return (
    <div className="sb-suggestions-panel" aria-label="Word suggestions">
      <span className="sb-section-label sr-only">Suggestions</span>
      <div className="sb-suggestions-list" role="list">
        {suggestions.map((word) => (
          <button
            key={word}
            type="button"
            className="sb-suggestion-pill"
            onClick={() => setWord(word)}
            aria-label={`Select suggestion ${word}`}
          >
            {word}
          </button>
        ))}
      </div>
    </div>
  );
});
