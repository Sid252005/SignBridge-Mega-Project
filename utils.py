"""
utils.py — SignBridge AI shared utilities
=========================================
Provides:
  - normalize_landmarks()   : wrist-relative, palm-scale-invariant normalization
  - augment_dataset()       : synthetic data augmentation
                              v3 adds: perspective warp + finger dropout
                              for better accuracy on 24 similar-looking signs
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# LANDMARK NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe gives 21 hand landmarks (x, y, z) = 63 raw numbers.
# Problem: raw coords change with hand position & camera distance.
# Fix:
#   1. Translate so wrist (point 0) is at origin.
#   2. Scale by palm size (wrist → middle-finger MCP, points 0 & 9).
# This makes features position- and scale-invariant.

WRIST_IDX = 0          # landmark index 0 = wrist
PALM_REF_IDX = 9       # landmark index 9 = middle finger MCP (base)


def normalize_landmarks(raw: list | np.ndarray) -> np.ndarray:
    """
    Normalize a flat array of 63 landmark values.

    Parameters
    ----------
    raw : list or np.ndarray of shape (63,)
        Raw x,y,z coordinates from MediaPipe, flattened.

    Returns
    -------
    np.ndarray of shape (63,)
        Normalized, position- and scale-invariant landmarks.
    """
    pts = np.array(raw, dtype=np.float32).reshape(21, 3)

    # 1. Translate — subtract wrist
    pts -= pts[WRIST_IDX]

    # 2. Scale — divide by palm length (wrist → middle MCP)
    palm_len = np.linalg.norm(pts[PALM_REF_IDX])
    if palm_len > 1e-6:          # avoid divide-by-zero
        pts /= palm_len

    return pts.flatten()


# ─────────────────────────────────────────────────────────────────────────────
# DATA AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
# Artificially multiply training samples by adding realistic variation:
#   1. Gaussian noise   — simulates sensor jitter
#   2. Scale jitter     — hand closer/farther from camera
#   3. 2-D rotation     — slight wrist tilt (in x-y plane)

def _add_noise(pts: np.ndarray, sigma: float = 0.015) -> np.ndarray:
    return pts + np.random.randn(*pts.shape).astype(np.float32) * sigma


def _scale_jitter(pts: np.ndarray, low: float = 0.85, high: float = 1.15) -> np.ndarray:
    return pts * np.random.uniform(low, high)


def _rotate_2d(pts: np.ndarray, max_angle_deg: float = 15.0) -> np.ndarray:
    """Rotate x-y axes only; leave z unchanged. Increased to 15 deg for 24-class."""
    angle = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts = pts.reshape(21, 3).copy()
    pts[:, :2] = pts[:, :2] @ R.T
    return pts.flatten()


def mirror_hand(pts: np.ndarray) -> np.ndarray:
    """
    Flip the X-axis of normalized landmarks to simulate the OPPOSITE hand.

    After wrist-relative normalization, a left hand is simply the mirror
    of a right hand along X.  Negating X produces a geometrically perfect
    left-hand sample from a right-hand sample (and vice-versa).

    This is the most effective fix for cross-hand detection failures:
    apply it during augmentation so the model learns BOTH orientations.

    Also exported for use in predict.py (runtime hand-flip).
    """
    pts = pts.reshape(21, 3).copy()
    pts[:, 0] = -pts[:, 0]   # flip X; Y and Z stay the same
    return pts.flatten()


def _perspective_warp(pts: np.ndarray, strength: float = 0.08) -> np.ndarray:
    """
    Simulate a camera-angle shift by applying a mild shear to x and y.
    Helps distinguish signs that look identical from a straight-on view
    (e.g. G/Q, P/K, M/N) when viewed from a slight angle.
    """
    pts = pts.reshape(21, 3).copy()
    shear_x = np.random.uniform(-strength, strength)
    shear_y = np.random.uniform(-strength, strength)
    # shear: x' = x + shear_x*y,  y' = y + shear_y*x
    new_x = pts[:, 0] + shear_x * pts[:, 1]
    new_y = pts[:, 1] + shear_y * pts[:, 0]
    pts[:, 0] = new_x
    pts[:, 1] = new_y
    return pts.flatten()


def _finger_dropout(pts: np.ndarray, drop_prob: float = 0.08) -> np.ndarray:
    """
    Randomly zero out a few landmark coordinates to simulate partial occlusion.
    Forces the model to learn from partial hand information, making it more
    robust when fingers are partially hidden behind the palm.
    """
    pts = pts.copy()
    mask = np.random.rand(len(pts)) < drop_prob
    pts[mask] = 0.0
    return pts


def augment_dataset(X: np.ndarray, y: np.ndarray,
                    factor: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment a dataset by `factor` times using 5 augmentation types:
      1. Gaussian noise    — sensor jitter
      2. Scale jitter      — hand distance from camera
      3. 2-D rotation      — wrist tilt (up to 15 deg)
      4. Perspective warp  — slight camera angle shift (NEW in v3)
      5. Finger dropout    — partial landmark occlusion (NEW in v3)
      6. Mirror hand       — X-flip to simulate opposite hand (NEW in v4)

    Parameters
    ----------
    X : np.ndarray, shape (N, 63)   — already normalized landmarks
    y : np.ndarray, shape (N,)      — integer labels
    factor : int                    — augmentation multiplier (default 15)

    Returns
    -------
    X_aug, y_aug : augmented arrays (includes originals)
    """
    X_list, y_list = [X.copy()], [y.copy()]

    # Mirror ALL training samples to teach the model BOTH hands at once
    X_list.append(np.array([mirror_hand(r) for r in X], dtype=np.float32))
    y_list.append(y.copy())

    aug_ops = ['noise', 'scale', 'rotate', 'warp', 'dropout', 'mirror']

    for _ in range(factor - 1):
        aug = []
        for row in X:
            row = row.copy()
            # Randomly pick 1–3 augmentations to apply per sample
            ops = np.random.choice(aug_ops,
                                   size=np.random.randint(1, 4),
                                   replace=False)
            for op in ops:
                if op == 'noise':
                    row = _add_noise(row)
                elif op == 'scale':
                    row = _scale_jitter(row)
                elif op == 'rotate':
                    row = _rotate_2d(row)
                elif op == 'warp':
                    row = _perspective_warp(row)
                elif op == 'dropout':
                    row = _finger_dropout(row)
                elif op == 'mirror':
                    row = mirror_hand(row)
            aug.append(row)
        X_list.append(np.array(aug, dtype=np.float32))
        y_list.append(y.copy())

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# ISL (INDIAN SIGN LANGUAGE) SUPPORT
# ─────────────────────────────────────────────────────────────────────────────
# ISL uses TWO hands for many signs, so we need 126-dim features
# (63 per hand). The first 63 numbers represent the right hand,
# the last 63 represent the left hand (mirrored during collection).
# If only one hand is visible, the other is zero-padded.

def normalize_isl_landmarks(raw_hand1, raw_hand2=None):
    """
    Normalize ISL landmarks for two hands independently.

    Parameters
    ----------
    raw_hand1 : list or np.ndarray of shape (63,)
        Raw x,y,z coordinates for the first hand (typically right).
    raw_hand2 : list or np.ndarray of shape (63,) or None
        Raw x,y,z coordinates for the second hand (typically left).
        If None or all zeros, treated as absent (zero-padded).

    Returns
    -------
    np.ndarray of shape (126,)
        Concatenated normalized landmarks: [hand1_63 | hand2_63].
        Zero-padded hands remain as zeros (no normalization applied).
    """
    # Normalize first hand
    hand1 = np.array(raw_hand1, dtype=np.float32).flatten()
    if np.any(np.abs(hand1) > 1e-8):
        hand1_norm = normalize_landmarks(hand1)
    else:
        hand1_norm = np.zeros(63, dtype=np.float32)

    # Normalize second hand
    if raw_hand2 is None:
        hand2_norm = np.zeros(63, dtype=np.float32)
    else:
        hand2 = np.array(raw_hand2, dtype=np.float32).flatten()
        if np.any(np.abs(hand2) > 1e-8):
            hand2_norm = normalize_landmarks(hand2)
        else:
            hand2_norm = np.zeros(63, dtype=np.float32)

    return np.concatenate([hand1_norm, hand2_norm])


def augment_isl_dataset(X: np.ndarray, y: np.ndarray,
                        factor: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment an ISL dataset of 126-length arrays by `factor` times.
    Each 126-length sample is split into two 63-length halves,
    augmented independently (only non-zero halves), then recombined.

    Uses the same augmentation types as augment_dataset():
      1. Gaussian noise    2. Scale jitter    3. 2-D rotation
      4. Perspective warp  5. Finger dropout  6. Mirror hand

    Parameters
    ----------
    X : np.ndarray, shape (N, 126)  — already normalized ISL landmarks
    y : np.ndarray, shape (N,)      — integer labels
    factor : int                    — augmentation multiplier

    Returns
    -------
    X_aug, y_aug : augmented arrays (includes originals)
    """
    X_list, y_list = [X.copy()], [y.copy()]

    # Mirror ALL samples (both halves independently)
    mirrored = []
    for row in X:
        h1 = row[:63].copy()
        h2 = row[63:].copy()
        if np.any(np.abs(h1) > 1e-8):
            h1 = mirror_hand(h1)
        if np.any(np.abs(h2) > 1e-8):
            h2 = mirror_hand(h2)
        mirrored.append(np.concatenate([h1, h2]))
    X_list.append(np.array(mirrored, dtype=np.float32))
    y_list.append(y.copy())

    aug_ops = ['noise', 'scale', 'rotate', 'warp', 'dropout', 'mirror']

    for _ in range(factor - 1):
        aug = []
        for row in X:
            h1 = row[:63].copy()
            h2 = row[63:].copy()

            # Randomly pick 1–3 augmentations to apply per sample
            ops = np.random.choice(aug_ops,
                                   size=np.random.randint(1, 4),
                                   replace=False)
            for op in ops:
                if op == 'noise':
                    if np.any(np.abs(h1) > 1e-8):
                        h1 = _add_noise(h1)
                    if np.any(np.abs(h2) > 1e-8):
                        h2 = _add_noise(h2)
                elif op == 'scale':
                    if np.any(np.abs(h1) > 1e-8):
                        h1 = _scale_jitter(h1)
                    if np.any(np.abs(h2) > 1e-8):
                        h2 = _scale_jitter(h2)
                elif op == 'rotate':
                    if np.any(np.abs(h1) > 1e-8):
                        h1 = _rotate_2d(h1)
                    if np.any(np.abs(h2) > 1e-8):
                        h2 = _rotate_2d(h2)
                elif op == 'warp':
                    if np.any(np.abs(h1) > 1e-8):
                        h1 = _perspective_warp(h1)
                    if np.any(np.abs(h2) > 1e-8):
                        h2 = _perspective_warp(h2)
                elif op == 'dropout':
                    if np.any(np.abs(h1) > 1e-8):
                        h1 = _finger_dropout(h1)
                    if np.any(np.abs(h2) > 1e-8):
                        h2 = _finger_dropout(h2)
                elif op == 'mirror':
                    if np.any(np.abs(h1) > 1e-8):
                        h1 = mirror_hand(h1)
                    if np.any(np.abs(h2) > 1e-8):
                        h2 = mirror_hand(h2)
            aug.append(np.concatenate([h1, h2]))
        X_list.append(np.array(aug, dtype=np.float32))
        y_list.append(y.copy())

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# MOTION SEQUENCE AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
# Used by train_words_motion_model.py to augment (N, 30, 63) motion sequences.
# Applies lightweight noise + scale jitter per-sequence to multiply training data.

def augment_sequences(sequences, labels, factor=2):
    import numpy as np
    augmented_X, augmented_y = [], []
    for seq, label in zip(sequences, labels):
        for _ in range(factor):
            noise = np.random.normal(0, 0.005, seq.shape)
            scale = np.random.uniform(0.9, 1.1)
            augmented_X.append(seq * scale + noise)
            augmented_y.append(label)
    return (
        np.array(augmented_X + list(sequences)),
        np.array(augmented_y + list(labels))
    )
