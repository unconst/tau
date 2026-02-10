"""Braille progress bar spinners for Telegram thinking animations.

10 animated spinner designs using 10-character braille lanes.
A random spinner is selected for each query to give visual variety.

Designs:
    1. ping_pong      — single dot bounces left ⇄ right
    2. scanner         — thick rotating head sweeps
    3. comet           — bright head + trailing decay
    4. elastic         — sinusoidal, slows at edges
    5. dual_particles  — two dots converge, collide, diverge
    6. wave_sweep      — gaussian bump sweeps across
    7. fill_drain      — bar fills left→right, drains right→left
    8. pendulum        — oscillates around center
    9. liquid          — phase-shifted multi-cell spinner
   10. heartbeat       — pulse expands from center, collapses
"""

import math
import random

__all__ = ["get_random_spinner", "BrailleSpinner"]

WIDTH = 10
_B = 0x2800  # Braille Unicode block base


def _b(n: int) -> str:
    """Braille character from dot-pattern bitmask."""
    return chr(_B + n)


E = _b(0)  # Empty braille cell ⠀

# ── Dot-pattern constants ────────────────────────────────────────────

# Individual dots in standard braille numbering order
_DOTS = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80]

# Cumulative fill (adding dots progressively for "growth" effects)
_FILL = [0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3F, 0x7F, 0xFF]

# Scanner head rotation (6-dot base with one dot toggling off)
_SCAN = [0x3F, 0x3E, 0x3D, 0x3B, 0x1F, 0x2F, 0x37, 0x27]

# Classic rotating spinner phases (10 phases)
_SPIN = [0x0B, 0x19, 0x39, 0x38, 0x3C, 0x34, 0x26, 0x27, 0x07, 0x0F]

FULL = 0x3F  # ⠿  all 6 standard dots


# ── Helpers ──────────────────────────────────────────────────────────

def _frame(cell_map: dict[int, int], w: int = WIDTH) -> str:
    """Build a WIDTH-char frame from {cell_index: dot_pattern}."""
    cells = [E] * w
    for i, p in cell_map.items():
        if 0 <= i < w and p > 0:
            cells[i] = _b(p)
    return "".join(cells)


def _bounce(fwd: list[str]) -> list[str]:
    """Append reversed interior frames for ping-pong effect."""
    if len(fwd) <= 2:
        return list(fwd) + list(reversed(fwd))
    return list(fwd) + list(reversed(fwd[1:-1]))


# ── Spinner class ────────────────────────────────────────────────────

class BrailleSpinner:
    """Pre-computed braille animation that indexes by elapsed time."""
    __slots__ = ("name", "frames", "cycle_s")

    def __init__(self, name: str, frames: list[str], cycle_s: float = 5.0):
        self.name = name
        self.frames = frames
        self.cycle_s = cycle_s

    def frame_at(self, elapsed: float) -> str:
        """Get the animation frame for the given elapsed time."""
        n = len(self.frames)
        if n == 0:
            return E * WIDTH
        idx = int(elapsed / self.cycle_s * n) % n
        return self.frames[idx]


# ── 1. Single-dot ping-pong ─────────────────────────────────────────

def _pingpong() -> BrailleSpinner:
    """One dot bounces left ⇄ right across the lane."""
    fwd = []
    for cell in range(WIDTH):
        for dot in _DOTS:
            fwd.append(_frame({cell: dot}))
    return BrailleSpinner("ping_pong", _bounce(fwd), 3.0)


# ── 2. Scanner bar ──────────────────────────────────────────────────

def _scanner() -> BrailleSpinner:
    """Thick rotating head sweeps back and forth (sci-fi console)."""
    fwd = []
    for cell in range(WIDTH):
        for phase in _SCAN:
            fwd.append(_frame({cell: phase}))
    return BrailleSpinner("scanner", _bounce(fwd), 3.0)


# ── 3. Comet with tail ──────────────────────────────────────────────

def _comet() -> BrailleSpinner:
    """Bright head + trailing decay, bounces at edges."""
    tail = [0x37, 0x07, 0x03, 0x01]  # decay behind head

    fwd = []
    for head in range(WIDTH):
        f = {head: FULL}
        for ti, tp in enumerate(tail):
            t = head - ti - 1
            if 0 <= t < WIDTH:
                f[t] = tp
        fwd.append(_frame(f))

    bwd = []
    for head in range(WIDTH - 1, -1, -1):
        f = {head: FULL}
        for ti, tp in enumerate(tail):
            t = head + ti + 1
            if 0 <= t < WIDTH:
                f[t] = tp
        bwd.append(_frame(f))

    return BrailleSpinner("comet", fwd + bwd, 2.0)


# ── 4. Elastic bounce ───────────────────────────────────────────────

def _elastic() -> BrailleSpinner:
    """Sinusoidal motion — slows at edges, speeds through the middle."""
    n = 60
    frames = []
    for i in range(n):
        t = i / n
        # Sinusoidal oscillation: slow at edges, fast at center
        pos = (math.sin(2 * math.pi * t - math.pi / 2) + 1) / 2 * (WIDTH - 1)
        cell = min(int(pos), WIDTH - 1)
        frac = pos - cell
        dot_i = min(int(frac * len(_DOTS)), len(_DOTS) - 1)
        f: dict[int, int] = {cell: _DOTS[dot_i]}
        # Stretch trail when moving fast (near center)
        speed = abs(math.cos(2 * math.pi * t - math.pi / 2))
        if speed > 0.6:
            if cell > 0:
                f[cell - 1] = 0x80
            if cell < WIDTH - 1:
                f[cell + 1] = 0x01
        frames.append(_frame(f))
    return BrailleSpinner("elastic", frames, 2.5)


# ── 5. Dual particles ───────────────────────────────────────────────

def _dual() -> BrailleSpinner:
    """Two dots converge from edges, collide at center, bounce back."""
    mid = WIDTH // 2
    frames = []

    # Converge
    for step in range(mid + 1):
        left, right = step, WIDTH - 1 - step
        if left >= right:
            frames.append(_frame({left: FULL}))
        else:
            frames.append(_frame({left: 0x01, right: 0x80}))

    # Collision flash
    frames.append(_frame({mid: 0xFF}))
    frames.append(_frame({mid: FULL}))

    # Diverge (reversed dot roles — they bounced)
    for step in range(mid - 1, -1, -1):
        left, right = step, WIDTH - 1 - step
        frames.append(_frame({left: 0x80, right: 0x01}))

    return BrailleSpinner("dual_particles", frames, 2.0)


# ── 6. Wave sweep ───────────────────────────────────────────────────

def _wave() -> BrailleSpinner:
    """Gaussian-ish bump sweeps across and bounces back."""
    bump = [0x01, 0x07, 0x1F, 0x3F, 0x1F, 0x07, 0x01]
    bh = len(bump) // 2

    fwd = []
    for center in range(WIDTH):
        f = {}
        for bi, bp in enumerate(bump):
            c = center + bi - bh
            if 0 <= c < WIDTH:
                f[c] = bp
        fwd.append(_frame(f))

    return BrailleSpinner("wave_sweep", _bounce(fwd), 2.0)


# ── 7. Fill-and-drain oscillator ─────────────────────────────────────

def _fill_drain() -> BrailleSpinner:
    """Bar fills left→right, then drains right→left."""
    phases = [0x01, 0x03, 0x07, 0x1F, 0x3F]  # 5 sub-steps per cell

    frames = []
    # Fill left → right
    for cell in range(WIDTH):
        for p in phases:
            f = {c: FULL for c in range(cell)}
            f[cell] = p
            frames.append(_frame(f))

    # Full bar
    frames.append(_frame({c: FULL for c in range(WIDTH)}))

    # Drain right → left
    for cell in range(WIDTH - 1, -1, -1):
        for p in reversed(phases):
            f = {c: FULL for c in range(cell)}
            f[cell] = p
            frames.append(_frame(f))

    return BrailleSpinner("fill_drain", frames, 4.0)


# ── 8. Pendulum ─────────────────────────────────────────────────────

def _pendulum() -> BrailleSpinner:
    """Oscillates around the center with a faint direction trail."""
    center = WIDTH / 2
    amp = (WIDTH - 2) / 2
    n = 48
    frames = []
    for i in range(n):
        angle = math.sin(2 * math.pi * i / n)
        pos = center + angle * amp
        cell = max(0, min(int(pos), WIDTH - 1))
        di = min(int((pos - cell) * len(_DOTS)), len(_DOTS) - 1)
        f: dict[int, int] = {cell: _DOTS[di]}
        # Faint trail in motion direction
        if angle > 0.1 and cell > 0:
            f[cell - 1] = 0x01
        elif angle < -0.1 and cell < WIDTH - 1:
            f[cell + 1] = 0x01
        frames.append(_frame(f))
    return BrailleSpinner("pendulum", frames, 2.0)


# ── 9. Phase-shifted spinner (liquid motion) ────────────────────────

def _liquid() -> BrailleSpinner:
    """Each cell is phase-shifted — looks like liquid flowing."""
    np = len(_SPIN)  # 10 phases
    frames = []
    for shift in range(np):
        row = "".join(_b(_SPIN[(cell + shift) % np]) for cell in range(WIDTH))
        frames.append(row)
    return BrailleSpinner("liquid", frames, 1.5)


# ── 10. Heartbeat ───────────────────────────────────────────────────

def _heartbeat() -> BrailleSpinner:
    """Pulse expands from center then collapses — liveness indicator."""
    center = WIDTH // 2
    max_r = center + 1

    frames = []
    # Expand
    for r in range(max_r + 1):
        f = {}
        for off in range(-r, r + 1):
            c = center + off
            if 0 <= c < WIDTH:
                dist = abs(off)
                bright = max(0, min(len(_FILL) - 1, r - dist))
                f[c] = _FILL[bright]
        frames.append(_frame(f))

    # Hold at peak
    frames.append(frames[-1])

    # Contract
    for r in range(max_r - 1, -1, -1):
        f = {}
        for off in range(-r, r + 1):
            c = center + off
            if 0 <= c < WIDTH:
                dist = abs(off)
                bright = max(0, min(len(_FILL) - 1, r - dist))
                f[c] = _FILL[bright]
        frames.append(_frame(f))

    return BrailleSpinner("heartbeat", frames, 1.5)


# ── Public API ───────────────────────────────────────────────────────

_FACTORIES = [
    _pingpong,
    _scanner,
    _comet,
    _elastic,
    _dual,
    _wave,
    _fill_drain,
    _pendulum,
    _liquid,
    _heartbeat,
]


def get_random_spinner() -> BrailleSpinner:
    """Return a randomly selected braille spinner."""
    return random.choice(_FACTORIES)()
