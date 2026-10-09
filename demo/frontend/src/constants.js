export const MAX_FEATURES = 800;

export const HIT_RADIUS         = 12;
export const POINT_RADIUS       = 3.2;
export const POINT_RADIUS_HOVER = 6;
export const HALO_RADIUS        = 10;
export const DIM_OPACITY        = 0.1;

export const COLOR_UNMATCHED = '#8a93a6';
export const COLOR_LINE      = '#ffd166';
export const GOLDEN_ANGLE    = 137.508;

export const pairColor = (i) => `hsl(${(i * GOLDEN_ANGLE) % 360}, 85%, 62%)`;