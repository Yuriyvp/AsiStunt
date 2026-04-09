import { useRef, useEffect, useCallback } from 'react';

// Simplex-style 2D noise (compact implementation)
function createNoise() {
  const perm = new Uint8Array(512);
  const p = new Uint8Array(256);
  for (let i = 0; i < 256; i++) p[i] = i;
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [p[i], p[j]] = [p[j], p[i]];
  }
  for (let i = 0; i < 512; i++) perm[i] = p[i & 255];

  const grad2 = [[1,1],[-1,1],[1,-1],[-1,-1],[1,0],[-1,0],[0,1],[0,-1]];

  return function noise2D(x, y) {
    const xi = Math.floor(x) & 255;
    const yi = Math.floor(y) & 255;
    const xf = x - Math.floor(x);
    const yf = y - Math.floor(y);
    const u = xf * xf * (3 - 2 * xf);
    const v = yf * yf * (3 - 2 * yf);

    const aa = perm[perm[xi] + yi] & 7;
    const ab = perm[perm[xi] + yi + 1] & 7;
    const ba = perm[perm[xi + 1] + yi] & 7;
    const bb = perm[perm[xi + 1] + yi + 1] & 7;

    const dot = (g, dx, dy) => g[0] * dx + g[1] * dy;
    const x1 = dot(grad2[aa], xf, yf) * (1 - u) + dot(grad2[ba], xf - 1, yf) * u;
    const x2 = dot(grad2[ab], xf, yf - 1) * (1 - u) + dot(grad2[bb], xf - 1, yf - 1) * u;
    return x1 * (1 - v) + x2 * v;
  };
}

const STATES = {
  IDLE:        { speed: 0.3,  amplitude: 0.08, glow: 0.3, pulseSpeed: 1.5 },
  LISTENING:   { speed: 0.6,  amplitude: 0.15, glow: 0.5, pulseSpeed: 2.0 },
  PROCESSING:  { speed: 1.2,  amplitude: 0.12, glow: 0.6, pulseSpeed: 3.0 },
  SPEAKING:    { speed: 0.8,  amplitude: 0.18, glow: 0.7, pulseSpeed: 2.5 },
  INTERRUPTED: { speed: 2.0,  amplitude: 0.25, glow: 1.0, pulseSpeed: 5.0 },
};

const MOOD_COLORS = {
  neutral:   [68, 102, 204],
  warm:      [204, 136, 68],
  playful:   [204, 102, 102],
  calm:      [136, 153, 170],
  concerned: [204, 170, 68],
};

const ORB_LABELS = {
  IDLE: 'Assistant is idle',
  LISTENING: 'Assistant is listening',
  PROCESSING: 'Assistant is thinking',
  SPEAKING: 'Assistant is speaking',
  INTERRUPTED: 'Assistant was interrupted, now listening',
};

export default function Orb({ state = 'IDLE', mood = 'neutral', energy = 0, size = 180 }) {
  const canvasRef = useRef(null);
  const noiseRef = useRef(createNoise());
  const frameRef = useRef(null);
  const timeRef = useRef(0);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = size;
    const h = size;

    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      ctx.scale(dpr, dpr);
    }

    const cfg = STATES[state] || STATES.IDLE;
    const noise = noiseRef.current;
    const cx = w / 2;
    const cy = h / 2;
    const baseRadius = w * 0.3;

    timeRef.current += 1 / 30;
    const t = timeRef.current;

    ctx.clearRect(0, 0, w, h);

    // Mood color
    const [r, g, b] = MOOD_COLORS[mood] || MOOD_COLORS.neutral;

    // Outer glow
    const glowRadius = baseRadius * 1.8;
    const glowGrad = ctx.createRadialGradient(cx, cy, baseRadius * 0.5, cx, cy, glowRadius);
    const glowAlpha = cfg.glow * 0.15 + Math.sin(t * cfg.pulseSpeed) * 0.05;
    glowGrad.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${glowAlpha})`);
    glowGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = glowGrad;
    ctx.fillRect(0, 0, w, h);

    // Main orb shape with noise displacement
    const segments = 128;
    const amplitude = cfg.amplitude * baseRadius;
    const energyBoost = energy * 0.15 * baseRadius;

    ctx.beginPath();
    for (let i = 0; i <= segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      const nx = Math.cos(angle) * 2 + t * cfg.speed;
      const ny = Math.sin(angle) * 2 + t * cfg.speed * 0.7;
      const displacement = noise(nx, ny) * (amplitude + energyBoost);
      const pulse = Math.sin(t * cfg.pulseSpeed) * baseRadius * 0.02;
      const radius = baseRadius + displacement + pulse;

      const x = cx + Math.cos(angle) * radius;
      const y = cy + Math.sin(angle) * radius;

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();

    // Fill gradient
    const fillGrad = ctx.createRadialGradient(cx - baseRadius * 0.2, cy - baseRadius * 0.2, 0, cx, cy, baseRadius * 1.2);
    fillGrad.addColorStop(0, `rgba(${Math.min(255, r + 60)}, ${Math.min(255, g + 60)}, ${Math.min(255, b + 60)}, 0.9)`);
    fillGrad.addColorStop(0.5, `rgba(${r}, ${g}, ${b}, 0.7)`);
    fillGrad.addColorStop(1, `rgba(${Math.max(0, r - 40)}, ${Math.max(0, g - 40)}, ${Math.max(0, b - 40)}, 0.5)`);
    ctx.fillStyle = fillGrad;
    ctx.fill();

    // Inner highlight
    const innerGrad = ctx.createRadialGradient(cx - baseRadius * 0.15, cy - baseRadius * 0.2, 0, cx, cy, baseRadius * 0.6);
    innerGrad.addColorStop(0, 'rgba(255, 255, 255, 0.15)');
    innerGrad.addColorStop(1, 'rgba(255, 255, 255, 0)');
    ctx.fillStyle = innerGrad;
    ctx.fill();

    frameRef.current = requestAnimationFrame(draw);
  }, [state, mood, energy, size]);

  useEffect(() => {
    frameRef.current = requestAnimationFrame(draw);
    return () => {
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
    };
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label={ORB_LABELS[state] || 'Assistant'}
      style={{
        width: size,
        height: size,
        display: 'block',
        margin: '0 auto',
      }}
    />
  );
}
