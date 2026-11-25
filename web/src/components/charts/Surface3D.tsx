// Interactive implied-volatility surface on a 2D canvas: drag to rotate (1:1 with the pointer, then it coasts
// with the release velocity and settles), arrow keys rotate too, hover reads the nearest grid point. It idles
// with a slow turn until touched; reduced motion keeps it still.
import { useEffect, useMemo, useRef, useState } from "react";
import { useReducedMotion, useWidth } from "@/components/cine";

const STOPS: [number, [number, number, number]][] = [
  [0, [76, 195, 201]],
  [0.5, [239, 164, 58]],
  [1, [242, 104, 60]],
];
function colour(t: number, a = 1) {
  const x = Math.max(0, Math.min(1, t));
  const i = x <= 0.5 ? 0 : 1;
  const [t0, c0] = STOPS[i], [t1, c1] = STOPS[i + 1];
  const f = (x - t0) / (t1 - t0);
  const c = c0.map((v, k) => Math.round(v + (c1[k] - v) * f));
  return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
}

export interface SurfaceGridData {
  k: (number | null)[];
  days: (number | null)[];
  iv: (number | null)[][];
}

type Hover = { k: number; days: number; iv: number; x: number; y: number } | null;

export function Surface3D({ grid, height = 460, label, ambient = false }: { grid: SurfaceGridData; height?: number; label: string; ambient?: boolean }) {
  const [wrap, w] = useWidth<HTMLDivElement>();
  const canvas = useRef<HTMLCanvasElement>(null);
  const reduce = useReducedMotion();
  const view = useRef({ yaw: -0.65, pitch: 0.52, vYaw: 0, vPitch: 0, touched: false });
  const drag = useRef<{ x: number; y: number; t: number; vx: number; vy: number } | null>(null);
  const [hover, setHover] = useState<Hover>(null);
  const projected = useRef<{ sx: number; sy: number; i: number; j: number }[]>([]);

  const norm = useMemo(() => {
    const ks = grid.k.map((v) => v ?? 0), ds = grid.days.map((v) => v ?? 0);
    const vals = grid.iv.flat().filter((v): v is number => v != null && Number.isFinite(v));
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const kMin = Math.min(...ks), kMax = Math.max(...ks), dMin = Math.min(...ds), dMax = Math.max(...ds);
    return { ks, ds, lo, hi, kMin, kMax, dMin, dMax };
  }, [grid]);

  useEffect(() => {
    const cv = canvas.current;
    if (!cv || !w) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = w * dpr;
    cv.height = height * dpr;
    const ctx = cv.getContext("2d")!;
    const css = getComputedStyle(cv);
    const fg = css.getPropertyValue("--muted-foreground").trim() || "#a7b1c4";
    const line = css.getPropertyValue("--line-strong").trim() || "#2a3854";
    const { ks, ds, lo, hi, kMin, kMax, dMin, dMax } = norm;
    const nx = ks.length, nz = ds.length;
    let raf = 0;

    const draw = () => {
      const v = view.current;
      if (!drag.current) {
        v.yaw += v.vYaw;
        v.pitch = Math.max(0.12, Math.min(1.25, v.pitch + v.vPitch));
        v.vYaw *= 0.94;
        v.vPitch *= 0.9;
        if (!v.touched && !reduce) v.yaw += 0.0016;
      }
      const s = Math.min(w * 0.3, height * 0.4);
      const cx = w / 2, cy = height * 0.66;
      const cyw = Math.cos(v.yaw), syw = Math.sin(v.yaw), cp = Math.cos(v.pitch), sp = Math.sin(v.pitch);
      const P = (x: number, y: number, z: number) => {
        const x1 = x * cyw - z * syw, z1 = x * syw + z * cyw;
        const y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
        const persp = 1 / (1 + z2 * 0.18);
        return { sx: cx + x1 * s * persp, sy: cy - y1 * s * persp, depth: z2 };
      };
      const X = (k: number) => ((k - kMin) / (kMax - kMin || 1)) * 2 - 1;
      const Z = (d: number) => ((d - dMin) / (dMax - dMin || 1)) * 2 - 1;
      const Y = (iv: number) => ((iv - lo) / (hi - lo || 1)) * 0.85;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, height);
      // floor and axes
      ctx.strokeStyle = line;
      ctx.lineWidth = 1;
      const corners = [P(-1, 0, -1), P(1, 0, -1), P(1, 0, 1), P(-1, 0, 1)];
      ctx.beginPath();
      corners.forEach((c, i) => (i ? ctx.lineTo(c.sx, c.sy) : ctx.moveTo(c.sx, c.sy)));
      ctx.closePath();
      ctx.stroke();
      for (let g = 1; g < 4; g++) {
        const t = -1 + g * 0.5;
        const a = P(t, 0, -1), b = P(t, 0, 1), c = P(-1, 0, t), d = P(1, 0, t);
        ctx.globalAlpha = 0.45;
        ctx.beginPath();
        ctx.moveTo(a.sx, a.sy);
        ctx.lineTo(b.sx, b.sy);
        ctx.moveTo(c.sx, c.sy);
        ctx.lineTo(d.sx, d.sy);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      // surface quads, back to front
      const pts: { sx: number; sy: number; depth: number }[][] = [];
      const flat: { sx: number; sy: number; i: number; j: number }[] = [];
      for (let i = 0; i < nz; i++) {
        pts[i] = [];
        for (let j = 0; j < nx; j++) {
          const iv = grid.iv[i]?.[j];
          const p = P(X(ks[j]), iv == null ? 0 : Y(iv), Z(ds[i]));
          pts[i][j] = p;
          if (iv != null) flat.push({ sx: p.sx, sy: p.sy, i, j });
        }
      }
      projected.current = flat;
      const quads: { d: number; i: number; j: number }[] = [];
      for (let i = 0; i < nz - 1; i++)
        for (let j = 0; j < nx - 1; j++) {
          if ([grid.iv[i][j], grid.iv[i + 1][j], grid.iv[i][j + 1], grid.iv[i + 1][j + 1]].some((x) => x == null)) continue;
          quads.push({ d: (pts[i][j].depth + pts[i + 1][j + 1].depth) / 2, i, j });
        }
      quads.sort((a, b) => b.d - a.d);
      for (const q of quads) {
        const { i, j } = q;
        const a = pts[i][j], b = pts[i][j + 1], c = pts[i + 1][j + 1], d = pts[i + 1][j];
        const iv = (grid.iv[i][j]! + grid.iv[i + 1][j + 1]!) / 2;
        const t = (iv - lo) / (hi - lo || 1);
        ctx.beginPath();
        ctx.moveTo(a.sx, a.sy);
        ctx.lineTo(b.sx, b.sy);
        ctx.lineTo(c.sx, c.sy);
        ctx.lineTo(d.sx, d.sy);
        ctx.closePath();
        ctx.fillStyle = colour(t, 0.82);
        ctx.fill();
        ctx.strokeStyle = colour(t, 1);
        ctx.globalAlpha = 0.35;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      if (ambient) {
        raf = requestAnimationFrame(draw);
        return;
      }
      // axis labels at the floor's edges
      ctx.fillStyle = fg;
      ctx.font = "11px 'IBM Plex Sans', sans-serif";
      const lbl = (p: { sx: number; sy: number }, text: string, align: CanvasTextAlign = "center") => {
        ctx.textAlign = align;
        ctx.fillText(text, p.sx, p.sy);
      };
      lbl(P(0, -0.12, -1.25), "Strike vs forward");
      lbl(P(-1, -0.08, -1.12), `${(Math.exp(kMin) * 100 - 100).toFixed(0)}%`);
      lbl(P(1, -0.08, -1.12), `+${(Math.exp(kMax) * 100 - 100).toFixed(0)}%`);
      lbl(P(1.25, -0.12, 0), "Days to expiry");
      lbl(P(1.14, -0.08, -1), `${dMin.toFixed(0)}d`);
      lbl(P(1.14, -0.08, 1), `${dMax.toFixed(0)}d`);
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [w, height, grid, norm, reduce, ambient]);

  const onDown = (e: React.PointerEvent) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, t: performance.now(), vx: 0, vy: 0 };
    view.current.touched = true;
    view.current.vYaw = view.current.vPitch = 0;
  };
  const onMove = (e: React.PointerEvent) => {
    const d = drag.current;
    const r = e.currentTarget.getBoundingClientRect();
    if (d) {
      const dx = e.clientX - d.x, dy = e.clientY - d.y, now = performance.now(), dt = Math.max(now - d.t, 1);
      view.current.yaw += dx * 0.008;
      view.current.pitch = Math.max(0.12, Math.min(1.25, view.current.pitch + dy * 0.006));
      d.vx = (dx * 0.008 * 16) / dt;
      d.vy = (dy * 0.006 * 16) / dt;
      d.x = e.clientX;
      d.y = e.clientY;
      d.t = now;
      setHover(null);
      return;
    }
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let best: (typeof projected.current)[number] | null = null, bd = 196;
    for (const p of projected.current) {
      const dd = (p.sx - mx) ** 2 + (p.sy - my) ** 2;
      if (dd < bd) {
        bd = dd;
        best = p;
      }
    }
    setHover(best ? { k: norm.ks[best.j], days: norm.ds[best.i], iv: grid.iv[best.i][best.j]!, x: best.sx, y: best.sy } : null);
  };
  const onUp = () => {
    const d = drag.current;
    if (d) {
      // hand the release velocity to the coast, so dragging and coasting have no seam
      view.current.vYaw = d.vx;
      view.current.vPitch = d.vy;
    }
    drag.current = null;
  };
  const onKey = (e: React.KeyboardEvent) => {
    const v = view.current;
    const map: Record<string, () => void> = {
      ArrowLeft: () => (v.yaw -= 0.15),
      ArrowRight: () => (v.yaw += 0.15),
      ArrowUp: () => (v.pitch = Math.max(0.12, v.pitch - 0.1)),
      ArrowDown: () => (v.pitch = Math.min(1.25, v.pitch + 0.1)),
    };
    if (map[e.key]) {
      e.preventDefault();
      v.touched = true;
      map[e.key]();
    }
  };

  if (ambient)
    return (
      <div ref={wrap} className="pointer-events-none relative h-full" aria-hidden>
        <canvas ref={canvas} style={{ width: w || "100%", height }} className="block" />
      </div>
    );
  return (
    <div ref={wrap} className="relative select-none">
      <canvas
        ref={canvas}
        role="img"
        tabIndex={0}
        aria-label={`${label}. Drag or use the arrow keys to rotate.`}
        style={{ width: w || "100%", height }}
        className="block cursor-grab touch-none outline-none focus-visible:ring-2 focus-visible:ring-saffron/70 active:cursor-grabbing"
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onUp}
        onPointerLeave={() => !drag.current && setHover(null)}
        onKeyDown={onKey}
      />
      {hover && (
        <div
          className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-[calc(100%+12px)] rounded-lg border border-line bg-panel/95 px-3 py-2 text-[12.5px] shadow-[0_10px_30px_-12px_rgb(0_0_0/0.6)] backdrop-blur"
          style={{ left: hover.x, top: hover.y }}
        >
          <b className="num text-foreground">{(hover.iv * 100).toFixed(1)}%</b> <span className="text-muted-foreground">implied vol</span>
          <br />
          <span className="text-muted-foreground">
            strike {hover.k >= 0 ? "+" : "−"}
            {Math.abs((Math.exp(hover.k) - 1) * 100).toFixed(1)}% vs forward, {hover.days.toFixed(0)} days
          </span>
        </div>
      )}
      <div className="mt-2 flex items-center justify-end gap-2 text-[11.5px] text-muted-foreground" aria-hidden>
        {(norm.lo * 100).toFixed(0)}%
        <span className="h-1.5 w-24 rounded-full" style={{ background: `linear-gradient(90deg, ${colour(0)}, ${colour(0.5)}, ${colour(1)})` }} aria-hidden />
        {(norm.hi * 100).toFixed(0)}%
      </div>
    </div>
  );
}
