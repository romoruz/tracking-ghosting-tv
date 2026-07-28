#!/usr/bin/env python3
"""
Valida el simulador de cámara y produce el reporte de oclusión.

Es la primera verificación del proyecto: con W=44 m el viewport debe mostrar
entre 14 y 16 jugadores de 22. Si no, las coordenadas o el simulador están mal
y todo lo que venga después será ficción.

Uso:
    python scripts/01_viewport_report.py
    python scripts/01_viewport_report.py --fps 5 --widths 36 44 52 60

Salidas:
    reports/tables/viewport_stats.csv
    reports/figures/occlusion_<match>.png
    reports/figures/frame_<match>.png
"""
import argparse, csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from ghosting.io import Match  # noqa: E402
from ghosting.camera import simulate, occlusion_stats, ViewportConfig  # noqa: E402
from ghosting.viz import plot_occlusion_histogram, plot_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, TAB, FIG = ROOT/"data"/"processed", ROOT/"reports"/"tables", ROOT/"reports"/"figures"

# Referencia publicada (Choi 2026, tabla 2, media sobre Metrica g1-g3)
REF = {36: 13.1, 44: 14.8, 52: 16.3, 60: 17.4}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--minutes", type=float, default=None,
                    help="recortar a los primeros N minutos "
                         "(usa 45 para replicar el protocolo de Choi 2026)")
    ap.add_argument("--widths", type=float, nargs="+", default=[36, 44, 52, 60])
    ap.add_argument("--figures", action="store_true", default=True)
    args = ap.parse_args()

    files = sorted(PROC.glob("*.npz"))
    if not files:
        print("No hay partidos en data/processed/. Corre antes scripts/00_download.py")
        return 1

    TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    rows = []

    for f in files:
        m = Match.load(f)
        if m.fps > args.fps:
            m = m.resample(args.fps)
        if args.minutes:
            m = m.head_minutes(args.minutes)
        print(f"\n=== {m.match_id} [{m.provider}] "
              f"{m.n_frames:,} frames @ {m.fps:g} fps ===")
        print(f"{'W (m)':>6} {'visibles':>10} {'±':>6} {'≤2s':>7} {'2-9.6s':>8} "
              f"{'>9.6s':>7} {'gap med':>9}   referencia")
        for W in args.widths:
            v = simulate(m, ViewportConfig(width_m=float(W)))
            s = occlusion_stats(m, v)
            ref = REF.get(int(W))
            flag = ""
            if ref:
                d = s["visible_mean"] - ref
                flag = f"{ref:.1f} (Δ{d:+.1f})" + ("  OK" if abs(d) < 1.6 else "  <-- revisar")
            print(f"{W:>6.0f} {s['visible_mean']:>10.1f} {s['visible_std']:>6.1f} "
                  f"{s['hidden_share_le2s']:>7.0%} {s['hidden_share_2_96s']:>8.0%} "
                  f"{s['hidden_share_gt96s']:>7.0%} {s['gap_median_s']:>8.1f}s   {flag}")
            rows.append({"match_id": m.match_id, "provider": m.provider,
                         "fps": m.fps, "width_m": W, **s})
            if args.figures and int(W) == 44:
                plot_occlusion_histogram(m, v, FIG/f"occlusion_{m.match_id}.png")
                t = int(m.n_frames*0.35)
                plot_frame(m, v, None, t, out=FIG/f"frame_{m.match_id}.png")

    out = TAB/"viewport_stats.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nTabla -> {out}")
    print(f"Figuras -> {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
