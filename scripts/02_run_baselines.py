#!/usr/bin/env python3
"""
Ejecuta la escalera B0-B4 y produce la tabla comparativa.

Esta tabla es el entregable del prototipo: el piso que cualquier modelo
aprendido debe superar, medido sobre tus propios datos con tu propio código.

Uso:
    python scripts/02_run_baselines.py
    python scripts/02_run_baselines.py --width 44 --fps 5 --boot 1000

Salidas:
    reports/tables/ladder.csv
    reports/figures/ladder_<match>.png
    reports/figures/ghosts_<match>.png
"""
import argparse, csv, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from ghosting.io import Match  # noqa: E402
from ghosting.camera import simulate, ViewportConfig  # noqa: E402
from ghosting.baselines import run_ladder, LadderConfig, ALL_METHODS  # noqa: E402
from ghosting.metrics import evaluate  # noqa: E402
from ghosting.viz import plot_ladder, plot_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, TAB, FIG = ROOT/"data"/"processed", ROOT/"reports"/"tables", ROOT/"reports"/"figures"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--minutes", type=float, default=None,
                    help="recortar a los primeros N minutos "
                         "(usa 45 para replicar el protocolo de Choi 2026)")
    ap.add_argument("--width", type=float, default=44.0)
    ap.add_argument("--include-dead-ball", action="store_true",
                    help="incluir frames de balón parado. Necesario para "
                         "comparar con Choi (2026): usó Metrica, que no expone "
                         "ball_state, así que su protocolo los incluye "
                         "forzosamente")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--gk-anchor", choices=["team", "goal"], default="goal")
    ap.add_argument("--include-gk", action="store_true",
                    help="incluir porteros en las métricas")
    ap.add_argument("--tag", default=None,
                    help="sufijo del archivo de salida; por defecto se deriva "
                         "de la configuración para no pisar corridas previas")
    args = ap.parse_args()

    files = sorted(PROC.glob("*.npz"))
    if not files:
        print("No hay partidos en data/processed/. Corre antes scripts/00_download.py")
        return 1

    TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    cfg_l = LadderConfig(gk_anchor=args.gk_anchor)
    print(f"config: porteros {'INCLUIDOS' if args.include_gk else 'excluidos'} "
          f"en la métrica | ancla de portero: {args.gk_anchor} | "
          f"W={args.width:.0f} m"
          + (f" | primeros {args.minutes:.0f} min" if args.minutes else ""))
    tag = args.tag or (
        f"gk{'in' if args.include_gk else 'ex'}-{args.gk_anchor}"
        f"-W{int(args.width)}" + (f"-{int(args.minutes)}min" if args.minutes else "")
    )
    all_rows = []

    for f in files:
        m = Match.load(f)
        if m.fps > args.fps:
            m = m.resample(args.fps)
        if args.minutes:
            m = m.head_minutes(args.minutes)
        v = simulate(m, ViewportConfig(width_m=args.width))
        print(f"\n=== {m.match_id} | W={args.width:.0f} m | "
              f"{v.visible.sum(1).mean():.1f} visibles de {m.n_players} ===")
        hdr = (f"{'método':<7}{'n':>9}{'mediana':>9}{'IC 95%':>16}"
               f"{'≤2s':>8}{'2-9.6s':>9}{'>9.6s':>8}{'seg':>7}")
        print(hdr); print("-"*len(hdr))

        rows = []
        for mth in ALL_METHODS:
            if mth == "B0":
                continue
            t0 = time.time()
            est = run_ladder(m, v, mth, cfg_l)
            r = evaluate(m, v, est, mth, n_boot=args.boot, include_gk=args.include_gk,
                         alive_only=not args.include_dead_ball)
            r["elapsed_s"] = round(time.time()-t0, 1)
            rows.append(r)
            ci = f"[{r.get('median_ci_lo', float('nan')):.1f},{r.get('median_ci_hi', float('nan')):.1f}]"
            print(f"{mth:<7}{r['n']:>9,}{r['median_all']:>9.1f}{ci:>16}"
                  f"{r['median_<=2s']:>8.1f}{r['median_2-9.6s']:>9.1f}"
                  f"{r['median_>9.6s']:>8.1f}{r['elapsed_s']:>7.1f}")
            if mth == "B4" and "ci_lo_>9.6s" in r:
                print(f"{'':>7}{'IC 95% por bin:':>9}"
                      f"  ≤2s [{r['ci_lo_<=2s']:.1f},{r['ci_hi_<=2s']:.1f}]"
                      f"  2-9.6s [{r['ci_lo_2-9.6s']:.1f},{r['ci_hi_2-9.6s']:.1f}]"
                      f"  >9.6s [{r['ci_lo_>9.6s']:.1f},{r['ci_hi_>9.6s']:.1f}]")

        best = min(rows, key=lambda r: r["median_all"])
        print(f"\n  mejor: {best['method']} con {best['median_all']:.1f} m de mediana")
        print(f"  pesos: ≤2s {best['share_<=2s']:.0%} | "
              f"2-9.6s {best['share_2-9.6s']:.0%} | >9.6s {best['share_>9.6s']:.0%}")

        plot_ladder(rows, FIG/f"ladder_{m.match_id}_{tag}.png")
        est_b4 = run_ladder(m, v, "B4", cfg_l)
        plot_frame(m, v, est_b4, int(m.n_frames*0.35),
                   title=f"{m.match_id} — imputación B4 (voto de centroide)",
                   out=FIG/f"ghosts_{m.match_id}_{tag}.png")
        all_rows.extend(rows)

    out = TAB/f"ladder_{tag}.csv"
    keys = sorted({k for r in all_rows for k in r})
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(all_rows)
    print(f"\nTabla -> {out}\nFiguras -> {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
