#!/usr/bin/env python3
"""
Ablación del portero: resuelve la ambigüedad de protocolo y MIDE la mejora.

Choi (2026) no especifica si incluye porteros en la métrica de error de
posición, ni cómo los ancla. El hueco importa: el portero vive en un extremo de
la cancha, así que la cámara lo pierde mucho más que a un jugador de campo y
aporta ~2.5x más muestras ocultas que un jugador promedio, casi todas en el
régimen de oclusión larga.

Tres configuraciones
--------------------
A) porteros incluidos, ancla de equipo   -> réplica literal de la escalera publicada
B) porteros incluidos, ancla de portería -> partición del portero (contribución propia)
C) porteros excluidos, ancla de portería -> métrica limpia de jugadores de campo

CÓMO SE DECIDE QUÉ CONFIGURACIÓN COMPARAR CON EL PAPER
------------------------------------------------------
NO por cuál da mejor error: eso sería tunear contra el conjunto de comparación.
Se decide por el PESO DE LOS BINS, que es una propiedad estructural de la
población evaluada y no una medida de rendimiento. Si el reparto
<=2s / 2-9.6s / >9.6s no coincide con el publicado, no se está midiendo sobre
la misma población y ninguna comparación de error significa nada, gane quien
gane. Choi reporta 50-57% de las muestras ocultas en el bin >9.6 s.

POR QUÉ EL DELTA A->B SE MIDE PAREADO
-------------------------------------
A y B se evalúan sobre exactamente los mismos frames, jugadores y cámara; lo
único que cambia es el ancla del portero. Comparar sus intervalos marginales y
concluir "se solapan, no hay diferencia" es un error conservador hasta la
inutilidad: la mayor parte de esa incertidumbre es común a ambos y se cancela
al restar. El intervalo se calcula sobre la DIFERENCIA, remuestreando bloques y
restando dentro de cada réplica. Es el mismo procedimiento que usa Choi para
contrastar B4 contra B2.

Uso:
    python scripts/03_gk_ablation.py --minutes 45 --boot 1000
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ghosting.io import Match  # noqa: E402
from ghosting.camera import simulate, ViewportConfig  # noqa: E402
from ghosting.baselines import run_ladder, LadderConfig, ALL_METHODS  # noqa: E402
from ghosting.metrics import (  # noqa: E402
    evaluate, paired_position_errors, paired_block_bootstrap_ci, pool_paired,
)

ROOT = Path(__file__).resolve().parents[1]
PROC, TAB = ROOT / "data" / "processed", ROOT / "reports" / "tables"

CONFIGS = [
    ("A_replica_literal", True, "team"),
    ("B_particion_portero", True, "goal"),
    ("C_solo_campo", False, "goal"),
]

CHOI = {
    "median_all": (9.7, 11.6),
    "median_<=2s": (3.3, 3.7),
    "median_2-9.6s": (7.2, 8.9),
    "median_>9.6s": (15.6, 16.9),
    "share_>9.6s": (0.50, 0.57),
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--minutes", type=float, default=45.0)
    ap.add_argument("--include-dead-ball", action="store_true",
                    help="incluir frames de balón parado. Necesario para "
                         "comparar con Choi (2026): usó Metrica, que no expone "
                         "ball_state, así que su protocolo los incluye "
                         "forzosamente")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--width", type=float, default=44.0)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--method", default="B4", choices=ALL_METHODS)
    ap.add_argument("--only", default="metrica",
                    help="prefijo de match_id a analizar; '' para todos")
    args = ap.parse_args()

    files = sorted(f for f in PROC.glob("*.npz") if f.stem.startswith(args.only))
    if not files:
        print(f"No hay partidos '{args.only}*' en data/processed/")
        return 1

    TAB.mkdir(parents=True, exist_ok=True)
    rows, deltas, piezas = [], [], []

    for f in files:
        m = Match.load(f)
        if m.fps > args.fps:
            m = m.resample(args.fps)
        if args.minutes:
            m = m.head_minutes(args.minutes)
        view = simulate(m, ViewportConfig(width_m=args.width))

        print(f"\n=== {m.match_id} | W={args.width:.0f} m | "
              f"{m.n_frames:,} frames @ {m.fps:g} fps ===", flush=True)

        est = {}
        for label, inc_gk, anchor in CONFIGS:
            est[label] = run_ladder(m, view, args.method,
                                    LadderConfig(gk_anchor=anchor))
            r = evaluate(m, view, est[label], args.method,
                         n_boot=args.boot, include_gk=inc_gk,
                         alive_only=not args.include_dead_ball)
            r["config"] = label
            rows.append(r)

        pr = paired_position_errors(
            m, view, est["A_replica_literal"], est["B_particion_portero"],
            include_gk=True, alive_only=not args.include_dead_ball,
        )
        piezas.append(pr)
        fps_comun = m.fps
        ci = paired_block_bootstrap_ci(pr, m.fps, n_boot=args.boot)
        for k, (d, lo, hi) in ci.items():
            deltas.append({
                "match_id": m.match_id, "bin": k, "delta_m": d,
                "ci_lo": lo, "ci_hi": hi,
                "excluye_cero": bool(lo > 0 or hi < 0),
                "n": int(pr["err_a"].size),
            })

    print(f"\n{'=' * 92}\n  {args.method} vs Choi (2026)\n{'=' * 92}")
    hdr = (f"{'config':<22}{'partido':<12}{'n':>9}{'mediana':>9}"
           f"{'<=2s':>8}{'2-9.6s':>9}{'>9.6s':>8}{'peso>9.6s':>12}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def mark(k, r=r):
            v = float(r[k])
            lo, hi = CHOI[k]
            txt = f"{v:.1%}" if k.startswith("share") else f"{v:.1f}"
            return txt + ("*" if lo <= v <= hi else " ")
        print(f"{r['config']:<22}{r['match_id']:<12}{r['n']:>9,}"
              f"{mark('median_all'):>9}{mark('median_<=2s'):>8}"
              f"{mark('median_2-9.6s'):>9}{mark('median_>9.6s'):>8}"
              f"{mark('share_>9.6s'):>12}")
    print("\n  * = dentro del rango publicado")
    print("  Lee primero 'peso>9.6s': es estructural, no de rendimiento. La")
    print("  configuracion cuyo peso cae en 50.0-57.0% evalua sobre la misma")
    print("  poblacion que el paper; solo dentro de ella son comparables los errores.")

    print(f"\n{'=' * 92}")
    print("  CONTRIBUCION: particion del portero  (A ancla de equipo -> B ancla de porteria)")
    print(f"{'=' * 92}")
    hdr2 = f"{'partido':<12}{'bin':<10}{'delta (m)':>11}{'IC 95% pareado':>22}   veredicto"
    print(hdr2)
    print("-" * (len(hdr2) + 8))
    for d in deltas:
        v = ("MEJORA CREIBLE" if d["excluye_cero"] and d["delta_m"] > 0
             else "empeora" if d["excluye_cero"] else "incluye 0")
        print(f"{d['match_id']:<12}{d['bin']:<10}{d['delta_m']:>+11.2f}"
              f"   [{d['ci_lo']:>+6.2f}, {d['ci_hi']:>+6.2f}]   {v}")
    
    if len(piezas) > 1:
        pooled = pool_paired(piezas, fps_comun)
        ci_p = paired_block_bootstrap_ci(pooled, fps_comun, n_boot=args.boot)
        print("-" * (len(hdr2) + 8))
        for k, (d, lo, hi) in ci_p.items():
            v = ("MEJORA CREIBLE" if (lo > 0 or hi < 0) and d > 0
                 else "empeora" if (lo > 0 or hi < 0) else "incluye 0")
            print(f"{'AGRUPADO':<12}{k:<10}{d:>+11.2f}"
                  f"   [{lo:>+6.2f}, {hi:>+6.2f}]   {v}")
            deltas.append({
                "match_id": "AGRUPADO", "bin": k, "delta_m": d,
                "ci_lo": lo, "ci_hi": hi,
                "excluye_cero": bool(lo > 0 or hi < 0),
                "n": int(pooled["err_a"].size),
            })

    print("\n  delta = mediana(A) - mediana(B).  Positivo => B es mejor.")
    print("  Pareado: mismos frames, jugadores y camara; solo cambia el ancla.")
    print("  Si el intervalo excluye 0, la mejora es creible al 95%.")

    for name, data in (("gk_ablation.csv", rows), ("gk_ablation_deltas.csv", deltas)):
        if not data:
            continue
        keys = sorted({k for r in data for k in r})
        with open(TAB / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(data)
        print(f"\n-> {TAB / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
