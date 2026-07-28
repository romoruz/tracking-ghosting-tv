#!/usr/bin/env python3
"""
Test externo congelado: modelos entrenados en Bundesliga, evaluados en Metrica.

Metrica no se ha tocado en ningún momento del desarrollo. Otro proveedor
(Metrica Sports frente a TRACAB gen-5), otra liga, otro año. Los siete modelos
de la validación cruzada solo han visto partidos de Bundesliga.

Si le ganan a B4 aquí, el resultado deja de ser "funciona en nuestros datos" y
pasa a ser "generaliza".

POR QUÉ NO SE AGRUPAN LAS 14 EVALUACIONES
Siete modelos por dos partidos son catorce mediciones, y la tentación es
meterlas todas en un bootstrap agrupado. Sería un error: los mismos frames de
Metrica aparecerían siete veces, así que el intervalo saldría
artificialmente estrecho por multiplicar observaciones que no son
independientes.

Se hace de dos formas, y ambas se reportan:

1. POR TANDA. Cada modelo se evalúa por separado en cada partido. Catorce
   deltas con su intervalo. Si los catorce son positivos, eso es convincente
   sin necesidad de agrupar nada. La dispersión entre tandas es, además, la
   medida honesta de cuánto depende el resultado de qué partidos tocó entrenar.

2. CONJUNTO (ensemble). Se promedian las predicciones de los siete modelos y
   se evalúa UNA vez por partido. Ahí sí se pueden agrupar los dos partidos,
   porque cada frame entra una sola vez. Es además lo que se desplegaría en
   producción.

NOTA DE PROTOCOLO
Metrica no expone `ball_state`, así que su evaluación incluye balón parado por
necesidad. Los números de Sportec se calcularon filtrándolo. No son
directamente comparables entre conjuntos; sí lo son dentro de cada uno, que es
lo que importa para medir modelo contra B4.

Uso:
    python scripts/08_external_test.py --device cuda
"""
import argparse
import csv
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ghosting.io import Match  # noqa: E402
from ghosting.camera import simulate, ViewportConfig  # noqa: E402
from ghosting.baselines import run_ladder, LadderConfig  # noqa: E402
from ghosting.models import ResidualImputer, WindowConfig  # noqa: E402
from ghosting.metrics import (  # noqa: E402
    paired_position_errors, paired_block_bootstrap_ci, pool_paired,
    position_errors, stratified_median,
)

ROOT = Path(__file__).resolve().parents[1]
PROC, TAB, CV = (ROOT / "data" / "processed", ROOT / "reports" / "tables",
                 ROOT / "reports" / "cv")
BINS = [("global", "median_all"), ("<=2s", "median_<=2s"),
        ("2-9.6s", "median_2-9.6s"), (">9.6s", "median_>9.6s")]


def cargar_evaluador():
    """Reutiliza `predict_match` del script 05 sin duplicar código."""
    spec = importlib.util.spec_from_file_location(
        "ev", ROOT / "scripts" / "05_evaluate_model.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    ev.model_max_players = WindowConfig().max_players
    return ev


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--matches", nargs="+", default=["metrica_1", "metrica_2"])
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    ckpts = sorted(CV.glob("fold_*.pt"))
    if not ckpts:
        print(f"No hay checkpoints en {CV}. Corre antes 07_cross_validate.py")
        return 1
    print(f"{len(ckpts)} modelos de validación cruzada encontrados\n")

    dev = torch.device(args.device)
    ev = cargar_evaluador()
    filas = []

    for stem in args.matches:
        hits = list(PROC.glob(f"*{stem}*.npz"))
        if not hits:
            print(f"[aviso] no encontrado: {stem}")
            continue

        print(f"{'=' * 78}\n  {stem}  (nunca visto en entrenamiento ni en desarrollo)")
        print(f"{'=' * 78}")

        base_m = view = b4 = None
        preds = []

        for ck_path in ckpts:
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            a = ck["args"]
            model = ResidualImputer(dim=a["dim"], n_blocks=a["blocks"],
                                    causal=not a["bidirectional"])
            model.load_state_dict(ck["state_dict"])
            model.to(dev).eval()

            if base_m is None:
                m = Match.load(hits[0])
                if m.fps > a["fps"]:
                    m = m.resample(a["fps"])
                if a["minutes"]:
                    m = m.head_minutes(a["minutes"])
                view = simulate(m, ViewportConfig(width_m=a["width"]))
                b4 = run_ladder(m, view, "B4", LadderConfig())
                base_m = m
                s_b4 = stratified_median(position_errors(m, view, b4))
                print(f"\n  B4 en este partido:  " + "  ".join(
                    f"{k} {s_b4[lbl]:.2f}" for k, lbl in BINS))
                print(f"\n  {'tanda':<10}" + "".join(f"{k:>16}" for k, _ in BINS))
                print("  " + "-" * 74)

            pred = ev.predict_match(model, base_m, view, b4, a["window"])
            preds.append(pred)

            pr = paired_position_errors(base_m, view, b4, pred)
            ci = paired_block_bootstrap_ci(pr, base_m.fps, n_boot=args.boot)
            fold = ck_path.stem.replace("fold_", "")
            linea = f"  {fold:<10}"
            for k, _ in BINS:
                if k in ci:
                    d, lo, hi = ci[k]
                    marca = "*" if lo > 0 else "!" if hi < 0 else " "
                    linea += f"{f'{d:+.2f}{marca}':>16}"
                    filas.append({"partido": stem, "modelo": fold, "bin": k,
                                  "delta_m": d, "ci_lo": lo, "ci_hi": hi,
                                  "excluye_cero": bool(lo > 0 or hi < 0)})
            print(linea)

        # --- Conjunto: promedio de las predicciones de las siete tandas ---
        ens = np.nanmean(np.stack(preds), axis=0)
        s_ens = stratified_median(position_errors(base_m, view, ens))
        s_b4 = stratified_median(position_errors(base_m, view, b4))
        pr_ens = paired_position_errors(base_m, view, b4, ens)
        ci_ens = paired_block_bootstrap_ci(pr_ens, base_m.fps, n_boot=args.boot)

        print("  " + "-" * 74)
        linea = f"  {'CONJUNTO':<10}"
        for k, _ in BINS:
            if k in ci_ens:
                d, lo, hi = ci_ens[k]
                marca = "*" if lo > 0 else "!" if hi < 0 else " "
                linea += f"{f'{d:+.2f}{marca}':>16}"
        print(linea)
        print(f"\n  conjunto:  " + "  ".join(
            f"{k} {s_b4[lbl]:.2f}->{s_ens[lbl]:.2f}" for k, lbl in BINS))

        for k, _ in BINS:
            if k in ci_ens:
                d, lo, hi = ci_ens[k]
                filas.append({"partido": stem, "modelo": "CONJUNTO", "bin": k,
                              "delta_m": d, "ci_lo": lo, "ci_hi": hi,
                              "excluye_cero": bool(lo > 0 or hi < 0)})
        # Guardado para el agrupado final entre partidos
        filas[-1]["_pr"] = None
        globals().setdefault("_piezas", []).append(pr_ens)
        print()

    # ------------------------------------------------------------------
    piezas = globals().get("_piezas", [])
    if len(piezas) > 1:
        print(f"{'=' * 78}")
        print("  CONJUNTO AGRUPADO SOBRE LOS PARTIDOS DE METRICA")
        print(f"{'=' * 78}")
        pooled = pool_paired(piezas, 5.0)
        ci = paired_block_bootstrap_ci(pooled, 5.0, n_boot=args.boot)
        print(f"{'bin':<10}{'delta (m)':>11}{'IC 95% pareado':>22}   veredicto")
        print("-" * 62)
        for k, (d, lo, hi) in ci.items():
            v = "MEJORA CREIBLE" if lo > 0 else "empeora" if hi < 0 else "incluye 0"
            print(f"{k:<10}{d:>+11.2f}   [{lo:>+6.2f}, {hi:>+6.2f}]   {v}")
            filas.append({"partido": "AGRUPADO", "modelo": "CONJUNTO", "bin": k,
                          "delta_m": d, "ci_lo": lo, "ci_hi": hi,
                          "excluye_cero": bool(lo > 0 or hi < 0)})

    # Consistencia: cuántas de las evaluaciones individuales son positivas.
    ind = [f for f in filas if f["modelo"] != "CONJUNTO"]
    if ind:
        print(f"\n{'bin':<10}{'evaluaciones con mejora':>26}{'rango del delta':>24}")
        print("-" * 62)
        for k, _ in BINS:
            ds = [f["delta_m"] for f in ind if f["bin"] == k]
            cr = [f for f in ind if f["bin"] == k and f["excluye_cero"]
                  and f["delta_m"] > 0]
            if ds:
                print(f"{k:<10}{f'{sum(1 for d in ds if d > 0)}/{len(ds)}'
                      f' ({len(cr)} creíbles)':>26}"
                      f"{f'[{min(ds):+.2f}, {max(ds):+.2f}]':>24}")

    TAB.mkdir(parents=True, exist_ok=True)
    out = TAB / "external_test_metrica.csv"
    keys = [k for k in filas[0] if not k.startswith("_")]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(filas)
    print(f"\n  * = IC excluye 0 (mejora creíble)   ! = empeora creíblemente")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
