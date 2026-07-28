#!/usr/bin/env python3
"""
Validación cruzada dejando un partido fuera (leave-one-match-out).

POR QUÉ ESTO Y NO "MÁS ÉPOCAS"
Con GPU sobra cómputo, y la tentación es subir épocas, ancho del modelo y
tamaño de lote. Pero el resultado actual —+1.32 m sobre B4— está medido en UN
SOLO partido de test. Su debilidad no es que el modelo esté poco entrenado:
es que no sabemos si ese partido fue afortunado.

Rotar qué partido hace de test produce siete estimaciones independientes de la
mejora, cada una sobre datos que ese modelo no vio nunca. Convierte

    "le ganamos a B4 en un partido de Bundesliga"

en

    "le ganamos a B4 en los siete, con delta agrupado X [IC]"

Eso es un salto de credibilidad que ninguna cantidad de épocas compra. Y con
~12 min por entrenamiento en una T4, las siete tandas caben en hora y media.

Protocolo
Para cada partido i de los siete:
    test  = i
    val   = el siguiente en orden circular  (nunca el test: sesgaría la
            selección de modelo hacia el conjunto de evaluación)
    train = los cinco restantes

Cada tanda entrena desde cero con su propia semilla y su propio checkpoint.
Al final, los errores pareados de las siete tandas se agrupan con
`pool_paired`, que respeta las fronteras entre partidos al remuestrear
bloques.

Uso:
    python scripts/07_cross_validate.py --epochs 100 --device cuda
    python scripts/07_cross_validate.py --epochs 100 --device cuda --long
"""
import argparse
import csv
import json
import subprocess
import sys
import time
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
PROC, TAB, CKPT = (ROOT / "data" / "processed", ROOT / "reports" / "tables",
                   ROOT / "reports" / "cv")

MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]

# Sustitución automática si solo hay datos sintéticos (prueba del pipeline).
if not list((ROOT / "data" / "processed").glob("*J03*")):
    MATCHES = sorted(p.stem.replace("synthetic_", "")
                     for p in (ROOT / "data" / "processed").glob("*SYNTH*"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--monitor", default=">9.6s")
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=len(MATCHES),
                    help="nº de tandas; menos de 7 para una prueba rápida")
    args = ap.parse_args()

    CKPT.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    filas, piezas, t_ini = [], [], time.time()

    for k in range(args.folds):
        test_id = MATCHES[k]
        val_id = MATCHES[(k + 1) % len(MATCHES)]
        train_ids = [m for m in MATCHES if m not in (test_id, val_id)]
        ck = CKPT / f"fold_{test_id}.pt"

        print(f"\n{'=' * 78}")
        print(f"  TANDA {k + 1}/{args.folds}   test={test_id}   val={val_id}")
        print(f"  train={' '.join(train_ids)}")
        print(f"{'=' * 78}", flush=True)

        cmd = [sys.executable, str(ROOT / "scripts" / "04_train.py"),
               "--train-ids", *train_ids, "--val-id", val_id,
               "--test-id", test_id, "--ckpt", str(ck),
               "--epochs", str(args.epochs), "--batch", str(args.batch),
               "--device", args.device, "--dim", str(args.dim),
               "--blocks", str(args.blocks), "--monitor", args.monitor,
               "--patience", str(args.patience), "--seed", str(1000 + k),
               "--no-bar"]
        if args.long:
            cmd.append("--long")
        if args.stride:
            cmd += ["--stride", str(args.stride)]
        subprocess.run(cmd, check=True)

        # --- evaluar esta tanda sobre SU partido de test ---
        ckd = torch.load(ck, map_location="cpu", weights_only=False)
        a = ckd["args"]
        model = ResidualImputer(dim=a["dim"], n_blocks=a["blocks"],
                                causal=not a["bidirectional"])
        model.load_state_dict(ckd["state_dict"])
        model.to(torch.device(args.device)).eval()

        m = Match.load(next(PROC.glob(f"*{test_id}*.npz")))
        if m.fps > a["fps"]:
            m = m.resample(a["fps"])
        if a["minutes"]:
            m = m.head_minutes(a["minutes"])
        view = simulate(m, ViewportConfig(width_m=a["width"]))
        b4 = run_ladder(m, view, "B4", LadderConfig())

        sys.path.insert(0, str(ROOT / "scripts"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ev", ROOT / "scripts" / "05_evaluate_model.py")
        ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev)
        ev.model_max_players = WindowConfig().max_players
        pred = ev.predict_match(model, m, view, b4, a["window"])

        s_b4 = stratified_median(position_errors(m, view, b4))
        s_md = stratified_median(position_errors(m, view, pred))
        pr = paired_position_errors(m, view, b4, pred)
        piezas.append(pr)
        ci = paired_block_bootstrap_ci(pr, m.fps, n_boot=args.boot)

        print(f"\n  --- {test_id} (test de esta tanda) ---")
        for key, lbl in [("global", "median_all"), ("<=2s", "median_<=2s"),
                         ("2-9.6s", "median_2-9.6s"), (">9.6s", "median_>9.6s")]:
            if key not in ci:
                continue
            d, lo, hi = ci[key]
            v = "MEJORA" if lo > 0 else "empeora" if hi < 0 else "nulo"
            print(f"  {key:<9}B4 {s_b4[lbl]:>6.2f} -> modelo {s_md[lbl]:>6.2f}"
                  f"   delta {d:>+6.2f} [{lo:>+6.2f},{hi:>+6.2f}]  {v}")
            filas.append({"fold": k + 1, "test_id": test_id, "bin": key,
                          "b4_m": s_b4[lbl], "modelo_m": s_md[lbl],
                          "delta_m": d, "ci_lo": lo, "ci_hi": hi,
                          "excluye_cero": bool(lo > 0 or hi < 0)})
        print(f"\n  [{(time.time() - t_ini) / 60:.0f} min transcurridos]", flush=True)

    # ------------------------------------------------------------------
    print(f"\n{'=' * 78}\n  RESULTADO AGRUPADO SOBRE {len(piezas)} PARTIDOS\n{'=' * 78}")
    pooled = pool_paired(piezas, 5.0)
    ci = paired_block_bootstrap_ci(pooled, 5.0, n_boot=args.boot)
    print(f"{'bin':<10}{'delta (m)':>11}{'IC 95% pareado':>22}   veredicto")
    print("-" * 62)
    for key, (d, lo, hi) in ci.items():
        v = "MEJORA CREIBLE" if lo > 0 else "empeora" if hi < 0 else "incluye 0"
        print(f"{key:<10}{d:>+11.2f}   [{lo:>+6.2f}, {hi:>+6.2f}]   {v}")
        filas.append({"fold": 0, "test_id": "AGRUPADO", "bin": key,
                      "b4_m": np.nan, "modelo_m": np.nan, "delta_m": d,
                      "ci_lo": lo, "ci_hi": hi,
                      "excluye_cero": bool(lo > 0 or hi < 0)})

    # Consistencia entre tandas: cuántas van en la misma dirección.
    print(f"\n{'bin':<10}{'tandas con mejora':>20}{'rango del delta':>24}")
    print("-" * 56)
    for key in ["global", "<=2s", "2-9.6s", ">9.6s"]:
        ds = [f["delta_m"] for f in filas if f["bin"] == key and f["fold"] > 0]
        if ds:
            print(f"{key:<10}{f'{sum(1 for d in ds if d > 0)}/{len(ds)}':>20}"
                  f"{f'[{min(ds):+.2f}, {max(ds):+.2f}]':>24}")

    out = TAB / "cross_validation.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader(); w.writerows(filas)
    print(f"\ntiempo total: {(time.time() - t_ini) / 60:.0f} min")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
