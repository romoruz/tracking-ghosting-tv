#!/usr/bin/env python3
"""
Evalúa el modelo entrenado CONTRA B4, con el protocolo ya validado.

El número que importa no es el error del modelo en abstracto, sino el delta
pareado sobre B4: mismos frames, mismos jugadores, misma cámara. Se reutiliza
exactamente la maquinaria del benchmark (bootstrap de bloques pareado,
estratificación por gap) para que el resultado sea comparable con todo lo
anterior.

Reensamblado de ventanas
El modelo ve ventanas de T frames. Para reconstruir el partido completo se usan
ventanas solapadas al 50% y se toma de cada una **solo su segunda mitad**: así
todo frame evaluado tiene al menos T/2 frames de contexto por detrás. Tomar la
primera mitad mezclaría predicciones hechas casi sin historia con predicciones
bien informadas, y en modo causal eso infla el error de forma artificial.

Uso:
    python scripts/05_evaluate_model.py
    python scripts/05_evaluate_model.py --checkpoint reports/imputer_bidir.pt
    python scripts/05_evaluate_model.py --matches metrica_1 metrica_2   # test externo
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ghosting.io import Match  # noqa: E402
from ghosting.camera import simulate, ViewportConfig  # noqa: E402
from ghosting.baselines import run_ladder, LadderConfig  # noqa: E402
from ghosting.models import (  # noqa: E402
    build_features, WindowConfig, ResidualImputer,
)
from ghosting.models.dataset import pad_players  # noqa: E402
from ghosting.metrics import (  # noqa: E402
    paired_position_errors, paired_block_bootstrap_ci, pool_paired,
    position_errors, stratified_median,
)

ROOT = Path(__file__).resolve().parents[1]
PROC, TAB = ROOT / "data" / "processed", ROOT / "reports" / "tables"


@torch.no_grad()
def predict_match(model, match, view, baseline, length: int) -> np.ndarray:
    """
    Predicción del modelo sobre un partido completo, reensamblada por ventanas.

    Devuelve (T, N, 2). Los frames sin predicción (arranque) conservan B4.
    """
    per = build_features(match, view, baseline)
    T, N = per["loss_mask"].shape
    out = baseline.copy()
    half = length // 2

    for s in range(0, T - length + 1, half):
        e = s + length
        w = pad_players({
            "feats": per["feats"][s:e], "target": per["target"][s:e],
            "loss_mask": per["loss_mask"][s:e], "base": per["base"][s:e],
        }, model_max_players)
        # Índices originales de los jugadores conservados por pad_players
        on = per["feats"][s:e, :, -1].max(axis=0) > 0
        score = per["loss_mask"][s:e].sum(axis=0) + on * 1e6
        keep = np.sort(np.argsort(-score)[:model_max_players])

        dev = next(model.parameters()).device
        b = {k: torch.from_numpy(v[None]).to(dev) for k, v in w.items()}
        pred = model.predict_positions(b["feats"], b["player_mask"], b["base"])
        pred = pred[0].cpu().numpy()

        # Solo la segunda mitad: garantiza >= T/2 frames de contexto.
        lo = s if s == 0 else s + half
        off = 0 if s == 0 else half
        out[lo:e, keep[: min(len(keep), model_max_players)]] = \
            pred[off:, : len(keep)]
    return out


def main():
    global model_max_players
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="reports/imputer_causal.pt")
    ap.add_argument("--matches", nargs="+", default=["J03WR9"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--include-dead-ball", action="store_true")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    ck = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = ResidualImputer(dim=a["dim"], n_blocks=a["blocks"],
                            causal=not a["bidirectional"])
    model.load_state_dict(ck["state_dict"])
    model.to(torch.device(args.device)).eval()
    model_max_players = WindowConfig().max_players

    modo = "BIDIRECCIONAL (interpolación)" if a["bidirectional"] else "CAUSAL (online)"
    print(f"checkpoint: {args.checkpoint} | modo {modo}")
    print(f"error en validación durante el entrenamiento: {ck['val_err_m']:.2f} m "
          f"(B4: {ck['b4_val_err_m']:.2f} m)\n")

    alive = not args.include_dead_ball
    filas, piezas = [], []

    for stem in args.matches:
        hits = list(PROC.glob(f"*{stem}*.npz"))
        if not hits:
            print(f"[aviso] no encontrado: {stem}")
            continue
        m = Match.load(hits[0])
        if m.fps > a["fps"]:
            m = m.resample(a["fps"])
        if a["minutes"]:
            m = m.head_minutes(a["minutes"])
        view = simulate(m, ViewportConfig(width_m=a["width"]))
        b4 = run_ladder(m, view, "B4", LadderConfig())
        pred = predict_match(model, m, view, b4, a["window"])

        s_b4 = stratified_median(position_errors(m, view, b4, alive_only=alive))
        s_md = stratified_median(position_errors(m, view, pred, alive_only=alive))
        pr = paired_position_errors(m, view, b4, pred, alive_only=alive)
        piezas.append(pr)
        ci = paired_block_bootstrap_ci(pr, m.fps, n_boot=args.boot)

        print(f"=== {m.match_id} ===")
        print(f"{'bin':<10}{'B4':>8}{'modelo':>9}{'delta':>9}{'IC 95% pareado':>21}   veredicto")
        print("-" * 76)
        for k, lbl in [("global", "median_all"), ("<=2s", "median_<=2s"),
                       ("2-9.6s", "median_2-9.6s"), (">9.6s", "median_>9.6s")]:
            if k not in ci:
                continue
            d, lo, hi = ci[k]
            v = ("MEJORA CREIBLE" if lo > 0 else "empeora" if hi < 0 else "incluye 0")
            print(f"{k:<10}{s_b4[lbl]:>8.2f}{s_md[lbl]:>9.2f}{d:>+9.2f}"
                  f"   [{lo:>+6.2f}, {hi:>+6.2f}]   {v}")
            filas.append({"match_id": m.match_id, "bin": k,
                          "b4_m": s_b4[lbl], "modelo_m": s_md[lbl],
                          "delta_m": d, "ci_lo": lo, "ci_hi": hi,
                          "excluye_cero": bool(lo > 0 or hi < 0)})
        print()

    if len(piezas) > 1:
        pooled = pool_paired(piezas, a["fps"])
        ci = paired_block_bootstrap_ci(pooled, a["fps"], n_boot=args.boot)
        print("=== AGRUPADO ===")
        for k, (d, lo, hi) in ci.items():
            v = ("MEJORA CREIBLE" if lo > 0 else "empeora" if hi < 0 else "incluye 0")
            print(f"{k:<10}{'':>17}{d:>+9.2f}   [{lo:>+6.2f}, {hi:>+6.2f}]   {v}")
            filas.append({"match_id": "AGRUPADO", "bin": k, "b4_m": np.nan,
                          "modelo_m": np.nan, "delta_m": d, "ci_lo": lo,
                          "ci_hi": hi, "excluye_cero": bool(lo > 0 or hi < 0)})

    print("\n  delta = mediana(B4) - mediana(modelo).  Positivo => el modelo mejora.")
    if filas:
        TAB.mkdir(parents=True, exist_ok=True)
        out = TAB / "model_vs_b4.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0]))
            w.writeheader(); w.writerows(filas)
        print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
