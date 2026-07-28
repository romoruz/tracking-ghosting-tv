#!/usr/bin/env python3
"""
Figura estática: error de imputación desglosado por ROL ESTIMADO.

Qué cuenta
La cifra global "el modelo baja el error de 7.9 a 5.6 m" no dice DÓNDE está la
ganancia ni qué roles siguen siendo difíciles. Esta figura lo abre por rol:
para cada rol estimado, la mediana del error de B4 frente a la del modelo, sobre
exactamente la misma población (comparación pareada), con su mejora y el
intervalo de confianza pareado por block bootstrap.

Es la imagen que respalda dos frases del pitch:
  - "el modelo mejora de forma transversal, no solo en un tipo de jugador";
  - y, si aparece, el hallazgo honesto de que el PORTERO y los DEFENSAS ABIERTOS
    (laterales) son los más difíciles, porque son los que la cámara pierde más
    tiempo y los que peor describe el centroide de equipo de B4.

Los roles NO vienen de metadatos: se infieren por geometría sobre los frames
visibles (ver src/ghosting/roles/infer.py). Por eso el título dice
explícitamente "(inferencia geométrica)": el análisis opera sobre tracking crudo
y es agnóstico a la formación privada del proveedor.

Rigor
- MEDIANA, no media (cola derecha pesada), igual que todo el proyecto.
- El delta por rol se mide PAREADO (mismos frames, jugadores y cámara; solo
  cambia el método) y con block bootstrap de bloques de 1 min, reutilizando la
  misma maquinaria ya validada de metrics/position.py. Comparar dos barras
  marginales y "ver si se solapan" sería el error conservador que ese módulo
  existe para evitar.
- Se muestra n por rol: un rol con pocas muestras ocultas tiene un delta ruidoso
  y hay que leerlo como tal.

Uso:
    python scripts/11_role_error_figure.py --match J03WR9
    python scripts/11_role_error_figure.py --match metrica_1 --bin ">9.6s"
    python scripts/11_role_error_figure.py --match J03WR9 --checkpoint reports/cv/fold_J03WR9.pt
"""
import argparse
import csv
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import torch                             # noqa: E402

from ghosting.io import Match                                       # noqa: E402
from ghosting.camera import simulate, ViewportConfig                # noqa: E402
from ghosting.baselines import run_ladder, LadderConfig            # noqa: E402
from ghosting.models import ResidualImputer, WindowConfig          # noqa: E402
from ghosting.metrics import paired_block_bootstrap_ci             # noqa: E402
from ghosting.metrics.position import BIN_EDGES, BIN_LABELS        # noqa: E402
from ghosting.roles import (                                        # noqa: E402
    infer_all_roles, ROLE_COLORS, ROLE_LABEL, ROLE_ORDER,
)

ROOT = Path(__file__).resolve().parents[1]
PROC, TAB, FIG = (ROOT / "data" / "processed", ROOT / "reports" / "tables",
                  ROOT / "reports" / "figures")

C_B4 = "#9aa3ad"        # baseline: gris tenue (misma paleta que la animación)


def load_predict_match():
    """
    Reutiliza el predict_match del script 10 (animación).

    IMPORTANTE: se toma del 10 y NO del 05. El de 05 selecciona los jugadores a
    conservar con el índice `d_last` (17) mientras `pad_players` los seleccionó
    con `on_pitch` (14); cuando difieren, escribe predicciones en columnas de
    jugador equivocadas y el modelo aparece PEOR que B4 (ver RUNBOOK_animacion.md).
    El del 10 usa el índice correcto y, además, precalienta por periodo.
    """
    spec = importlib.util.spec_from_file_location(
        "anim10", ROOT / "scripts" / "10_animate_ghosting.py")
    anim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(anim)
    maxp = WindowConfig().max_players
    return lambda model, match, view, baseline, length: anim.predict_match(
        model, match, view, baseline, length, maxp)


def paired_role_samples(match, view, est_a, est_b, roles_by_period,
                        alive_only: bool):
    """
    Errores pareados de dos estimadores, ANOTADOS con el rol de cada jugador.

    A diferencia de metrics.paired_position_errors, aquí se conserva el índice
    de jugador para poder mapearlo a su rol estimado en el periodo del frame.
    Incluye porteros: el rol GK es justamente uno de los que la figura muestra.

    Returns
    -------
    dict con err_a, err_b, gap, frame, role  (todos alineados índice a índice).
    """
    scorable = (
        match.on_pitch
        & ~view.visible
        & np.isfinite(view.gap_s)
        & np.isfinite(est_a[..., 0])
        & np.isfinite(est_b[..., 0])
        & np.isfinite(match.positions[..., 0])
    )
    if alive_only:
        scorable &= match.ball_alive[:, None]

    fi, pi = np.where(scorable)
    if fi.size == 0:
        z = np.array([], dtype=np.float64)
        return {"err_a": z, "err_b": z, "gap": z,
                "frame": np.array([], np.int64), "role": np.array([], dtype=object)}

    truth = match.positions[fi, pi]
    role = np.empty(fi.size, dtype=object)
    periods = match.period[fi]
    for j in range(fi.size):
        role[j] = roles_by_period.get(int(periods[j]), {}).get(int(pi[j]), "?")

    return {
        "err_a": np.linalg.norm(est_a[fi, pi] - truth, axis=1).astype(np.float64),
        "err_b": np.linalg.norm(est_b[fi, pi] - truth, axis=1).astype(np.float64),
        "gap": view.gap_s[fi, pi].astype(np.float64),
        "frame": fi.astype(np.int64),
        "role": role,
    }


def bin_mask(gap, bin_label):
    """Máscara del bin de oclusión pedido; 'global' = todo."""
    if bin_label == "global":
        return np.ones(gap.shape, dtype=bool)
    lo, hi = dict(zip(BIN_LABELS,
                      zip(BIN_EDGES[:-1], BIN_EDGES[1:])))[bin_label]
    return (gap > lo) & (gap <= hi) if lo > 0 else (gap >= lo) & (gap <= hi)


def main():
    global model_max_players
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", default="J03WR9")
    ap.add_argument("--checkpoint", default=None,
                    help="por defecto, reports/cv/fold_<match>.pt")
    ap.add_argument("--bin", default="global",
                    choices=["global", "<=2s", "2-9.6s", ">9.6s"],
                    help="régimen de oclusión a desglosar por rol")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--min-n", type=int, default=25,
                    help="roles con menos muestras ocultas se marcan como ruidosos")
    ap.add_argument("--include-dead-ball", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ck_path = Path(args.checkpoint) if args.checkpoint else \
        ROOT / "reports" / "cv" / f"fold_{args.match}.pt"
    if not ck_path.exists():
        cands = sorted((ROOT / "reports" / "cv").glob("fold_*.pt"))
        if not cands:
            print(f"No hay checkpoint en {ck_path}")
            return 1
        ck_path = cands[0]
        print(f"[aviso] usando {ck_path.name}")

    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = ResidualImputer(dim=a["dim"], n_blocks=a["blocks"],
                            causal=not a["bidirectional"])
    model.load_state_dict(ck["state_dict"])
    model.to(torch.device(args.device)).eval()

    hits = list(PROC.glob(f"*{args.match}*.npz"))
    if not hits:
        print(f"No encontrado: {args.match}")
        return 1
    m = Match.load(hits[0])
    if m.fps > a["fps"]:
        m = m.resample(a["fps"])
    if a["minutes"]:
        m = m.head_minutes(a["minutes"])

    view = simulate(m, ViewportConfig(width_m=a["width"]))
    b4 = run_ladder(m, view, "B4", LadderConfig())
    predict_match = load_predict_match()
    pred = predict_match(model, m, view, b4, a["window"])

    roles_by_period = infer_all_roles(m, view)
    alive = not args.include_dead_ball
    samp = paired_role_samples(m, view, b4, pred, roles_by_period, alive)

    if samp["err_a"].size == 0:
        print("No hay muestras puntuables.")
        return 1

    sel_bin = bin_mask(samp["gap"], args.bin)

    # Agregado por rol
    rows = []
    for role in ROLE_ORDER:
        rmask = (samp["role"] == role) & sel_bin
        n = int(rmask.sum())
        if n == 0:
            continue
        sub = {
            "err_a": samp["err_a"][rmask],
            "err_b": samp["err_b"][rmask],
            "gap": samp["gap"][rmask],
            "frame": samp["frame"][rmask],
        }
        med_b4 = float(np.median(sub["err_a"]))
        med_md = float(np.median(sub["err_b"]))
        ci = paired_block_bootstrap_ci(sub, m.fps, n_boot=args.boot)
        d, lo, hi = ci.get("global", (med_b4 - med_md, np.nan, np.nan))
        rows.append({
            "role": role, "label": ROLE_LABEL[role], "n": n,
            "median_b4": med_b4, "median_model": med_md,
            "delta_m": d, "ci_lo": lo, "ci_hi": hi,
            "credible": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
            "noisy": n < args.min_n,
        })

    if not rows:
        print(f"No hay muestras en el bin {args.bin}.")
        return 1

    # Consola
    print(f"\n=== {m.match_id} · error por rol estimado · bin {args.bin} ===")
    hdr = f"{'rol':<18}{'n':>7}{'B4':>8}{'modelo':>9}{'delta':>9}{'IC 95% pareado':>22}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        ci = (f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"
              if np.isfinite(r["ci_lo"]) else "  (n insuf.)")
        star = " *" if r["credible"] else ("  ~" if r["noisy"] else "")
        print(f"{r['label']:<18}{r['n']:>7,}{r['median_b4']:>8.2f}"
              f"{r['median_model']:>9.2f}{r['delta_m']:>+9.2f}{ci:>22}{star}")
    print("\n  * = IC pareado excluye 0 (mejora creíble)   ~ = pocas muestras")

    # Figura
    FIG.mkdir(parents=True, exist_ok=True)
    labels = [r["label"] for r in rows]
    y = np.arange(len(rows))
    h = 0.38
    med_b4 = [r["median_b4"] for r in rows]
    med_md = [r["median_model"] for r in rows]
    role_cols = [ROLE_COLORS[r["role"]] for r in rows]

    fig, ax = plt.subplots(figsize=(11.5, 0.86 * len(rows) + 2.4))

    ax.barh(y + h / 2, med_b4, height=h, color=C_B4, edgecolor="white",
            label="B4 (heurística, sin entrenamiento)", zorder=3)
    ax.barh(y - h / 2, med_md, height=h, color=role_cols, edgecolor="white",
            label="modelo aprendido", zorder=3)

    xmax = max(max(med_b4), max(med_md))
    for r, yy in zip(rows, y):
        # etiqueta de mejora al final de la fila
        if np.isfinite(r["ci_lo"]):
            mark = "✓" if r["credible"] else ""
            txt = f"−{r['delta_m']:.1f} m{('  ' + mark) if mark else ''}" \
                if r["delta_m"] >= 0 else f"+{-r['delta_m']:.1f} m"
        else:
            txt = "n insuf."
        ax.text(xmax * 1.02, yy, txt, va="center", ha="left", fontsize=9.5,
                color="#222", weight="bold" if r["credible"] else "normal")
        # n de muestras, tenue, bajo la etiqueta de rol
        note = f"n={r['n']:,}" + ("  · pocas muestras" if r["noisy"] else "")
        ax.text(-xmax * 0.015, yy, note, va="center", ha="right",
                fontsize=8, color="#888")

    ax.set_yticks(y, labels, fontsize=11)
    ax.invert_yaxis()                       # portero arriba, delantero abajo
    ax.set_xlabel("error mediano de imputación (m)", fontsize=11)
    ax.set_xlim(0, xmax * 1.16)
    bin_txt = "todas las oclusiones" if args.bin == "global" \
        else f"oclusión {args.bin}"
    ax.set_title(
        f"Error por rol estimado (inferencia geométrica)\n"
        f"{m.match_id} · {bin_txt} · B4 vs. modelo aprendido · comparación pareada",
        fontsize=13, weight="bold", pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.01, 0.01,
             "Roles inferidos por geometría sobre frames visibles; no provienen "
             "de metadatos de formación. La barra corta es mejor.",
             fontsize=8, color="#999", ha="left")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    binslug = args.bin.replace(">", "gt").replace("<=", "le").replace(".", "")
    out = FIG / f"role_error_{m.match_id}_{binslug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n-> {out}")

    # CSV
    TAB.mkdir(parents=True, exist_ok=True)
    csv_out = TAB / f"role_error_{m.match_id}_{binslug}.csv"
    with open(csv_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"-> {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
