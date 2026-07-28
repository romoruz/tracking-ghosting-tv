#!/usr/bin/env python3
"""
La figura de tres paneles: control real / control ciego / control con fantasmas.

Es la única salida de todo el proyecto que se entiende sin saber matemáticas, y
por eso es la que va a la presentación. Cuenta la historia entera en una
imagen:

  1. VERDAD      — control de cancha con los 22 jugadores. Lo que de verdad pasó.
  2. SOLO VISIBLE — control usando únicamente a los que la cámara muestra. Es lo
                    que produce hoy cualquier pipeline de tracking por TV sin
                    capa de imputación, y asigna al equipo equivocado zonas
                    enteras donde había un defensor fuera de cuadro.
  3. CON FANTASMAS — control usando visibles + las posiciones que estima el
                    modelo. Debe parecerse al panel 1.

Bajo cada panel va el error respecto a la verdad, en puntos porcentuales, y el
share de control por equipo. El número que importa es cuánto se reduce el error
del panel 2 al 3.

Uso:
    python scripts/09_figure_panels.py --match J03WR9 --checkpoint reports/cv/fold_J03WR9.pt
    python scripts/09_figure_panels.py --match metrica_1 --n-frames 6
"""
import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ghosting.io import Match  # noqa: E402
from ghosting.camera import simulate, ViewportConfig  # noqa: E402
from ghosting.baselines import run_ladder, LadderConfig  # noqa: E402
from ghosting.models import ResidualImputer, WindowConfig  # noqa: E402
from ghosting.metrics.pitch_control import (  # noqa: E402
    make_grid, pitch_control, control_share, control_mae,
)
from ghosting.viz.pitch import draw_pitch, C_HOME, C_AWAY, C_GHOST, C_VIEW  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, FIG = ROOT / "data" / "processed", ROOT / "reports" / "figures"


def velocidades(pos_t, pos_prev, fps):
    v = (pos_t - pos_prev) * fps
    sp = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(sp > 11.0, v * (11.0 / np.maximum(sp, 1e-6)), v)


def panel(ax, control, xs, ys, pos, team, vis, on, pitch, ghosts=None,
          titulo="", sub="", center=None, width=None):
    L, A = pitch
    ax.imshow(control.reshape(len(ys), len(xs)), extent=(0, L, 0, A),
              origin="lower", cmap="RdBu_r", vmin=0, vmax=1, alpha=0.72,
              interpolation="bilinear", zorder=0)
    draw_pitch(ax, pitch)
    ax.set_facecolor("none")

    if center is not None and width is not None:
        for xx in (center - width / 2, center + width / 2):
            ax.axvline(xx, color=C_VIEW, lw=2.2, ls="--", alpha=0.95, zorder=6)

    for t, c in ((0, C_HOME), (1, C_AWAY)):
        sel = (team == t) & on
        v, h = sel & vis, sel & ~vis
        ax.scatter(pos[v, 0], pos[v, 1], s=115, c=c, edgecolors="white",
                   linewidths=1.5, zorder=5)
        ax.scatter(pos[h, 0], pos[h, 1], s=115, facecolors="none", edgecolors=c,
                   linewidths=2.0, alpha=0.85, zorder=4)

    if ghosts is not None:
        ax.scatter(ghosts[:, 0], ghosts[:, 1], s=110, marker="D", c=C_GHOST,
                   edgecolors="white", linewidths=1.2, zorder=7)

    ax.set_title(titulo, fontsize=13, weight="bold", pad=8)
    ax.text(0.5, -0.06, sub, transform=ax.transAxes, ha="center",
            fontsize=11, color="#333")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", default="J03WR9")
    ap.add_argument("--checkpoint", default=None,
                    help="por defecto, reports/cv/fold_<match>.pt si existe")
    ap.add_argument("--n-frames", type=int, default=4,
                    help="cuántos frames candidatos dibujar; se eligen los de "
                         "mayor diferencia entre el mapa ciego y el real")
    ap.add_argument("--step", type=float, default=2.0)
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

    spec = importlib.util.spec_from_file_location(
        "ev", ROOT / "scripts" / "05_evaluate_model.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    ev.model_max_players = WindowConfig().max_players
    pred = ev.predict_match(model, m, view, b4, a["window"])

    xs, ys, pts = make_grid(m.pitch, args.step)
    FIG.mkdir(parents=True, exist_ok=True)

    # Elegir frames: los que más se distorsionan al ignorar a los ocultos.
    # No se eligen al azar: la figura debe mostrar el problema, y en un frame
    # donde casi todos son visibles no hay nada que enseñar.
    cand = np.where(m.ball_alive & (view.visible.sum(axis=1) < 16)
                    & (np.arange(m.n_frames) > 5))[0]
    rng = np.random.default_rng(0)
    cand = rng.choice(cand, size=min(120, cand.size), replace=False)

    puntuados = []
    for t in cand:
        on, vis = m.on_pitch[t], view.visible[t]
        h, aw = m.team_idx == 0, m.team_idx == 1
        c_true = pitch_control(pts, m.positions[t, on & h], m.positions[t, on & aw])
        c_blind = pitch_control(pts, m.positions[t, vis & h], m.positions[t, vis & aw])
        puntuados.append((control_mae(c_true, c_blind), t))
    puntuados.sort(reverse=True)

    for rank, (_, t) in enumerate(puntuados[: args.n_frames]):
        on, vis = m.on_pitch[t], view.visible[t]
        h, aw = m.team_idx == 0, m.team_idx == 1
        hid = on & ~vis

        vh = velocidades(m.positions[t, on & h], m.positions[t - 1, on & h], m.fps)
        va = velocidades(m.positions[t, on & aw], m.positions[t - 1, on & aw], m.fps)

        c_true = pitch_control(pts, m.positions[t, on & h], m.positions[t, on & aw],
                               vh, va)
        c_blind = pitch_control(pts, m.positions[t, vis & h], m.positions[t, vis & aw])

        pos_g = m.positions[t].copy()
        pos_g[hid] = pred[t, hid]
        c_ghost = pitch_control(pts, pos_g[on & h], pos_g[on & aw])

        # Zona oculta: fuera del viewport, que es donde se concentra el daño.
        c0, w = float(view.center[t]), float(view.width[t])
        fuera = np.abs(pts[:, 0] - c0) > w / 2

        fig, axes = plt.subplots(1, 3, figsize=(21, 5.4))
        for ax, ctrl, ttl, sub, gh in [
            (axes[0], c_true, "1 · La verdad",
             f"22 jugadores  ·  control local {control_share(c_true):.0%}", None),
            (axes[1], c_blind, "2 · Lo que ve la cámara",
             f"solo {int(vis.sum())} visibles  ·  error {control_mae(c_true, c_blind):.1f} pp"
             f"  (zona oculta {control_mae(c_true, c_blind, fuera):.1f} pp)", None),
            (axes[2], c_ghost, "3 · Con nuestros fantasmas",
             f"visibles + {int(hid.sum())} imputados  ·  error "
             f"{control_mae(c_true, c_ghost):.1f} pp"
             f"  (zona oculta {control_mae(c_true, c_ghost, fuera):.1f} pp)",
             pred[t, hid]),
        ]:
            panel(ax, ctrl, xs, ys, m.positions[t], m.team_idx, vis, on, m.pitch,
                  ghosts=gh, titulo=ttl, sub=sub, center=c0, width=w)

        red = 1 - control_mae(c_true, c_ghost, fuera) / max(
            control_mae(c_true, c_blind, fuera), 1e-9)
        fig.suptitle(
            f"{m.match_id} · min {t / m.fps / 60:.1f} — "
            f"la imputación recorta el error de control en la zona oculta un {red:.0%}",
            fontsize=15, weight="bold", y=1.03)
        fig.tight_layout()
        out = FIG / f"paneles_{m.match_id}_{rank + 1}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"-> {out}   (ciego {control_mae(c_true, c_blind, fuera):.1f} pp "
              f"-> fantasmas {control_mae(c_true, c_ghost, fuera):.1f} pp)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
