"""
Figuras del proyecto.

La figura clave es `plot_frame`: un panel con la cancha, la banda del viewport,
los jugadores visibles, los ocultos reales (verdad de terreno) y los fantasmas
imputados unidos por una línea al jugador real. Esa línea ES el error, y se
entiende sin saber matemáticas. Es la figura que va en la presentación.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin pantalla: funciona por SSH y en CI
import matplotlib.pyplot as plt
import numpy as np

from ..io.schema import Match
from ..camera.viewport import ViewportResult

# Paleta
C_PITCH = "#1b6b3a"
C_LINE = "#ffffff"
C_HOME = "#2b7bba"
C_AWAY = "#d1495b"
C_GHOST = "#f4a261"
C_BALL = "#ffffff"
C_VIEW = "#ffd166"


def draw_pitch(ax, pitch=(105.0, 68.0)) -> None:
    """Dibuja las líneas de una cancha reglamentaria."""
    L, A = pitch
    ax.set_facecolor(C_PITCH)
    kw = dict(color=C_LINE, lw=1.4, zorder=1)

    ax.plot([0, 0, L, L, 0], [0, A, A, 0, 0], **kw)
    ax.plot([L / 2, L / 2], [0, A], **kw)
    ax.add_patch(plt.Circle((L / 2, A / 2), 9.15, fill=False, **kw))
    ax.plot(L / 2, A / 2, "o", color=C_LINE, ms=3, zorder=1)

    for x0, sgn in ((0, 1), (L, -1)):
        # área grande (16.5 m) y área chica (5.5 m)
        for depth, half in ((16.5, 20.16), (5.5, 9.16)):
            ax.plot(
                [x0, x0 + sgn * depth, x0 + sgn * depth, x0],
                [A / 2 - half, A / 2 - half, A / 2 + half, A / 2 + half],
                **kw,
            )
        ax.plot(x0 + sgn * 11, A / 2, "o", color=C_LINE, ms=3, zorder=1)
        # portería
        ax.plot([x0, x0], [A / 2 - 3.66, A / 2 + 3.66], color=C_LINE, lw=3.5, zorder=2)

    ax.set_xlim(-4, L + 4)
    ax.set_ylim(-4, A + 4)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_frame(
    match: Match,
    view: ViewportResult,
    estimate: np.ndarray | None,
    t: int,
    title: str | None = None,
    out: str | Path | None = None,
):
    """
    Un frame: qué ve la cámara, qué hay realmente, y dónde pone el modelo a los ocultos.

    Elementos
    ---------
    - Banda amarilla  : región visible del viewport.
    - Círculos llenos : jugadores visibles.
    - Círculos huecos : jugadores ocultos, posición REAL (verdad de terreno).
    - Rombos naranjas : fantasmas imputados.
    - Línea punteada  : error de imputación (une fantasma con posición real).
    """
    L, A = match.pitch
    fig, ax = plt.subplots(figsize=(12, 7.6))
    draw_pitch(ax, match.pitch)

    c, w = float(view.center[t]), float(view.width[t])
    ax.axvspan(c - w / 2, c + w / 2, color=C_VIEW, alpha=0.16, zorder=0)
    for xx in (c - w / 2, c + w / 2):
        ax.axvline(xx, color=C_VIEW, lw=1.6, ls="--", alpha=0.85, zorder=1)

    pos, vis, on = match.positions[t], view.visible[t], match.on_pitch[t]

    for team, color in ((0, C_HOME), (1, C_AWAY)):
        sel = (match.team_idx == team) & on
        v = sel & vis
        h = sel & ~vis
        ax.scatter(pos[v, 0], pos[v, 1], s=170, c=color,
                   edgecolors="white", linewidths=1.6, zorder=5)
        ax.scatter(pos[h, 0], pos[h, 1], s=170, facecolors="none",
                   edgecolors=color, linewidths=2.2, alpha=0.75, zorder=4)

    if estimate is not None:
        hidden = on & ~vis & np.isfinite(estimate[t, :, 0])
        for k in np.where(hidden)[0]:
            gx, gy = estimate[t, k]
            rx, ry = pos[k]
            if np.isfinite(rx):
                ax.plot([gx, rx], [gy, ry], color=C_GHOST, lw=1.3,
                        ls=":", alpha=0.9, zorder=3)
            ax.scatter(gx, gy, s=130, marker="D", c=C_GHOST,
                       edgecolors="white", linewidths=1.2, zorder=6)

    if np.isfinite(match.ball[t]).all():
        ax.scatter(*match.ball[t], s=90, c=C_BALL, marker="o",
                   edgecolors="black", linewidths=1.4, zorder=8)

    n_vis, n_hid = int(vis.sum()), int((on & ~vis).sum())
    ttl = title or (
        f"t = {t / match.fps:6.1f} s   |   W = {w:.0f} m   |   "
        f"{n_vis} visibles, {n_hid} ocultos"
    )
    ax.set_title(ttl, fontsize=13, pad=12)

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=11, mfc=C_HOME, mec="w", label="visible"),
        plt.Line2D([], [], marker="o", ls="", ms=11, mfc="none", mec=C_HOME, label="oculto (real)"),
        plt.Line2D([], [], marker="D", ls="", ms=10, mfc=C_GHOST, mec="w", label="fantasma imputado"),
        plt.Line2D([], [], ls=":", color=C_GHOST, label="error"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=4, frameon=False, fontsize=10)

    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return None
    return fig


def plot_occlusion_histogram(
    match: Match, view: ViewportResult, out: str | Path | None = None
):
    """
    Histograma de duraciones de oclusión.

    Es la primera figura del proyecto y ya cuenta una historia sola: en un
    partido real, una fracción enorme de las oclusiones supera los 9.6 s, que
    es donde el estado del arte deja de estar definido.
    """
    from ..camera.viewport import LONG_OCCLUSION_S

    gap = view.gap_s[match.on_pitch & ~view.visible & match.ball_alive[:, None]]
    gap = gap[np.isfinite(gap) & (gap > 0)]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))

    a1.hist(np.clip(gap, 0, 60), bins=60, color="#2a9d8f", edgecolor="white", lw=0.4)
    a1.axvline(2.0, color="#264653", ls="--", lw=1.4)
    a1.axvline(LONG_OCCLUSION_S, color="#e76f51", ls="--", lw=1.8)
    a1.text(LONG_OCCLUSION_S + 0.7, a1.get_ylim()[1] * 0.9, "9.6 s",
            color="#e76f51", fontsize=10, weight="bold")
    a1.set_xlabel("duración de la oclusión (s)")
    a1.set_ylabel("nº de observaciones ocultas")
    a1.set_title("Distribución de gaps de oclusión")

    shares = [
        float((gap <= 2).mean()),
        float(((gap > 2) & (gap <= LONG_OCCLUSION_S)).mean()),
        float((gap > LONG_OCCLUSION_S).mean()),
    ]
    bars = a2.bar(["≤2 s", "2–9.6 s", ">9.6 s"], shares,
                  color=["#2a9d8f", "#e9c46a", "#e76f51"], edgecolor="white")
    for b, s in zip(bars, shares):
        a2.text(b.get_x() + b.get_width() / 2, s + 0.012, f"{s:.0%}",
                ha="center", fontsize=12, weight="bold")
    a2.set_ylim(0, max(shares) * 1.25)
    a2.set_ylabel("fracción de muestras ocultas")
    a2.set_title("Peso de cada régimen de oclusión")
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)
        a2.spines[s].set_visible(False)

    fig.suptitle(
        f"{match.match_id} — viewport W = {view.config.width_m:.0f} m",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return None
    return fig


def plot_ladder(rows: list[dict], out: str | Path | None = None):
    """Comparativa de la escalera: mediana global y por régimen de oclusión."""
    from ..metrics.position import BIN_LABELS

    rows = [r for r in rows if r.get("n", 0) > 0]
    methods = [r["method"] for r in rows]
    x = np.arange(len(methods))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.8))

    med = [r["median_all"] for r in rows]
    lo = [r["median_all"] - r.get("median_ci_lo", np.nan) for r in rows]
    hi = [r.get("median_ci_hi", np.nan) - r["median_all"] for r in rows]
    colors = ["#e76f51" if m == "B4" else "#457b9d" for m in methods]
    a1.bar(x, med, yerr=[lo, hi], color=colors, capsize=4, edgecolor="white")
    a1.set_xticks(x, methods)
    a1.set_ylabel("error mediano (m)")
    a1.set_title("Error global  ·  IC 95% por block bootstrap")

    for label, col in zip(BIN_LABELS, ["#2a9d8f", "#e9c46a", "#e76f51"]):
        a2.plot(x, [r.get(f"median_{label}", np.nan) for r in rows],
                "o-", color=col, label=label, lw=2, ms=7)
    a2.set_xticks(x, methods)
    a2.set_ylabel("error mediano (m)")
    a2.set_title("Error por régimen de oclusión")
    a2.legend(title="gap", frameon=False)

    for a in (a1, a2):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return None
    return fig
