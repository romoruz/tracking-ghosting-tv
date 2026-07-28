#!/usr/bin/env python3
"""
Animación "ghosting en accion", version v3.

Genera clips de un solo panel que cuentan la historia sin matematicas: la camara
panea siguiendo el balon, unos jugadores salen del cuadro y se vuelven FANTASMAS,
el modelo los reconstruye, y cuando reaparecen se ve cuanto acerto.

Novedades v3
- ESTELA DIRECCIONAL (~1 s): cada fantasma arrastra una cola que se desvanece,
  convencion de broadcast profesional. Se dibuja con los puntos que el modelo
  predijo REALMENTE a 5 fps, unidos, sin interpolar (--trail-seconds, 0 la apaga).
- PRE-CALENTAMIENTO CAUSAL POR PERIODO: la reconstruccion por ventanas arranca en
  el primer frame de cada periodo y no cruza el descanso, de modo que el contexto
  causal de cada frame mostrado es identico al de la evaluacion del benchmark.
- UI SOBRIA: la franja inferior es una curva de la mediana del error a lo largo
  del clip (B4 vs modelo), no un widget de barras que oscile a 5 Hz. Nada compite
  con las trayectorias en la cancha. Los roles inferidos por geometria NO se usan
  aqui a proposito: en el video el color codifica EQUIPO (azul/rojo); el desglose
  por rol vive en su figura estatica dedicada (scripts/11_role_error_figure.py).

Modos (--mode)
  attack   : dibuja fantasmas SOLO del equipo en posesion (los que se estiran).
  defense  : dibuja fantasmas SOLO del equipo sin balon (la linea que se repliega).
  full     : ambos equipos, en camara lenta (dilatacion temporal, no interpolacion).

Salida
  reports/video/ataque/ , reports/video/defensa/ , reports/video/completo/

Todo es CAUSAL (solo pasado). Nada bidireccional entra aqui.

Limpieza visual (por que la version anterior saturaba)
- Color por equipo en TODO: rombos, lineas de error y texto heredan el color del
  equipo (azul local / rojo visitante). Se acabo el naranja monocromo.
- B4 (baseline) se dibuja muy tenue y en gris, para que el ojo vaya al modelo.
  En modos attack/defense, B4 se dibuja SOLO para el peor jugador (mayor error),
  como referencia, no como ruido.
- Transparencia de la linea de error proporcional al error (error-based fading):
  prediccion casi perfecta -> linea casi invisible; error grande -> opaca. La
  pantalla se limpia sola y la atencion va a donde el modelo sufre.
- Rombos con alpha para no tapar a los jugadores reales.
- El portero se ATENUA (mas transparente, sin etiqueta) pero NO se falsea: sigue
  siendo un fantasma real. Forzarlo a visible rompe la fisica del broadcast y la
  comparabilidad con Choi; ver viewport.py y 03_modelo_de_camara.md.

Honestidad de la camara lenta
El modelo predice a 5 fps. Para "lento y fluido" NO se interpolan posiciones
(seria inventar coordenadas que la red nunca infirio). Se dilata el tiempo:
cada prediccion se sostiene mas frames de video. Los saltos discretos siguen
evidenciando la tasa real de 5 Hz.

Uso:
    python scripts/10_animate_ghosting.py --match J03WR9 --mode full --preview
    python scripts/10_animate_ghosting.py --match J03WR9 --mode attack
    python scripts/10_animate_ghosting.py --match J03WR9 --mode defense
    python scripts/10_animate_ghosting.py --match J03WR9 --mode full --slow 2.0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.animation as manim     # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
import numpy as np                       # noqa: E402
import torch                             # noqa: E402

from ghosting.io import Match                                    # noqa: E402
from ghosting.camera import simulate, ViewportConfig, LONG_OCCLUSION_S  # noqa: E402
from ghosting.baselines import run_ladder, LadderConfig         # noqa: E402
from ghosting.models import ResidualImputer, WindowConfig, build_features  # noqa: E402
from ghosting.models.dataset import pad_players, FEATURES       # noqa: E402
from ghosting.metrics import position_errors, stratified_median  # noqa: E402
from ghosting.viz.pitch import draw_pitch, C_HOME, C_AWAY, C_VIEW  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
VID = ROOT / "reports" / "video"

# Temas
# light  : fondo claro, lineas oscuras. Estandar de figura academica/paper
#          (arXiv, PDFs). Es el que reproduce el estilo de los notebooks de
#          soccer analytics (Friends of Tracking, StatsBomb).
# dark   : fondo carbon, colores vibrantes. Estandar de dashboard/broadcast
#          para demo comercial (Second Spectrum, Stats Perform). Mas "premium".
# El mismo script sirve para ambos publicos: elige con --theme.
THEMES = {
    "light": dict(fig="#FFFFFF", ax="#F8F9FA", line="#343A40", text="#111111",
                  muted="#6c757d", home="#2b7bba", away="#d1495b",
                  b4_home="#e07b1a", b4_away="#8e44ad", grid="#000000",
                  band="#ffd166", band_a=0.10),
    "dark":  dict(fig="#1A1A1A", ax="#1E1E1E", line="#404040", text="#E0E0E0",
                  muted="#9aa3ad", home="#4DA3FF", away="#FF5C6C",
                  b4_home="#FFB454", b4_away="#C08CF0", grid="#FFFFFF",
                  band="#ffd166", band_a=0.07),
}

# Estas globales las fija apply_theme() al arrancar; los defaults son 'light'.
_TH = THEMES["light"]
C_HOME = _TH["home"]
C_AWAY = _TH["away"]
C_VIEW = _TH["band"]
TEAM_COL = {0: _TH["home"], 1: _TH["away"]}
C_B4 = _TH["muted"]                          # baseline gris (solo estela GK)
C_B4_TEAM = {0: _TH["b4_home"], 1: _TH["b4_away"]}
C_PITCH_LINE = _TH["line"]
C_FIG, C_AX, C_TEXT, C_GRID = _TH["fig"], _TH["ax"], _TH["text"], _TH["grid"]
C_MUTED, C_BAND, C_BAND_A = _TH["muted"], _TH["band"], _TH["band_a"]


def apply_theme(name: str) -> dict:
    """Fija las globales de color segun el tema. Se llama una vez en main()."""
    global _TH, C_HOME, C_AWAY, C_VIEW, TEAM_COL, C_B4, C_B4_TEAM, C_PITCH_LINE
    global C_FIG, C_AX, C_TEXT, C_GRID, C_MUTED, C_BAND, C_BAND_A
    _TH = THEMES[name]
    C_HOME, C_AWAY, C_VIEW = _TH["home"], _TH["away"], _TH["band"]
    TEAM_COL = {0: _TH["home"], 1: _TH["away"]}
    C_B4 = _TH["muted"]
    C_B4_TEAM = {0: _TH["b4_home"], 1: _TH["b4_away"]}
    C_PITCH_LINE = _TH["line"]
    C_FIG, C_AX, C_TEXT, C_GRID = _TH["fig"], _TH["ax"], _TH["text"], _TH["grid"]
    C_MUTED, C_BAND, C_BAND_A = _TH["muted"], _TH["band"], _TH["band_a"]
    return _TH

# Umbrales de la tabla de clasificacion (norma euclidiana del error, metros).
# OJO: son umbrales de MAGNITUD de error, distintos de los bins de GAP de
# oclusion del benchmark (<=2s / 2-9.6s / >9.6s). Aqui clasifican calidad de
# reconstruccion instantanea, no dificultad. Configurables por CLI.
THR_OPT_DEFAULT = 2.0                        # <= optimo (radio de control/zancada)
THR_CRIT_DEFAULT = 5.0                       # > critico (fantasma fuera de zona)
ON = FEATURES.index("on_pitch")
MODE_DIR = {"attack": "ataque", "defense": "defensa", "full": "completo"}


# Inferencia (predict_match corregido: usa on_pitch, indice 14, no d_last)
def _period_bounds(period, T):
    """Lista de (inicio, fin) de cada periodo, medio abierto [inicio, fin)."""
    bounds, p0 = [], 0
    for t in range(1, T + 1):
        if t == T or period[t] != period[p0]:
            bounds.append((p0, t))
            p0 = t
    return bounds


@torch.no_grad()
def predict_match(model, match, view, baseline, length, max_players):
    """
    Reconstruye la prediccion del modelo por ventanas solapadas al 50%, tomando
    de cada ventana solo su segunda mitad (>= length/2 frames de contexto por
    detras). Los frames de arranque de cada periodo conservan B4.

    PRE-CALENTAMIENTO CAUSAL POR PERIODO
    ------------------------------------
    Las ventanas se generan DENTRO de cada periodo y arrancan en su primer frame,
    sin cruzar nunca el descanso. Dos razones, ambas de causalidad:
      - B4, el baseline sobre el que se predice el residuo, reinicia sus offsets
        recursivos en cada periodo (los equipos cambian de lado). Una ventana a
        caballo del descanso mezclaria estados de dos mitades con la cancha
        invertida.
      - La atencion temporal veria "futuro" de la otra mitad; en modo causal eso
        es incorrecto.
    Arrancar desde el inicio del periodo — y no desde 50 frames antes del
    segmento que se anima — es lo que garantiza que cada frame mostrado tenga
    EXACTAMENTE el mismo contexto causal que tuvo en la evaluacion del benchmark.
    """
    per = build_features(match, view, baseline)
    T, N = per["loss_mask"].shape
    out = baseline.copy()
    half = length // 2
    dev = next(model.parameters()).device

    for ps, pe in _period_bounds(match.period, T):
        if pe - ps < length:
            continue  # periodo mas corto que una ventana: se queda en B4
        for s in range(ps, pe - length + 1, half):
            e = s + length
            w = pad_players({
                "feats": per["feats"][s:e], "target": per["target"][s:e],
                "loss_mask": per["loss_mask"][s:e], "base": per["base"][s:e],
            }, max_players)
            on = per["feats"][s:e, :, ON].max(axis=0) > 0
            score = per["loss_mask"][s:e].sum(axis=0) + on * 1e6
            keep = np.sort(np.argsort(-score)[:max_players])
            b = {k: torch.from_numpy(v[None]).to(dev) for k, v in w.items()}
            pred = model.predict_positions(
                b["feats"], b["player_mask"], b["base"])[0].cpu().numpy()
            # Primera ventana del periodo: se conserva entera (arranque causal
            # desde el inicio del periodo). Las siguientes: solo su segunda mitad.
            lo = s if s == ps else s + half
            off = 0 if s == ps else half
            out[lo:e, keep[:min(len(keep), max_players)]] = pred[off:, :len(keep)]
    return out


# Posesion aproximada por frame: equipo cuyo jugador esta mas cerca del balon
def possession_team(match):
    T = match.n_frames
    poss = np.zeros(T, dtype=np.int8)
    for t in range(T):
        b = match.ball[t]
        if not np.isfinite(b).all():
            poss[t] = poss[t - 1] if t else 0
            continue
        best_d, best_team = np.inf, 0
        for team in (0, 1):
            sel = (match.team_idx == team) & match.on_pitch[t]
            p = match.positions[t, sel]
            p = p[np.isfinite(p[:, 0])]
            if p.size:
                d = np.min(np.hypot(p[:, 0] - b[0], p[:, 1] - b[1]))
                if d < best_d:
                    best_d, best_team = d, team
        poss[t] = best_team
    # suavizado de 1 s para que no parpadee
    k = int(round(match.fps))
    if k > 1:
        from numpy.lib.stride_tricks import sliding_window_view
        pad = np.pad(poss, (k // 2, k // 2), mode="edge")
        sm = np.round(sliding_window_view(pad, k).mean(axis=1)).astype(np.int8)
        poss = sm[:T]
    return poss


# Seleccion del segmento, sensible al modo
def pick_segment(match, view, clip_frames, mode, poss):
    gap = view.gap_s
    hidden = match.on_pitch & ~view.visible
    long_h = hidden & (gap > LONG_OCCLUSION_S)
    if mode in ("attack", "defense"):
        team_of_interest = None  # se decide por frame segun posesion
        # attack: fantasmas del equipo EN posesion ; defense: del equipo SIN balon
        want_attack = (mode == "attack")
        score_f = np.zeros(match.n_frames)
        for team in (0, 1):
            tmask = (match.team_idx == team)
            cnt = (long_h & tmask[None, :]).sum(1).astype(float)
            # frames donde ESTE equipo es el relevante
            rel = (poss == team) if want_attack else (poss != team)
            score_f += cnt * rel
    else:
        score_f = long_h.sum(1).astype(float)

    prevgap = np.zeros_like(gap); prevgap[1:] = gap[:-1]
    reap = (view.visible & (~np.isfinite(gap) | (gap == 0)) &
            (prevgap > LONG_OCCLUSION_S)).sum(1).astype(float)
    alive = match.ball_alive.astype(float)

    cS = np.concatenate([[0], np.cumsum(score_f)])
    cR = np.concatenate([[0], np.cumsum(reap)])
    cA = np.concatenate([[0], np.cumsum(alive)])
    best = None
    for s in range(0, match.n_frames - clip_frames):
        e = s + clip_frames
        if match.period[s] != match.period[e - 1]:
            continue
        if (cA[e] - cA[s]) / clip_frames < 0.85:
            continue
        sc = (cS[e] - cS[s]) + 25.0 * (cR[e] - cR[s])
        if best is None or sc > best[0]:
            best = (sc, s, e)
    return (best[1], best[2]) if best else (0, clip_frames)


# Error por frame (para la franja), por subconjunto de jugadores
def frame_errors(match, view, est, t, player_sel):
    # El portero se EXCLUYE de la metrica de error: su dinamica es un ancla
    # estacionaria a su porteria, no al centroide del equipo, y contamina la
    # mediana del bloque de campo (produce el pico cuando lo unico oculto es un
    # arquero). Misma politica que la tabla de umbrales y que el benchmark, que
    # reportan al portero por separado. Ver 09_decisiones_y_errores.md C.1.
    on, vis, gap = match.on_pitch[t], view.visible[t], view.gap_s[t]
    mask = (on & ~vis & player_sel & ~match.is_gk & np.isfinite(gap)
            & np.isfinite(est[t, :, 0]) & np.isfinite(match.positions[t, :, 0]))
    if not mask.any():
        return np.array([])
    return np.hypot(est[t, mask, 0] - match.positions[t, mask, 0],
                    est[t, mask, 1] - match.positions[t, mask, 1])


# Render de un frame
def err_alpha(d, dmax=25.0):
    """Error-based fading: poco error -> tenue; mucho -> opaco."""
    return float(np.clip(0.15 + 0.85 * (d / dmax), 0.15, 1.0))


def draw_trail(ax, ctx, t, k, color):
    """
    Estela direccional (~1 s) de la posicion imputada por el modelo.

    Convencion de broadcast (Second Spectrum, etc.): una cola que se desvanece
    hacia el pasado indica de donde viene el fantasma y hacia donde va. Aqui es
    ademas una afirmacion de honestidad: se dibujan los puntos que el modelo
    predijo REALMENTE a 5 fps, unidos, no una interpolacion suave. No se inventan
    coordenadas intermedias. La estela se reinicia en cada periodo.
    """
    m, pred = ctx["m"], ctx["pred"]
    tf = ctx["trail_frames"]
    if tf < 1:
        return
    t0 = max(int(ctx["period_start"][t]), t - tf)
    ts = np.arange(t0, t + 1)
    xy = pred[ts, k]
    good = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    ts, xy = ts[good], xy[good]
    if xy.shape[0] < 2:
        return
    n = xy.shape[0] - 1
    for j in range(n):
        frac = (j + 1) / n                    # 0 (viejo) -> 1 (reciente)
        ax.plot(xy[j:j + 2, 0], xy[j:j + 2, 1], color=color,
                lw=0.7 + 2.1 * frac, alpha=0.08 + 0.42 * frac,
                solid_capstyle="round", zorder=2)
    # puntos en cada muestra de 5 fps: hacen visible la cadencia real
    ax.scatter(xy[:-1, 0], xy[:-1, 1], s=9, color=color, alpha=0.30, zorder=2)


def threshold_counts(m, view, b4, pred, t, player_sel, thr_opt, thr_crit):
    """
    Clasifica a los fantasmas de campo ocluidos del frame t en tres umbrales de
    magnitud de error, para B4 y para el modelo por separado.

    Excluye SIEMPRE al portero: su dinamica es distinta y no entra en la
    contabilidad de ghosting (misma politica que el resto del pipeline).

    Devuelve dict con arrays [opt, medio, crit] para 'b4' y 'modelo', y el total.
    """
    on, vis, gap = m.on_pitch[t], view.visible[t], view.gap_s[t]
    sel = (on & ~vis & player_sel & ~m.is_gk
           & np.isfinite(gap)
           & np.isfinite(pred[t, :, 0]) & np.isfinite(b4[t, :, 0])
           & np.isfinite(m.positions[t, :, 0]))
    truth = m.positions[t, sel]
    e_b4 = np.hypot(b4[t, sel, 0] - truth[:, 0], b4[t, sel, 1] - truth[:, 1])
    e_md = np.hypot(pred[t, sel, 0] - truth[:, 0], pred[t, sel, 1] - truth[:, 1])

    def bucketize(e):
        opt = int((e <= thr_opt).sum())
        crit = int((e > thr_crit).sum())
        return [opt, e.size - opt - crit, crit]

    return {"b4": bucketize(e_b4), "modelo": bucketize(e_md), "n": int(e_b4.size)}


def draw_threshold_table(ax, ctx, t):
    """Tabla estatica-por-frame: cuantos fantasmas caen en cada umbral (B4 vs modelo)."""
    ax.clear()
    ax.axis("off")
    ax.set_facecolor(C_AX)
    thr_opt, thr_crit = ctx["thr_opt"], ctx["thr_crit"]
    c = threshold_counts(ctx["m"], ctx["view"], ctx["b4"], ctx["pred"], t,
                         ctx["player_sel_by_frame"][t], thr_opt, thr_crit)

    rows = [
        (f"Optimo  (\u2264 {thr_opt:.0f} m)", c["b4"][0], c["modelo"][0], "#2a9d8f"),
        (f"Medio  ({thr_opt:.0f}\u2013{thr_crit:.0f} m)", c["b4"][1], c["modelo"][1], "#e9a92a"),
        (f"Critico  (> {thr_crit:.0f} m)", c["b4"][2], c["modelo"][2], "#e76f51"),
    ]
    win = "#1f7a5a" if _TH is THEMES["dark"] else "#c9f0e4"
    cell_bg = C_AX if _TH is THEMES["dark"] else "#f1f3f5"
    hdr_bg = "#2b2b2b" if _TH is THEMES["dark"] else "#dee2e6"
    cell_text, cell_col = [], []
    for label, nb4, nmd, band in rows:
        cell_text.append([label, str(nb4), str(nmd)])
        # resalta la celda del modelo cuando gana en esa fila:
        # mas fantasmas en optimo, o menos en critico.
        better = (nmd > nb4) if band == "#2a9d8f" else \
                 (nmd < nb4) if band == "#e76f51" else (nmd >= nb4)
        cell_col.append([band, cell_bg, win if better else cell_bg])

    tbl = ax.table(cellText=cell_text,
                   colLabels=["reconstruccion", "B4", "modelo"],
                   cellColours=cell_col,
                   colColours=[hdr_bg, hdr_bg, hdr_bg],
                   cellLoc="center", colLoc="center",
                   colWidths=[0.56, 0.22, 0.22],
                   bbox=[0.0, 0.0, 1.0, 0.86])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    for (r, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(C_AX)
        cell.set_linewidth(1.5)
        txt = cell.get_text()
        if r == 0:                       # cabecera
            cell.set_text_props(weight="bold", color=C_TEXT)
        elif col == 0:                   # etiqueta de banda: texto blanco sobre color
            cell.set_text_props(weight="bold", color="white")
            txt.set_ha("center")
        else:                            # conteos
            cell.set_text_props(color=C_TEXT, fontsize=14, weight="bold")
    ax.set_title(f"Fantasmas de campo ocluidos ahora: {c['n']}",
                 fontsize=12, pad=4, color=C_TEXT, weight="bold")
    win_col = "#1f7a5a" if _TH is THEMES["dark"] else "#2a9d8f"
    ax.text(0.5, -0.06, "celda verde = el modelo gana esa fila",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, color=win_col, style="italic")



def build_legend(ax):
    """
    Dibuja la leyenda topologica UNA sola vez en su propio eje.

    Vive en un eje dedicado (no flotando sobre la cancha) por una razon de
    producto: en modo 'full' hay jugadores por todo el campo y una leyenda
    flotante taparia nodos de forma impredecible frame a frame — se ve amateur
    en una demo. Un panel fijo nunca encima datos y siempre es legible.
    Items EXPLICITOS por equipo, sin abreviar.
    """
    ax.clear()
    ax.axis("off")
    ax.set_facecolor(C_AX)
    handles = [
        Line2D([], [], marker="o", ls="", ms=13, mfc=C_MUTED, mec=C_TEXT,
               label="Observado  (la camara lo ve)"),
        Line2D([], [], marker="o", ls="", ms=13, mfc="none", mec=C_MUTED,
               label="Oculto  ·  posicion real (verdad terreno)"),
        Line2D([], [], marker="D", ls="", ms=12, mfc=C_HOME, mec=C_HOME,
               label="Modelo  ·  equipo local"),
        Line2D([], [], marker="D", ls="", ms=12, mfc=C_AWAY, mec=C_AWAY,
               label="Modelo  ·  equipo visitante"),
        Line2D([], [], marker="D", ls="", ms=12, mfc="none", mec=C_B4_TEAM[0],
               label="Baseline B4  ·  equipo local"),
        Line2D([], [], marker="D", ls="", ms=12, mfc="none", mec=C_B4_TEAM[1],
               label="Baseline B4  ·  equipo visitante"),
        Line2D([], [], ls="-", color=C_TEXT, lw=2.4, label="Error espacial  ·  modelo"),
        Line2D([], [], ls="--", color=C_B4_TEAM[0], lw=1.8, label="Error espacial  ·  B4"),
    ]
    leg = ax.legend(handles=handles, loc="center left", frameon=True,
                    framealpha=0.85, fontsize=12.5, borderaxespad=0.0,
                    handletextpad=0.8, labelspacing=0.95,
                    facecolor=C_AX, edgecolor=C_PITCH_LINE)
    for txt in leg.get_texts():
        txt.set_color(C_TEXT)
    ax.set_title("Leyenda", fontsize=12, color=C_MUTED, loc="left", pad=2)


def draw(ax_pitch, ax_strip, ax_table, ctx, t):
    m, view, b4, pred = ctx["m"], ctx["view"], ctx["b4"], ctx["pred"]
    seg, mode, poss = ctx["seg"], ctx["mode"], ctx["poss"]
    ghost_teams = ctx["ghost_teams_by_frame"][t]   # set de equipos a dibujar
    L, A = m.pitch
    ax_pitch.clear()
    draw_pitch(ax_pitch, m.pitch)
    # Contraste: la cancha por defecto trae lineas blancas (invisibles sobre
    # claro). Aplicamos el tema (fondo + lineas) solo en este lienzo, sin tocar
    # viz/pitch.py (que otras figuras usan en verde).
    ax_pitch.set_facecolor(C_AX)
    for ln in ax_pitch.lines:
        ln.set_color(C_PITCH_LINE)
    for pc in ax_pitch.patches:
        pc.set_edgecolor(C_PITCH_LINE)
    # margen interno minimo: que la cancha llene su celda (draw_pitch deja -4..L+4)
    ax_pitch.set_xlim(-1.5, L + 1.5)
    ax_pitch.set_ylim(-1.5, A + 1.5)

    c = float(view.center[t]); w = float(view.width[t])
    ax_pitch.axvspan(c - w / 2, c + w / 2, color=C_BAND, alpha=C_BAND_A, zorder=0)
    for xx in (c - w / 2, c + w / 2):
        ax_pitch.axvline(xx, color=C_BAND, lw=1.8, ls="--", alpha=0.75, zorder=1)

    pos, vis, on, gap = m.positions[t], view.visible[t], m.on_pitch[t], view.gap_s[t]

    # jugadores reales: visibles llenos, ocultos huecos, por equipo
    for team, color in TEAM_COL.items():
        sel = (m.team_idx == team) & on
        v = sel & vis
        ax_pitch.scatter(pos[v, 0], pos[v, 1], s=210, c=color,
                         edgecolors="white", linewidths=1.6, zorder=6)
        h = sel & ~vis
        ax_pitch.scatter(pos[h, 0], pos[h, 1], s=210, facecolors="none",
                         edgecolors=color, linewidths=2.0, alpha=0.45, zorder=4)

    # peor jugador (para B4 de referencia en modos de un equipo)
    worst_k = ctx["worst_by_frame"].get(t, -1)

    # fantasmas del/los equipo(s) elegido(s)
    for team in ghost_teams:
        color = TEAM_COL[team]
        tmask = (m.team_idx == team)
        hidden = on & ~vis & tmask & np.isfinite(pred[t, :, 0])
        for k in np.where(hidden)[0]:
            rx, ry = pos[k]
            gxm, gym = pred[t, k]
            is_gk = bool(m.is_gk[k])
            d = float(np.hypot(gxm - rx, gym - ry)) if np.isfinite(rx) else np.nan
            a_line = err_alpha(d) * (0.35 if is_gk else 1.0)
            # estela direccional del fantasma (~1 s), bajo todo lo demas
            draw_trail(ax_pitch, ctx, t, k, color if not is_gk else C_B4)
            # linea de error del modelo, color de equipo, alpha por error
            if np.isfinite(rx):
                ax_pitch.plot([gxm, rx], [gym, ry], color=color, lw=1.8,
                              alpha=a_line, zorder=3, solid_capstyle="round")
                if not is_gk and d >= 6.0:   # etiqueta solo si el error es apreciable
                    mx, my = (gxm + rx) / 2, (gym + ry) / 2
                    ax_pitch.text(mx, my + 0.6, f"{d:.0f}", color="white",
                                  fontsize=8, ha="center", va="bottom",
                                  weight="bold", zorder=9,
                                  bbox=dict(boxstyle="round,pad=0.12", fc=color,
                                            ec="none", alpha=0.85))
            # B4 de referencia: solo el peor jugador en modos de un equipo,
            # o todos en full. Rombo TRANSPARENTE en color de baseline por
            # equipo (naranja local / morado visitante), con su residual
            # trazado en linea discontinua hasta la posicion real.
            draw_b4 = (mode == "full") or (k == worst_k)
            if draw_b4 and np.isfinite(b4[t, k, 0]):
                cb4 = C_B4_TEAM[team]
                bx, by = b4[t, k]
                if np.isfinite(rx):
                    db4 = float(np.hypot(bx - rx, by - ry))
                    ax_pitch.plot([bx, rx], [by, ry], color=cb4, lw=1.4,
                                  ls="--", zorder=3, dash_capstyle="round",
                                  alpha=err_alpha(db4) * (0.35 if is_gk else 0.9))
                ax_pitch.scatter(bx, by, s=90, marker="D",
                                 facecolors="none", edgecolors=cb4,
                                 linewidths=1.6, zorder=5,
                                 alpha=0.30 if is_gk else 0.65)
            # rombo del modelo, semitransparente, borde de equipo
            ax_pitch.scatter(gxm, gym, s=170, marker="D", c=color,
                             alpha=0.30 if is_gk else 0.70,
                             edgecolors=color, linewidths=1.8, zorder=7)

    # balon
    if np.isfinite(m.ball[t]).all():
        ax_pitch.scatter(*m.ball[t], s=130, c="#ffffff", marker="o",
                         edgecolors=C_PITCH_LINE, linewidths=1.5, zorder=10)

    # (La leyenda vive en su propio eje, dibujada una sola vez por build_legend;
    #  asi la cancha ocupa todo su lienzo y la leyenda nunca encima jugadores.)

    # cabecera: estatica salvo el reloj. El conteo usa la MISMA mascara que la
    # tabla (solo jugadores de campo, sin porteros); los porteros ocluidos se
    # reportan aparte para que titulo y tabla nunca se contradigan.
    field_ghost = sum(int(((m.team_idx == te) & on & ~vis & ~m.is_gk).sum())
                      for te in ghost_teams)
    gk_ghost = int((on & ~vis & m.is_gk).sum())
    gk_txt = f"   ·   {gk_ghost} portero(s) oculto(s)" if gk_ghost else ""
    modo_txt = {"attack": "equipo en posesion", "defense": "equipo sin balon",
                "full": "ambos equipos"}[mode]
    ax_pitch.set_title(
        f"{ctx['headline']}   ·   {modo_txt}\n"
        f"t = {(t - seg[0]) / m.fps:4.1f} s   ·   {field_ghost} fantasmas de campo ocluidos{gk_txt}",
        fontsize=16, pad=14, weight="bold", color=C_TEXT)

    # tabla de clasificacion por umbral (B4 vs modelo), se actualiza cada frame
    draw_threshold_table(ax_table, ctx, t)

    # franja temporal por equipo (o total en full)
    ax_strip.clear()
    ax_strip.set_facecolor(C_AX)
    xs = np.arange(seg[0], t + 1)
    xt = (xs - seg[0]) / m.fps
    md_col = C_TEXT if mode == "full" else TEAM_COL[ctx["single_team"]]
    key_b4 = "b4_all" if mode == "full" else "b4_t"
    key_md = "md_all" if mode == "full" else "md_t"
    ax_strip.plot(xt, ctx["series"][key_b4][: len(xt)], color=C_MUTED, lw=2.0, label="B4")
    ax_strip.plot(xt, ctx["series"][key_md][: len(xt)], color=md_col, lw=2.6, label="modelo")
    ax_strip.axhline(LONG_OCCLUSION_S, color="#e76f51", lw=0.9, ls=":", alpha=0.5)
    ax_strip.set_xlim(0, (seg[1] - 1 - seg[0]) / m.fps)
    ymax = max(12, np.nanmax(ctx["series"]["b4_all"]) * 1.15)
    ax_strip.set_ylim(0, ymax)
    ax_strip.set_xlabel("segundos del clip", fontsize=12, color=C_TEXT)
    ax_strip.set_ylabel("mediana error (m)  ·  mov. 1 s", fontsize=11, color=C_TEXT)
    ax_strip.tick_params(labelsize=11, colors=C_TEXT)
    leg2 = ax_strip.legend(loc="upper right", fontsize=11, frameon=False, ncol=2)
    for txt in leg2.get_texts():
        txt.set_color(C_TEXT)
    for side, sp in ax_strip.spines.items():
        sp.set_visible(side in ("bottom", "left"))
        sp.set_color(C_PITCH_LINE)
    ax_strip.grid(axis="y", alpha=0.18, color=C_GRID)


def cumulative_median_fn(m, view, est, seg, player_sel_by_frame):
    """Devuelve funcion t -> mediana acumulada del error desde seg[0] hasta t."""
    cache = {}
    def fn(t):
        if t in cache:
            return cache[t]
        vals = []
        for tt in range(seg[0], t + 1):
            e = frame_errors(m, view, est, tt, player_sel_by_frame[tt])
            if e.size:
                vals.append(e)
        r = float(np.median(np.concatenate(vals))) if vals else np.nan
        cache[t] = r
        return r
    return fn


def build_context(m, view, b4, pred, seg, mode, poss, headline, trail_frames=5,
                  thr_opt=THR_OPT_DEFAULT, thr_crit=THR_CRIT_DEFAULT):
    T = m.n_frames
    all_players = np.ones(m.n_players, dtype=bool)

    # Inicio del periodo de cada frame: la estela se reinicia en el descanso.
    period_start = np.zeros(T, dtype=int)
    cur = 0
    for t in range(T):
        if t == 0 or m.period[t] != m.period[t - 1]:
            cur = t
        period_start[t] = cur

    # que equipos se dibujan como fantasmas en cada frame
    ghost_teams_by_frame = {}
    player_sel_by_frame = {}      # para el error de la franja/mediana
    single_team = None
    for t in range(seg[0], seg[1]):
        if mode == "full":
            teams = (0, 1)
            sel = all_players
        elif mode == "attack":
            teams = (int(poss[t]),)
            sel = (m.team_idx == poss[t])
        else:  # defense
            teams = (int(1 - poss[t]),)
            sel = (m.team_idx == (1 - poss[t]))
        ghost_teams_by_frame[t] = teams
        player_sel_by_frame[t] = sel

    # peor jugador por frame (para B4 de referencia en attack/defense)
    worst_by_frame = {}
    if mode in ("attack", "defense"):
        for t in range(seg[0], seg[1]):
            on, vis, gap = m.on_pitch[t], view.visible[t], view.gap_s[t]
            sel = player_sel_by_frame[t] & on & ~vis & np.isfinite(pred[t, :, 0]) & np.isfinite(m.positions[t, :, 0]) & ~m.is_gk
            if sel.any():
                d = np.hypot(pred[t, sel, 0] - m.positions[t, sel, 0],
                             pred[t, sel, 1] - m.positions[t, sel, 1])
                idxs = np.where(sel)[0]
                worst_by_frame[t] = int(idxs[np.argmax(d)])

    # series para la franja
    def series_over(est, sel_by_frame):
        out = []
        for t in range(seg[0], seg[1]):
            e = frame_errors(m, view, est, t, sel_by_frame[t])
            out.append(np.median(e) if e.size else np.nan)
        a = np.array(out)
        idx = np.arange(len(a)); good = np.isfinite(a)
        if good.any():
            a = np.interp(idx, idx[good], a[good])
        # Mediana movil de ~1 s: la mediana por-frame salta cuando un jugador
        # reaparece (su error cae a ~0 de golpe) o cuando queda 1 solo fantasma
        # dificil. Ese pico no dice nada del metodo; suavizarlo deja ver la
        # tendencia. Es una media movil de PRESENTACION, no cambia ninguna
        # metrica reportada.
        k = max(1, int(round(m.fps)))
        if k > 1 and a.size >= k:
            kernel = np.ones(k) / k
            a = np.convolve(a, kernel, mode="same")
        return a
    all_by_frame = {t: all_players for t in range(seg[0], seg[1])}
    series = {
        "b4_all": series_over(b4, all_by_frame),
        "md_all": series_over(pred, all_by_frame),
        "b4_t": series_over(b4, player_sel_by_frame),
        "md_t": series_over(pred, player_sel_by_frame),
    }
    if mode in ("attack", "defense"):
        single_team = int(poss[(seg[0] + seg[1]) // 2]) if mode == "attack" \
            else int(1 - poss[(seg[0] + seg[1]) // 2])

    run_b4 = cumulative_median_fn(m, view, b4, seg,
                                  all_by_frame if mode == "full" else player_sel_by_frame)
    run_md = cumulative_median_fn(m, view, pred, seg,
                                  all_by_frame if mode == "full" else player_sel_by_frame)

    return dict(m=m, view=view, b4=b4, pred=pred, seg=seg, mode=mode, poss=poss,
                headline=headline, ghost_teams_by_frame=ghost_teams_by_frame,
                worst_by_frame=worst_by_frame, series=series,
                player_sel_by_frame=player_sel_by_frame,
                single_team=single_team, run_b4=run_b4, run_md=run_md,
                period_start=period_start, trail_frames=int(trail_frames),
                thr_opt=float(thr_opt), thr_crit=float(thr_crit))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", default="J03WR9")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--mode", choices=["attack", "defense", "full"], default="full")
    ap.add_argument("--clip-seconds", type=float, default=12.0)
    ap.add_argument("--start-s", type=float, default=None)
    ap.add_argument("--fps-video", type=int, default=25)
    ap.add_argument("--slow", type=float, default=1.0,
                    help="factor de camara lenta por dilatacion temporal (2.0 = mitad de velocidad)")
    ap.add_argument("--trail-seconds", type=float, default=1.0,
                    help="duracion de la estela direccional del fantasma, en "
                         "segundos (0 la desactiva). Se dibuja con los puntos "
                         "reales del modelo a 5 fps, sin interpolar")
    ap.add_argument("--thr-opt", type=float, default=THR_OPT_DEFAULT,
                    help="umbral de error OPTIMO en metros (<= optimo)")
    ap.add_argument("--thr-crit", type=float, default=THR_CRIT_DEFAULT,
                    help="umbral de error CRITICO en metros (> critico)")
    ap.add_argument("--theme", choices=["light", "dark"], default="light",
                    help="light = figura de paper/arXiv (fondo claro). "
                         "dark = dashboard/demo comercial (fondo carbon).")
    ap.add_argument("--out", default=None)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    apply_theme(args.theme)

    ck_path = Path(args.checkpoint) if args.checkpoint else \
        ROOT / "reports" / "cv" / f"fold_{args.match}.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = ResidualImputer(dim=a["dim"], n_blocks=a["blocks"],
                            causal=not a["bidirectional"])
    model.load_state_dict(ck["state_dict"])
    model.to(torch.device(args.device)).eval()
    maxp = WindowConfig().max_players

    hits = list(PROC.glob(f"*{args.match}*.npz"))
    if not hits:
        print(f"No encontrado: {args.match}"); return 1
    m = Match.load(hits[0])
    if m.fps > a["fps"]:
        m = m.resample(a["fps"])
    if a["minutes"]:
        m = m.head_minutes(a["minutes"])

    print("simulando camara y baselines...")
    view = simulate(m, ViewportConfig(width_m=a["width"]))
    b4 = run_ladder(m, view, "B4", LadderConfig())
    print("infiriendo modelo (CPU ~20-30 s)...")
    pred = predict_match(model, m, view, b4, a["window"], maxp)
    print("estimando posesion...")
    poss = possession_team(m)

    s_b4 = stratified_median(position_errors(m, view, b4))
    s_md = stratified_median(position_errors(m, view, pred))
    headline = "Reconstruccion de jugadores fuera de camara — modo causal"
    print(f"global: B4 {s_b4['median_all']:.2f} -> modelo {s_md['median_all']:.2f} m")

    clip_frames = int(round(args.clip_seconds * m.fps))
    if args.start_s is not None:
        s0 = int(round(args.start_s * m.fps)); seg = (s0, s0 + clip_frames)
    else:
        seg = pick_segment(m, view, clip_frames, args.mode, poss)
    print(f"segmento [{args.mode}]: frames {seg[0]}-{seg[1]} (min {seg[0]/m.fps/60:.1f})")

    trail_frames = max(0, int(round(args.trail_seconds * m.fps)))
    ctx = build_context(m, view, b4, pred, seg, args.mode, poss, headline,
                        trail_frames=trail_frames,
                        thr_opt=args.thr_opt, thr_crit=args.thr_crit)

    outdir = VID / MODE_DIR[args.mode]
    outdir.mkdir(parents=True, exist_ok=True)

    # 16:9 a dpi 120 => 1920x1080 exactos (frames pares, compatibles yuv420p).
    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(C_FIG)
    # Layout maestro. La cancha domina; leyenda y tabla comparten la columna
    # derecha. 'left' NO puede ser 0: la franja necesita margen para sus numeros
    # de eje (si no, se recortan fuera del lienzo).
    gs = fig.add_gridspec(2, 2, height_ratios=[3.5, 1.05],
                          width_ratios=[2.85, 1.0], hspace=0.20, wspace=0.05,
                          left=0.055, right=0.985, top=0.905, bottom=0.09)
    ax_pitch = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_strip = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[1, 1])
    build_legend(ax_legend)   # estatica: una sola vez

    if args.preview:
        t = (seg[0] + seg[1]) // 2
        draw(ax_pitch, ax_strip, ax_table, ctx, t)
        out = args.out or str(outdir / f"preview_{args.match}_{args.theme}.png")
        # Sin bbox_inches="tight": conserva el encuadre 16:9 exacto (1920x1080)
        # que vera el video, para que lo que revisas sea lo que se renderiza.
        fig.savefig(out, dpi=120, facecolor=C_FIG)
        print(f"-> {out}"); return 0

    hold = max(1, int(round(args.fps_video / m.fps * args.slow)))
    n_local = seg[1] - seg[0]
    n_video = n_local * hold

    def update(vi):
        t = seg[0] + vi // hold
        draw(ax_pitch, ax_strip, ax_table, ctx, t)
        return []

    print(f"renderizando {n_video} frames ({n_local} del modelo x hold {hold}, slow {args.slow}x)...")
    ani = manim.FuncAnimation(fig, update, frames=n_video, blit=False)
    out = args.out or str(outdir / f"ghosting_{args.match}_{args.mode}_{args.theme}.mp4")

    # Barra de progreso con ETA
    # matplotlib llama progress_callback(i, n) por cada frame renderizado. Con
    # tqdm eso da una barra con velocidad (frame/s) y tiempo restante estimado.
    # Si no hay tqdm, cae a un porcentaje simple en una sola linea.
    try:
        from tqdm import tqdm
        _bar = tqdm(total=n_video, unit="fr", dynamic_ncols=True,
                    desc=f"{args.match}/{args.mode}")
        def _progress(i, n):
            _bar.update(1)
    except ImportError:
        _bar = None
        _tick = max(1, n_video // 50)
        def _progress(i, n):
            if i % _tick == 0 or i == n - 1:
                pct = 100 * (i + 1) // n
                print(f"\r  render {pct:3d}%  ({i + 1}/{n} frames)", end="", flush=True)
                if i == n - 1:
                    print()

    try:
        ani.save(out, writer=manim.FFMpegWriter(
            fps=args.fps_video, bitrate=6000,
            extra_args=["-pix_fmt", "yuv420p"]),   # compat VLC/Intel
            dpi=120,                               # 16x9 * 120 = 1920x1080 exactos
            savefig_kwargs={"facecolor": C_FIG},
            progress_callback=_progress)
    except Exception as ex:
        print(f"[ffmpeg fallo: {ex}] -> GIF")
        out = out.rsplit(".", 1)[0] + ".gif"
        ani.save(out, writer=manim.PillowWriter(fps=args.fps_video),
                 dpi=120,
                 savefig_kwargs={"facecolor": C_FIG},
                 progress_callback=_progress)
    finally:
        if _bar is not None:
            _bar.close()
    print(f"-> {out}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
