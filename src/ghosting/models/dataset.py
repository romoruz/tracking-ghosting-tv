"""
Dataset de ventanas para el imputador residual.

Qué predice el modelo
---------------------
NO la posición absoluta, sino el **residuo sobre B4**:

    p̂_j(t) = B4_j(t) + f_θ(...)_j(t)

Tres razones, y la primera es la que manda con 7 partidos de datos:

1. Aprender una corrección converge mucho más rápido que aprender la función
   completa. B4 ya explica la mayor parte de la señal ("el equipo se desplazó y
   el jugador conservó su rol"); el modelo solo tiene que aprender en qué se
   equivoca.
2. El modelo no puede ser catastróficamente peor que el baseline: si f_θ → 0,
   se recupera B4 exactamente. Eso acota el riesgo a la baja.
3. La comparación es directa e interpretable: el residuo *es* la mejora, y se
   mide con el mismo bootstrap pareado que ya está validado.

Entrada
-------
El tensor de entrada es "el mejor estado conocido" en cada instante:
- jugador visible  -> su posición REAL
- jugador oculto   -> la estimación de B4

más la máscara y el gap, que le dicen al modelo cuáles de esas entradas son
observación y cuáles son conjetura, y desde hace cuánto.

Esa es la razón de que B4 se calcule sobre el partido COMPLETO antes de
trocear: sus offsets son recursivos y se acumulan a lo largo del tiempo.
Calcularlo por ventana reiniciaría el estado y produciría un baseline peor —
que es exactamente el sesgo que infla artificialmente la mejora aparente del
modelo.

Partición
---------
Por partido, nunca por ventana. A 5 fps con solapamiento, dos ventanas vecinas
comparten casi todos sus frames; separarlas entre train y test daría números
de fantasía.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.schema import Match
from ..camera.viewport import ViewportResult

# Cota de velocidad humana usada para normalizar y para las restricciones.
V_MAX = 11.0    # m/s
A_MAX = 7.0     # m/s^2

# Escala del gap: los huecos van de 0.2 s a varios minutos, así que se
# comprime logarítmicamente. Sin esto, un gap de 300 s domina la entrada.
GAP_SCALE = 10.0


@dataclass(frozen=True)
class WindowConfig:
    """
    Parámetros del troceado.

    Attributes
    ----------
    length : int
        Frames por ventana. 50 a 5 fps = 10 s.
    stride : int
        Desplazamiento entre ventanas consecutivas. Menor stride = más
        ventanas y más solapamiento; no aporta información nueva pero sí
        estabiliza el gradiente.
    min_hidden : int
        Ventanas con menos de esta cantidad de pares (frame, jugador) ocultos
        y puntuables se descartan: no aportan señal de entrenamiento.
    max_players : int
        Tamaño al que se rellena el eje de jugadores. Los rosters de Sportec
        listan hasta 40 jugadores por partido (20 por equipo con suplentes);
        solo ~22 están en cancha simultáneamente, el resto va enmascarado.
    """

    length: int = 50
    stride: int = 10
    min_hidden: int = 20
    max_players: int = 26


# Nombres de las features de entrada, en orden. Documentar el orden importa:
# un desalineamiento aquí es un bug silencioso que degrada el modelo sin fallar.
FEATURES = [
    "x_norm",        # posición conocida (real si visible, B4 si oculto), normalizada
    "y_norm",
    "vx_norm",       # velocidad estimada desde la última observación
    "vy_norm",
    "visible",       # 1 = observación, 0 = conjetura de B4
    "gap_log",       # log(1 + gap/GAP_SCALE); 0 si visible
    "dx_ball",       # posición relativa al balón
    "dy_ball",
    "dx_centroid",   # posición relativa al centroide del propio equipo visible
    "dy_centroid",
    "team",          # 0 local / 1 visitante
    "is_gk",
    "ball_x",        # balón en absoluto (repetido por jugador: barato y útil)
    "ball_y",
    "on_pitch",      # 0 = relleno o jugador fuera de cancha
    # Ancla de largo alcance
    # Última posición REALMENTE observada del jugador, y su desplazamiento
    # respecto a la estimación actual de B4. Sin esto, para un jugador oculto
    # más tiempo que la ventana el modelo no tiene ninguna observación suya:
    # solo ve la estimación de B4 repetida, que ya ha "suavizado" la última
    # vista. Estas tres features son el único enlace con lo que de verdad se
    # vio, y son gratis (se calculan con un barrido lineal).
    "last_x",
    "last_y",
    "d_last",        # distancia entre la última vista y la estimación de B4
]
N_FEATURES = len(FEATURES)


def build_features(
    match: Match,
    view: ViewportResult,
    baseline: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Construye los tensores por partido, antes de trocear.

    Parameters
    ----------
    baseline : (T, N, 2)
        Salida de `run_ladder(..., "B4")` sobre el partido COMPLETO.

    Returns
    -------
    dict con:
        feats    : (T, N, N_FEATURES) float32
        target   : (T, N, 2) float32 -- residuo verdad - baseline, 0 si no puntuable
        loss_mask: (T, N) bool -- dónde se calcula la pérdida
        base     : (T, N, 2) float32 -- el baseline, para reconstruir la predicción
    """
    T, N = match.n_frames, match.n_players
    L, A = match.pitch

    # Estado conocido: real donde se ve, baseline donde no.
    known = np.where(
        view.visible[..., None], match.positions, baseline
    ).astype(np.float32)
    # Donde el baseline tampoco tiene nada (cold start), se usa el centro de
    # cancha como relleno neutro; esos pares quedan fuera de la pérdida.
    fill = np.array([L / 2, A / 2], dtype=np.float32)
    known = np.where(np.isfinite(known), known, fill)

    # Velocidad por diferencias hacia atrás sobre el estado conocido.
    vel = np.zeros_like(known)
    vel[1:] = (known[1:] - known[:-1]) * match.fps
    sp = np.linalg.norm(vel, axis=2, keepdims=True)
    vel = np.where(sp > V_MAX, vel * (V_MAX / np.maximum(sp, 1e-6)), vel)

    ball = np.where(np.isfinite(match.ball), match.ball, fill).astype(np.float32)

    # Centroide del equipo visible, por equipo y frame.
    centroid = np.zeros((T, N, 2), dtype=np.float32)
    for team in (0, 1):
        in_team = match.team_idx == team
        vis = view.visible & in_team[None, :]
        cnt = vis.sum(axis=1, keepdims=True).astype(np.float32)
        summ = (known * vis[..., None]).sum(axis=1)
        c = np.divide(summ, np.maximum(cnt, 1.0), dtype=np.float32)
        c = np.where(cnt > 0, c, fill)
        centroid[:, in_team, :] = c[:, None, :]

    gap = np.nan_to_num(view.gap_s, nan=0.0)

    # Última posición observada, arrastrada hacia adelante y reiniciada por
    # periodo (tras el descanso los equipos cambian de lado).
    last = np.full((T, N, 2), np.nan, dtype=np.float32)
    prev_period = None
    cur = np.full((N, 2), np.nan, dtype=np.float32)
    for t in range(T):
        if match.period[t] != prev_period:
            cur[:] = np.nan
            prev_period = match.period[t]
        vis_t = view.visible[t]
        cur[vis_t] = match.positions[t, vis_t]
        last[t] = cur
    last = np.where(np.isfinite(last), last, known)

    f = np.zeros((T, N, N_FEATURES), dtype=np.float32)
    f[..., 0] = known[..., 0] / L
    f[..., 1] = known[..., 1] / A
    f[..., 2] = vel[..., 0] / V_MAX
    f[..., 3] = vel[..., 1] / V_MAX
    f[..., 4] = view.visible
    f[..., 5] = np.log1p(gap / GAP_SCALE)
    f[..., 6] = (known[..., 0] - ball[:, None, 0]) / L
    f[..., 7] = (known[..., 1] - ball[:, None, 1]) / A
    f[..., 8] = (known[..., 0] - centroid[..., 0]) / L
    f[..., 9] = (known[..., 1] - centroid[..., 1]) / A
    f[..., 10] = match.team_idx[None, :]
    f[..., 11] = match.is_gk[None, :]
    f[..., 12] = (ball[:, 0] / L)[:, None]
    f[..., 13] = (ball[:, 1] / A)[:, None]
    f[..., 14] = match.on_pitch
    f[..., 15] = last[..., 0] / L
    f[..., 16] = last[..., 1] / A
    f[..., 17] = np.linalg.norm(known - last, axis=2) / L

    # Pérdida solo donde hay algo que corregir y verdad con la que comparar.
    loss_mask = (
        match.on_pitch
        & ~view.visible
        & np.isfinite(view.gap_s)
        & np.isfinite(baseline[..., 0])
        & np.isfinite(match.positions[..., 0])
    )

    target = np.zeros((T, N, 2), dtype=np.float32)
    target[loss_mask] = (match.positions[loss_mask] - baseline[loss_mask])

    base = np.where(np.isfinite(baseline), baseline, fill).astype(np.float32)
    return {"feats": f, "target": target, "loss_mask": loss_mask, "base": base}


def cut_windows(
    per_match: dict[str, np.ndarray], cfg: WindowConfig
) -> list[dict[str, np.ndarray]]:
    """Trocea los tensores de un partido en ventanas solapadas."""
    T, N = per_match["loss_mask"].shape
    out = []
    for s in range(0, T - cfg.length + 1, cfg.stride):
        e = s + cfg.length
        lm = per_match["loss_mask"][s:e]
        if lm.sum() < cfg.min_hidden:
            continue
        out.append({
            "feats": per_match["feats"][s:e],
            "target": per_match["target"][s:e],
            "loss_mask": lm,
            "base": per_match["base"][s:e],
        })
    return out


def pad_players(w: dict[str, np.ndarray], max_players: int) -> dict[str, np.ndarray]:
    """
    Rellena o recorta el eje de jugadores a un tamaño fijo.

    Los rosters varían entre partidos (26 a 40 en los datos abiertos) pero solo
    ~22 están en cancha. Se conservan los que aparecen en cancha en algún frame
    de la ventana, priorizando los que tienen pares puntuables; el resto se
    rellena con ceros y `player_mask=False` para que la atención los ignore.
    """
    T, N = w["loss_mask"].shape
    on = w["feats"][..., FEATURES.index("on_pitch")].max(axis=0) > 0
    score = w["loss_mask"].sum(axis=0) + on * 1e6      # en cancha primero
    keep = np.argsort(-score)[:max_players]
    keep = np.sort(keep)

    def take(arr):
        sel = arr[:, keep]
        if keep.size < max_players:
            pad = [(0, 0), (0, max_players - keep.size)] + [(0, 0)] * (arr.ndim - 2)
            sel = np.pad(sel, pad)
        return sel

    pm = np.zeros(max_players, dtype=bool)
    pm[: keep.size] = on[keep]
    return {
        "feats": take(w["feats"]),
        "target": take(w["target"]),
        "loss_mask": take(w["loss_mask"]),
        "base": take(w["base"]),
        "player_mask": pm,
    }


def build_windows(
    match: Match,
    view: ViewportResult,
    baseline: np.ndarray,
    cfg: WindowConfig | None = None,
) -> list[dict[str, np.ndarray]]:
    """Pipeline completo de un partido a lista de ventanas listas para el modelo."""
    cfg = cfg or WindowConfig()
    per_match = build_features(match, view, baseline)
    return [pad_players(w, cfg.max_players) for w in cut_windows(per_match, cfg)]
