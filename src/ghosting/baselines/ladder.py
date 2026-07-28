"""
Escalera de imputación sin entrenamiento (baselines B0-B4).

Réplica de la escalera de Choi (2026, arXiv:2607.11548), con una extensión
propia: el manejo diferenciado del portero (política GK, ver más abajo).

Todos los métodos son CAUSALES: en el frame t solo usan observaciones de
t' <= t. Ninguno requiere entrenamiento ni GPU. Corren más rápido que tiempo
real en un núcleo de CPU.

Por qué importan estos baselines
--------------------------------
Dos razones, y la segunda es la que suele olvidarse:

1. Son el piso que un modelo aprendido debe superar. Si tu red neuronal no le
   gana a B4, no tienes un resultado: tienes un modelo caro que empata con
   aritmética.
2. Reproducirlos sobre tus propios datos es la validación de tu pipeline. Si
   tus números de B0-B4 sobre Metrica no coinciden con los del paper, el bug
   está en tu carga de datos o en tu simulador de cámara, no en tu modelo. Es
   la única forma de detectar ese tipo de error antes de que contamine todo.

La escalera
-----------
B0  ignorar          : el jugador oculto simplemente no existe. Es lo que hace
                       implícitamente cualquier pipeline de GSR sin capa de
                       imputación.
B1  última vista     : mantener la última posición observada, decayendo hacia
                       la media del equipo visible con constante tau.
B2  ancla de formación: guardar el desplazamiento del jugador respecto al
                       centroide visible mientras se le ve; al ocultarse,
                       colocarlo en (centroide visible actual + ese offset).
B5  plantilla fija   : como B2 pero con el offset promediado sobre todo el
                       historial en vez del último. Es el análogo online de
                       una plantilla estática de formación. Resulta MUCHO peor
                       que B2, lo que evidencia deriva de rol dentro del mismo
                       tiempo.
B3  B2 + EMA + vel   : offsets suavizados por EMA más extrapolación a
                       velocidad constante para oclusiones recientes.
B4  voto de centroide: el mejor. Ver la sección siguiente.

Por qué funciona B4 (la intuición, sin fórmulas)
------------------------------------------------
El centroide del equipo VISIBLE está sesgado: cuando la cámara apunta a la
izquierda, los jugadores visibles de un equipo son los que están a la
izquierda, así que su promedio queda a la izquierda del centroide verdadero.
B2 hereda ese sesgo y lo arrastra a cada jugador que imputa.

B4 lo corrige por votación: cada jugador visible propone dónde estaría el
centroide COMPLETO restándose su propio desplazamiento de rol. Como el sesgo
individual se resta antes de promediar, se atenúa en vez de acumularse. Luego
los offsets se vuelven a estimar contra ese centroide votado, no contra el
visible: estimación y aplicación quedan en la misma referencia.

Partición del portero (extensión propia)
----------------------------------------
La dinámica del portero no está gobernada por el centroide de su equipo sino
por un atractor estacionario: su propia portería. Un modelo homogéneo arrastra
al portero campo arriba cuando el equipo ataca, que es exactamente cuando el
portero está oculto y cuando el error importa.

La corrección NO es forzar al portero a visible (eso sería falsear la física
del broadcast: en un partido real el portero del lado lejano desaparece de
cuadro durante minutos, y además rompería la comparabilidad con Choi). La
corrección es anclarlo distinto: `gk_anchor="goal"` imputa al portero
mediante una contracción hacia su portería con constante de tiempo propia.

Las métricas se reportan siempre con y sin porteros por separado.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..io.schema import Match
from ..camera.viewport import ViewportResult


@dataclass(frozen=True)
class LadderConfig:
    """
    Hiperparámetros de la escalera.

    Los valores por defecto son los de Choi (2026), que aclara que son "valores
    redondos fijados una vez durante el desarrollo y no optimizados por
    búsqueda". No los ajustes contra tu conjunto de test.
    """

    tau_decay_s: float = 8.0        # B1: constante de decaimiento hacia la media
    ema_weight: float = 0.1         # B3/B4: peso del EMA de offsets (por paso a 5 fps)
    vel_blend_s: float = 1.5        # B3: constante de mezcla de extrapolación
    min_voters: int = 3             # B4: votantes mínimos antes de usar la votación
    gk_anchor: Literal["team", "goal"] = "team"
    gk_tau_s: float = 3.0           # constante de contracción del portero a su meta




def _ema_weight_for_fps(w_5fps: float, fps: float) -> float:
    """
    Reescala el peso del EMA para preservar la constante de tiempo física.

    Mismo razonamiento que `_alpha_for_fps` en camera/viewport.py: el paper
    define el peso a 5 fps, así que a otra frecuencia hay que resolver
    (1-w_fps)**fps == (1-w_5)**5.
    """
    if abs(fps - 5.0) < 1e-9:
        return w_5fps
    return 1.0 - (1.0 - w_5fps) ** (5.0 / fps)


def _goal_position(match: Match, team: int) -> np.ndarray:
    """
    Centro de la portería que defiende `team`, por periodo.

    Se estima empíricamente: la portería defendida es la más cercana a la
    posición media del portero de ese equipo. Así se resuelve solo el cambio
    de lado en el descanso, sin depender de metadatos del proveedor.
    """
    L, A = match.pitch
    out = {}
    gk_mask = match.is_gk & (match.team_idx == team)
    for p in np.unique(match.period):
        sel = match.period == p
        xs = match.positions[sel][:, gk_mask, 0]
        xs = xs[np.isfinite(xs)]
        mean_x = float(np.mean(xs)) if xs.size else L / 2
        out[int(p)] = np.array([0.0 if mean_x < L / 2 else L, A / 2])
    return out




def run_ladder(
    match: Match,
    view: ViewportResult,
    method: str,
    config: LadderConfig | None = None,
) -> np.ndarray:
    """
    Ejecuta un método de la escalera y devuelve las posiciones imputadas.

    Parameters
    ----------
    match : Match
    view : ViewportResult
        Salida de `camera.viewport.simulate`.
    method : {"B0", "B1", "B2", "B3", "B3E", "B3V", "B4", "B5"}
    config : LadderConfig, optional

    Returns
    -------
    np.ndarray, shape (T, N, 2)
        Posiciones estimadas. NaN donde el método no produce estimación
        (B0 siempre; los demás en cold start o fuera de cancha). Donde el
        jugador es visible, se devuelve su posición observada.

    Notes
    -----
    Implementación deliberadamente iterativa por frame, no vectorizada: los
    métodos son recursivos (el offset del frame t depende del centroide votado
    en t, que depende de los offsets de t-1) y la claridad importa más que la
    velocidad aquí. Un partido de 90 min a 5 fps son 27k frames: segundos.
    """
    cfg = config or LadderConfig()
    valid = {"B0", "B1", "B2", "B3", "B3E", "B3V", "B4", "B5"}
    if method not in valid:
        raise ValueError(f"Método '{method}' desconocido. Válidos: {sorted(valid)}")

    T, N = match.n_frames, match.n_players
    fps = match.fps
    dt = 1.0 / fps
    est = np.full((T, N, 2), np.nan, dtype=np.float32)

    if method == "B0":
        # B0 no imputa nada; solo copia lo visible.
        est[view.visible] = match.positions[view.visible]
        return est

    w_ema = _ema_weight_for_fps(cfg.ema_weight, fps)
    use_ema = method in {"B3", "B3E", "B4"}
    use_vel = method in {"B3", "B3V"}
    use_vote = method == "B4"
    cumulative = method == "B5"

    goals = {0: _goal_position(match, 0), 1: _goal_position(match, 1)}

    # Estado por jugador
    last_pos = np.full((N, 2), np.nan, dtype=np.float64)
    last_vel = np.zeros((N, 2), dtype=np.float64)
    last_t = np.full(N, -1, dtype=np.int64)
    offset = np.full((N, 2), np.nan, dtype=np.float64)
    off_sum = np.zeros((N, 2), dtype=np.float64)   # para B5
    off_cnt = np.zeros(N, dtype=np.int64)
    prev_period = None

    for t in range(T):
        if match.period[t] != prev_period:
            last_pos[:] = np.nan
            last_vel[:] = 0.0
            last_t[:] = -1
            offset[:] = np.nan
            off_sum[:] = 0.0
            off_cnt[:] = 0
            prev_period = match.period[t]

        vis_t = view.visible[t]
        pos_t = match.positions[t].astype(np.float64)

        for team in (0, 1):
            in_team = match.team_idx == team
            vis_team = vis_t & in_team
            hidden_team = in_team & match.on_pitch[t] & ~vis_t

            if not vis_team.any():
                continue

            visible_centroid = pos_t[vis_team].mean(axis=0)


            if use_vote:
                voters = vis_team & np.isfinite(offset[:, 0])
                if voters.sum() >= cfg.min_voters:
                    centroid = (pos_t[voters] - offset[voters]).mean(axis=0)
                else:
                    centroid = visible_centroid   # fallback a B2
            else:
                centroid = visible_centroid


            new_off = pos_t[vis_team] - centroid
            idxs = np.where(vis_team)[0]
            for k, o in zip(idxs, new_off):
                if cumulative:
                    off_sum[k] += o
                    off_cnt[k] += 1
                    offset[k] = off_sum[k] / off_cnt[k]
                elif use_ema and np.isfinite(offset[k, 0]):
                    offset[k] = (1 - w_ema) * offset[k] + w_ema * o
                else:
                    offset[k] = o


            for k in np.where(hidden_team)[0]:
                if last_t[k] < 0:
                    continue  # cold start: sin estimación

                gap = (t - last_t[k]) * dt

                if match.is_gk[k] and cfg.gk_anchor == "goal":
                    # Contracción exponencial hacia la portería propia.
                    g = goals[team][int(match.period[t])]
                    wg = np.exp(-gap / cfg.gk_tau_s)
                    est[t, k] = wg * last_pos[k] + (1 - wg) * g
                    continue

                if method == "B1":
                    w = np.exp(-gap / cfg.tau_decay_s)
                    est[t, k] = w * last_pos[k] + (1 - w) * visible_centroid
                    continue

                if not np.isfinite(offset[k, 0]):
                    # Sin offset almacenado: caer a B1
                    w = np.exp(-gap / cfg.tau_decay_s)
                    est[t, k] = w * last_pos[k] + (1 - w) * visible_centroid
                    continue

                anchor = centroid if use_vote else visible_centroid
                p = anchor + offset[k]

                if use_vel:
                    wv = np.exp(-gap / cfg.vel_blend_s)
                    p = wv * (last_pos[k] + last_vel[k] * gap) + (1 - wv) * p

                est[t, k] = p


        idxs = np.where(vis_t)[0]
        for k in idxs:
            if last_t[k] >= 0:
                d = (t - last_t[k]) * dt
                if d > 0:
                    v = (pos_t[k] - last_pos[k]) / d
                    sp = np.linalg.norm(v)
                    if sp > 11.0:      # tope físico de velocidad humana
                        v *= 11.0 / sp
                    last_vel[k] = v
            last_pos[k] = pos_t[k]
            last_t[k] = t
        est[t, vis_t] = match.positions[t, vis_t]

    # Recortar a los límites de la cancha: nadie se imputa en la tribuna
    L, A = match.pitch
    np.clip(est[..., 0], 0.0, L, out=est[..., 0])
    np.clip(est[..., 1], 0.0, A, out=est[..., 1])
    return est


ALL_METHODS = ["B0", "B1", "B2", "B5", "B3E", "B3V", "B3", "B4"]
