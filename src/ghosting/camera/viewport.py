"""
Simulador de la cámara principal de televisión.

Este módulo genera la máscara de observación M[t,i] que define QUÉ jugadores
"ve" una cámara de broadcast en cada instante. Es el cimiento estadístico de
todo el proyecto: si la máscara no se parece a la oclusión real, todos los
números aguas abajo son ficción.

Modelo base (réplica exacta de Choi 2026, arXiv:2607.11548)
----------------------------------------------------------
La cámara paneá horizontalmente siguiendo una versión suavizada del balón:

    c(t) = alpha * x_b(t) + (1 - alpha) * c(t-1)

con alpha = 0.06 por frame a 25 fps. El EMA modela el retraso del camarógrafo:
la cámara nunca está exactamente sobre el balón, va detrás.

La región visible es una ventana vertical de ancho W que abarca todo el alto
de la cancha:

    V_t = { i : |x_i(t) - c(t)| <= W/2 }

Con W = 44 m sobre datos de Metrica, Choi reporta 14.6-15.0 jugadores visibles
en promedio, consistente con los 12.8 +/- 3.7 de Omidshafiei et al. (2022) y
con los 10-16 observados en clips reales.

Extensiones sobre el modelo base
--------------------------------
Choi señala como limitación explícita que su viewport "paneá pero no hace zoom
ni tilt". Este módulo implementa dos extensiones opcionales, desactivadas por
defecto para preservar la comparabilidad numérica:

1. ZOOM: W(t) variable en función de la velocidad del balón. La cámara abre
   el plano en juego lento y lo cierra en jugadas rápidas cerca del área.
2. TILT: banda vertical de alto H centrada en el balón, que ocluye a los
   jugadores de la banda lejana.

El script `scripts/03_calibrate_camera.py` ajusta los parámetros de estas
extensiones contra las estadísticas de visibilidad REALES de SkillCorner
opendata, que es tracking de broadcast genuino y por tanto contiene la máscara
de oclusión observada, no simulada.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np

from ..io.schema import Match


# Constante de referencia: umbral de oclusión "larga" del paper de Choi.
# Corresponde a la longitud de ventana del Graph Imputer de DeepMind (9.6 s),
# más allá de la cual ese modelo no está definido.
LONG_OCCLUSION_S = 9.6


@dataclass(frozen=True)
class ViewportConfig:
    """
    Parámetros del simulador de cámara.

    Los valores por defecto reproducen exactamente el protocolo de Choi (2026).
    No los cambies si quieres que tus números sean comparables con los suyos.

    Attributes
    ----------
    width_m : float
        Ancho W de la ventana visible, en metros. El barrido de sensibilidad
        del paper usa {36, 44, 52, 60}.
    alpha : float
        Coeficiente del EMA de paneo, expresado POR FRAME A 25 FPS. El código
        lo reescala automáticamente si el partido tiene otro fps (ver
        `_alpha_for_fps`), de modo que la constante de tiempo física del
        paneo sea la misma sin importar la frecuencia de muestreo.
    gk_policy : {"occlude", "always_visible"}
        Cómo tratar a los porteros.
        - "occlude" (por defecto): el portero se ocluye como cualquier otro.
          Es lo fiel a un broadcast real y lo comparable con Choi.
        - "always_visible": el portero nunca se ocluye. Hace el problema más
          fácil de forma artificial; se ofrece solo para ablación.
    enable_zoom : bool
        Activa W(t) variable. Rompe comparabilidad con el paper.
    zoom_min_m, zoom_max_m : float
        Rango de W(t) cuando enable_zoom=True.
    zoom_speed_ref : float
        Velocidad del balón (m/s) a la que W(t) alcanza su valor mínimo.
    enable_tilt : bool
        Activa la banda vertical. Rompe comparabilidad con el paper.
    height_m : float
        Alto H de la banda visible cuando enable_tilt=True.
    """

    width_m: float = 44.0
    alpha: float = 0.06
    gk_policy: Literal["occlude", "always_visible"] = "occlude"

    enable_zoom: bool = False
    zoom_min_m: float = 32.0
    zoom_max_m: float = 60.0
    zoom_speed_ref: float = 12.0

    enable_tilt: bool = False
    height_m: float = 52.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ViewportResult:
    """
    Salida del simulador.

    Attributes
    ----------
    visible : np.ndarray, shape (T, N), dtype bool
        Máscara de observación M. True = la cámara ve al jugador.
        Un jugador fuera de cancha (on_pitch=False) siempre es invisible.
    center : np.ndarray, shape (T,), dtype float32
        Posición x del centro de cámara c(t), en metros.
    width : np.ndarray, shape (T,), dtype float32
        Ancho W(t) usado en cada frame. Constante si enable_zoom=False.
    gap_s : np.ndarray, shape (T, N), dtype float32
        Tiempo transcurrido desde la última observación del jugador i, en
        segundos, evaluado en el frame t.
        - 0.0 si el jugador es visible en t.
        - NaN si el jugador nunca ha sido visto antes en el periodo (cold
          start): no es imputable por posición y se excluye de esa métrica.
        Esta es LA variable de estratificación del proyecto entero.
    config : ViewportConfig
        Configuración usada, para trazabilidad.
    """

    visible: np.ndarray
    center: np.ndarray
    width: np.ndarray
    gap_s: np.ndarray
    config: ViewportConfig


def _alpha_for_fps(alpha_25: float, fps: float) -> float:
    """
    Reescala el coeficiente del EMA para preservar la constante de tiempo.

    Un EMA con coeficiente `a` por paso tiene constante de tiempo
    tau = -dt / ln(1 - a). Para que el paneo tenga la MISMA inercia física a
    cualquier fps, hay que resolver:

        (1 - a_fps) ** fps  ==  (1 - a_25) ** 25

    de donde:

        a_fps = 1 - (1 - a_25) ** (25 / fps)

    Sin esta corrección, evaluar a 5 fps con alpha=0.06 daría una cámara cinco
    veces más lenta que la del paper y las estadísticas de oclusión no
    coincidirían.
    """
    if abs(fps - 25.0) < 1e-9:
        return alpha_25
    return 1.0 - (1.0 - alpha_25) ** (25.0 / fps)


def _ema_center(ball_x: np.ndarray, alpha: float, period: np.ndarray) -> np.ndarray:
    """
    Centro de cámara por EMA causal, reiniciado en cada periodo.

    El reinicio por periodo importa: entre el primer y el segundo tiempo la
    cámara no "arrastra" su posición previa. Los NaN del balón se puentean
    manteniendo el último centro válido (la cámara no se mueve si perdió la
    referencia).
    """
    T = ball_x.shape[0]
    center = np.empty(T, dtype=np.float64)

    prev_period = None
    c = np.nan
    for t in range(T):
        p = period[t]
        if p != prev_period:
            # Nuevo periodo: la cámara arranca sobre el balón, o en el centro
            # de la cancha si el balón aún no está rastreado.
            c = ball_x[t] if np.isfinite(ball_x[t]) else np.nan
            prev_period = p
        elif np.isfinite(ball_x[t]):
            if np.isfinite(c):
                c = alpha * ball_x[t] + (1.0 - alpha) * c
            else:
                c = ball_x[t]
        # Si ball_x[t] es NaN, c se mantiene sin cambio.
        center[t] = c

    # Puentear NaN iniciales hacia adelante y hacia atrás
    if np.isnan(center).any():
        idx = np.arange(T)
        good = np.isfinite(center)
        if good.any():
            center = np.interp(idx, idx[good], center[good])
        else:
            center = np.zeros(T, dtype=np.float64)
    return center


def _ball_speed(ball: np.ndarray, fps: float) -> np.ndarray:
    """Rapidez del balón en m/s por diferencias centradas, robusta a NaN."""
    T = ball.shape[0]
    v = np.zeros(T, dtype=np.float64)
    if T < 3:
        return v
    d = np.linalg.norm(np.diff(ball, axis=0), axis=1) * fps
    d = np.nan_to_num(d, nan=0.0, posinf=0.0)
    v[1:-1] = 0.5 * (d[:-1] + d[1:])
    v[0] = d[0]
    v[-1] = d[-1]
    # Suavizado de 1 s para que el zoom no tiemble frame a frame
    k = max(1, int(round(fps)))
    kernel = np.ones(k) / k
    return np.convolve(v, kernel, mode="same")


def compute_gap_seconds(
    visible: np.ndarray, on_pitch: np.ndarray, period: np.ndarray, fps: float
) -> np.ndarray:
    """
    Tiempo desde la última observación, por jugador y frame.

    Reglas
    ------
    - Si el jugador es visible en t, gap = 0.
    - Si es invisible pero fue visto antes EN EL MISMO PERIODO, gap = tiempo
      transcurrido desde esa última vista.
    - Si nunca fue visto antes en el periodo (cold start), gap = NaN. Estos
      casos no son puntuables por error de posición porque no hay ninguna
      observación previa de la cual partir; Choi los trata igual y los excluye
      del denominador de la métrica de posición.
    - Si no está en cancha, gap = NaN.

    El reinicio por periodo es deliberado: tras el descanso los equipos cambian
    de lado, así que una observación del primer tiempo no informa sobre la
    posición en el segundo.
    """
    T, N = visible.shape
    gap = np.full((T, N), np.nan, dtype=np.float32)
    dt = 1.0 / fps

    last_seen = np.full(N, -1, dtype=np.int64)
    prev_period = None

    for t in range(T):
        if period[t] != prev_period:
            last_seen[:] = -1
            prev_period = period[t]

        vis_t = visible[t]
        on_t = on_pitch[t]

        # Visibles: gap cero y se actualiza el reloj
        gap[t, vis_t] = 0.0
        last_seen[vis_t] = t

        # Ocultos con historia: gap positivo
        hidden = on_t & ~vis_t & (last_seen >= 0)
        if hidden.any():
            gap[t, hidden] = (t - last_seen[hidden]).astype(np.float32) * dt

        # Fuera de cancha: NaN (ya inicializado)
        gap[t, ~on_t] = np.nan

    return gap


def simulate(match: Match, config: ViewportConfig | None = None) -> ViewportResult:
    """
    Aplica el modelo de cámara a un partido y devuelve la máscara de observación.

    Parameters
    ----------
    match : Match
        Partido en esquema canónico.
    config : ViewportConfig, optional
        Parámetros de cámara. Por defecto, el protocolo de Choi (2026).

    Returns
    -------
    ViewportResult

    Examples
    --------
    >>> from ghosting.io.schema import Match
    >>> from ghosting.camera.viewport import simulate, ViewportConfig
    >>> res = simulate(match, ViewportConfig(width_m=44.0))
    >>> res.visible.sum(axis=1).mean()   # jugadores visibles en promedio
    """
    cfg = config or ViewportConfig()
    T, N = match.n_frames, match.n_players


    alpha = _alpha_for_fps(cfg.alpha, match.fps)
    center = _ema_center(match.ball[:, 0], alpha, match.period)


    if cfg.enable_zoom:
        speed = _ball_speed(match.ball, match.fps)
        # Interpolación lineal: balón quieto -> plano abierto,
        # balón rápido -> plano cerrado. Saturada en el rango configurado.
        frac = np.clip(speed / cfg.zoom_speed_ref, 0.0, 1.0)
        width = cfg.zoom_max_m - frac * (cfg.zoom_max_m - cfg.zoom_min_m)
    else:
        width = np.full(T, cfg.width_m, dtype=np.float64)


    x = match.positions[:, :, 0]
    dx = np.abs(x - center[:, None])
    visible = dx <= (width[:, None] / 2.0)

    if cfg.enable_tilt:
        cy = match.ball[:, 1]
        cy = np.where(np.isfinite(cy), cy, match.pitch[1] / 2.0)
        dy = np.abs(match.positions[:, :, 1] - cy[:, None])
        visible &= dy <= (cfg.height_m / 2.0)

    # Un jugador con posición NaN nunca es visible
    visible &= np.isfinite(x)
    # Un jugador fuera de cancha nunca es visible
    visible &= match.on_pitch

    # Política de portero
    if cfg.gk_policy == "always_visible":
        visible[:, match.is_gk] = match.on_pitch[:, match.is_gk]


    gap_s = compute_gap_seconds(visible, match.on_pitch, match.period, match.fps)

    return ViewportResult(
        visible=visible,
        center=center.astype(np.float32),
        width=width.astype(np.float32),
        gap_s=gap_s,
        config=cfg,
    )


def occlusion_stats(match: Match, res: ViewportResult, alive_only: bool = True) -> dict:
    """
    Estadísticas descriptivas de la oclusión inducida por la cámara.

    Estas son las cifras de validación del Paso 1. Si `visible_mean` no cae
    en el rango 14-16 con W=44 m, el simulador o las coordenadas están mal.

    Parameters
    ----------
    alive_only : bool
        Si True, restringe a frames con balón en juego. Recomendado: en balón
        muerto la cámara hace primeros planos y repeticiones que este modelo
        no representa.

    Returns
    -------
    dict con las claves:
        n_frames, visible_mean, visible_std, visible_p05, visible_p95,
        hidden_share_le2s, hidden_share_2_96s, hidden_share_gt96s,
        cold_start_share, gap_median_s, gap_p90_s, gap_max_s
    """
    sel = match.ball_alive if alive_only else np.ones(match.n_frames, dtype=bool)

    vis = res.visible[sel]
    n_vis = vis.sum(axis=1).astype(np.float64)

    gap = res.gap_s[sel]
    on = match.on_pitch[sel]

    # Población de muestras "ocultas": en cancha, no visible.
    hidden_mask = on & ~vis
    gaps_hidden = gap[hidden_mask]

    # Cold start = oculto y nunca visto antes en el periodo
    cold = np.isnan(gaps_hidden)
    warm = gaps_hidden[~cold]

    n_warm = warm.size
    if n_warm == 0:
        bins = (np.nan, np.nan, np.nan)
        gmed = gp90 = gmax = np.nan
    else:
        bins = (
            float((warm <= 2.0).mean()),
            float(((warm > 2.0) & (warm <= LONG_OCCLUSION_S)).mean()),
            float((warm > LONG_OCCLUSION_S).mean()),
        )
        gmed = float(np.median(warm))
        gp90 = float(np.percentile(warm, 90))
        gmax = float(warm.max())

    return {
        "n_frames": int(sel.sum()),
        "visible_mean": float(n_vis.mean()),
        "visible_std": float(n_vis.std()),
        "visible_p05": float(np.percentile(n_vis, 5)),
        "visible_p95": float(np.percentile(n_vis, 95)),
        "hidden_share_le2s": bins[0],
        "hidden_share_2_96s": bins[1],
        "hidden_share_gt96s": bins[2],
        "cold_start_share": float(cold.mean()) if gaps_hidden.size else np.nan,
        "gap_median_s": gmed,
        "gap_p90_s": gp90,
        "gap_max_s": gmax,
    }
