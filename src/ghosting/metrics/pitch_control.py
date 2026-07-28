"""
Control de cancha (pitch control).

Qué es y por qué importa
------------------------
El error de posición en metros es la métrica correcta para medir el modelo,
pero no dice nada a un entrenador. El control de cancha sí: es el mapa de qué
equipo llegaría antes a cada punto del campo. De ahí salen las preguntas que un
cuerpo técnico se hace de verdad — dónde hay espacio, quién domina qué zona,
si el bloque está compacto.

Y es donde el problema se ve de un vistazo: si ignoras a los jugadores que la
cámara no muestra, el mapa asigna al equipo equivocado zonas enteras donde en
realidad había un defensor que no se veía.

Modelo
------
Versión simplificada del modelo de Spearman (2017), en la línea de la
implementación pública de Laurie Shaw. Para cada punto r de una grilla, el
tiempo de llegada del jugador i es:

    tti_i(r) = t_react + || r - (p_i + v_i · t_react) || / v_max

es decir: el jugador sigue su velocidad actual durante el tiempo de reacción y
después corre en línea recta a velocidad máxima. El control del equipo local es

    C(r) = sigmoide( ( min_j∈visitante tti_j - min_i∈local tti_i ) / tau )

con tau el parámetro que suaviza la frontera: si los dos equipos llegan casi a
la vez, el control tiende a 0.5.

LIMITACIONES, porque esto va a una presentación
-----------------------------------------------
Es una simplificación deliberada. El modelo completo de Spearman integra sobre
la incertidumbre de la trayectoria del balón y modela la probabilidad de
control como un proceso en el tiempo. Aquí no hay balón: es puramente
geométrico-cinemático, "quién llega antes".

Sirve perfectamente para lo que se usa —comparar el MISMO instante con
distintos conjuntos de jugadores— porque el sesgo del modelo es idéntico en
los tres paneles y se cancela al comparar. No sirve para reportar valores
absolutos de control como si fueran probabilidades calibradas.
"""

from __future__ import annotations

import numpy as np

V_MAX = 11.0        # m/s, tope de velocidad humana sostenida
T_REACT = 0.7       # s, tiempo de reacción antes de arrancar
TAU = 0.45          # s, suavizado de la frontera de control


def make_grid(pitch: tuple[float, float], step: float = 2.0):
    """Grilla regular sobre la cancha. `step` en metros."""
    L, A = pitch
    xs = np.arange(step / 2, L, step)
    ys = np.arange(step / 2, A, step)
    gx, gy = np.meshgrid(xs, ys)
    return xs, ys, np.stack([gx.ravel(), gy.ravel()], axis=1)


def time_to_intercept(points: np.ndarray, pos: np.ndarray,
                      vel: np.ndarray | None = None) -> np.ndarray:
    """
    Tiempo de llegada de cada jugador a cada punto.

    Parameters
    ----------
    points : (G, 2)   puntos de la grilla
    pos    : (P, 2)   posiciones de los jugadores
    vel    : (P, 2)   velocidades, o None para tratarlos como parados

    Returns
    -------
    (P, G) tiempos en segundos
    """
    if pos.size == 0:
        return np.full((0, points.shape[0]), np.inf)
    start = pos if vel is None else pos + vel * T_REACT
    d = np.linalg.norm(points[None, :, :] - start[:, None, :], axis=2)
    return T_REACT + d / V_MAX


def pitch_control(
    points: np.ndarray,
    pos_home: np.ndarray, pos_away: np.ndarray,
    vel_home: np.ndarray | None = None, vel_away: np.ndarray | None = None,
) -> np.ndarray:
    """
    Control del equipo local en cada punto, en [0, 1].

    Devuelve 0.5 en todos lados si algún equipo se queda sin jugadores (no hay
    información para decidir), que es el comportamiento neutro correcto.
    """
    th = time_to_intercept(points, pos_home, vel_home)
    ta = time_to_intercept(points, pos_away, vel_away)
    if th.shape[0] == 0 or ta.shape[0] == 0:
        return np.full(points.shape[0], 0.5)
    return 1.0 / (1.0 + np.exp(-(ta.min(axis=0) - th.min(axis=0)) / TAU))


def control_share(control: np.ndarray) -> float:
    """Fracción de la cancha controlada por el local. Es el número de titular."""
    return float(np.mean(control > 0.5))


def control_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """
    Error absoluto medio entre dos mapas, en puntos porcentuales.

    `mask` permite restringir a una región — típicamente la zona oculta por la
    cámara, que es donde el efecto de ignorar jugadores se concentra y donde
    Choi (2026) reporta sus cifras.
    """
    d = np.abs(a - b)
    if mask is not None:
        d = d[mask]
    return float(np.mean(d) * 100)
