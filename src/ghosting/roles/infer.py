"""
Inferencia de roles por geometría, sin metadatos de formación.

Motivación
----------
El .npz canónico NO guarda la formación ni la posición nominal de cada jugador
(esos son metadatos privativos del proveedor y no se redistribuyen). Aun así,
para una figura de "error por rol" hace falta una etiqueta de rol. Se infiere
por geometría a partir de lo único que ya está en el .npz: las posiciones.

Esto es una VIRTUD, no un parche: demuestra que el análisis opera sobre tracking
crudo y es agnóstico a metadatos que un club nos daría o no. Por eso toda figura
que use estos roles debe etiquetarse "(inferencia geométrica)".

Criterio (sobre los frames VISIBLES del periodo)
------------------------------------------------
1. Portero: ya viene resuelto en `match.is_gk` (inferencia por periodo, ver
   io/loaders.py). No se reclasifica.
2. Eje X (líneas): K-Means(k=3) sobre la mediana longitudinal x̃_i separa
   defensa / mediocampo / ataque. Fallback a cuantiles si sklearn no está.
3. Eje Y (carriles): dentro de cada línea, la distancia absoluta de ỹ_i al eje
   longitudinal central de la cancha separa centrales de laterales/extremos.

Por qué la MEDIANA y no la media
--------------------------------
La mediana por jugador es robusta a las salidas puntuales (un central que sube a
un córner, un lateral que se mete al medio en una jugada). La media se dejaría
arrastrar por esos episodios y mezclaría carriles.

Por qué sobre los frames VISIBLES
---------------------------------
Es deliberado y honesto: el rol se infiere solo con lo que la cámara mostró, la
misma información de la que dispondría un pipeline real. La posición imputada (la
que luego se puntúa por error) NO entra aquí, así que no hay circularidad entre
"de dónde sale el rol" y "qué se está midiendo".

Limitación conocida
-------------------
Un jugador poco visible (defensa del lado lejano) tiene pocas muestras y su
mediana puede sesgarse hacia los instantes en que estuvo cerca del balón. Para
una clasificación gruesa (línea + carril) es suficiente; no pretende reconstruir
la formación táctica exacta.
"""

from __future__ import annotations

import numpy as np

from ..io.schema import Match
from ..camera.viewport import ViewportResult

# Taxonomía deliberadamente de CENTRAL-vs-ABIERTO, NO de izquierda-vs-derecha.
# Distinguir L/R exigiría conocer la dirección de ataque de cada equipo en cada
# periodo y fijar una convención de mano de cámara que se invierte en el
# descanso; para "¿qué rol es más difícil de imputar?" el lado es ruido y la
# distinción central/abierto es justo la que importa. Siete roles:
#   GK  portero
#   CB  defensa central          FB  defensa abierto (lateral/carrilero)
#   CM  medio central            WM  medio abierto
#   CF  atacante central (9/10)  WF  atacante abierto (extremo)
ROLE_COLORS = {
    "GK": "#e63946",   # rojo
    "CB": "#1d3f72",   # azul oscuro  (defensa central)
    "FB": "#4895ef",   # azul claro   (defensa abierto)
    "CM": "#c9a227",   # ámbar oscuro (medio central)
    "WM": "#8ac926",   # verde-lima   (medio abierto)
    "CF": "#f3722c",   # naranja      (atacante central)
    "WF": "#2a9d8f",   # verde        (atacante abierto)
    "?":  "#adb5bd",   # sin clasificar (pocos frames visibles)
}
ROLE_NAMES = [r for r in ROLE_COLORS if r != "?"]

# Nombre legible por rol, para etiquetas de figura.
ROLE_LABEL = {
    "GK": "Portero",
    "CB": "Defensa central", "FB": "Defensa abierto",
    "CM": "Medio central", "WM": "Medio abierto",
    "CF": "Atacante central", "WF": "Atacante abierto",
    "?":  "Sin clasificar",
}

# Grupos de línea, para agregados de cuatro categorías.
ROLE_GROUP = {
    "GK": "Portero",
    "CB": "Defensa", "FB": "Defensa",
    "CM": "Mediocampo", "WM": "Mediocampo",
    "CF": "Ataque", "WF": "Ataque",
    "?": "Sin clasificar",
}

# Orden canónico de presentación (retrasado -> adelantado).
ROLE_ORDER = ["GK", "CB", "FB", "CM", "WM", "CF", "WF"]

# Umbral de carril como fracción del ancho A: distancia al eje y=A/2 por encima
# de la cual un jugador se considera "abierto". Calibrado sobre Bundesliga.
_LANE_DEF = 0.22
_LANE_MID = 0.25
_LANE_ATT = 0.20

_MIN_VISIBLE_FRAMES = 2


def infer_roles_by_geometry(
    match: Match, view: ViewportResult, period: int
) -> dict[int, str]:
    """
    Infiere el rol de cada jugador en un periodo por medianas posicionales.

    Parameters
    ----------
    match : Match
    view : ViewportResult
        Se usa `view.visible` para restringir a los frames observados por la
        cámara. Ese es el punto: el rol sale solo de lo que se vio.
    period : int
        Periodo (1 o 2). Se procesa por separado porque los equipos cambian de
        lado en el descanso: mezclar periodos promediaría un central a x≈L/2.

    Returns
    -------
    dict : {player_idx -> role_string}
    """
    L, A = match.pitch
    on_period = match.on_pitch & (match.period == period)[:, None]
    seen = view.visible & on_period          # (T, N): visible y en cancha, este periodo

    # Mediana posicional por jugador
    kind = np.full(match.n_players, "field", dtype=object)
    xmed = np.full(match.n_players, np.nan)
    ymed = np.full(match.n_players, np.nan)

    for i in range(match.n_players):
        if match.is_gk[i]:
            kind[i] = "GK"
            continue
        col = seen[:, i]
        if col.sum() < _MIN_VISIBLE_FRAMES:
            kind[i] = "?"
            continue
        px = match.positions[col, i, 0]
        py = match.positions[col, i, 1]
        px = px[np.isfinite(px)]
        py = py[np.isfinite(py)]
        if px.size < _MIN_VISIBLE_FRAMES:
            kind[i] = "?"
            continue
        xmed[i] = float(np.median(px))
        ymed[i] = float(np.median(py))

    roles: dict[int, str] = {}
    for i in range(match.n_players):
        if kind[i] == "GK":
            roles[i] = "GK"
        elif kind[i] == "?":
            roles[i] = "?"

    field = [i for i in range(match.n_players)
             if kind[i] == "field" and np.isfinite(xmed[i])]
    if not field:
        return roles

    xs = xmed[field]

    # Tres líneas (defensa / mediocampo / ataque)
    # Ojo: el equipo que ataca hacia x=L tiene sus defensas en x bajo, y el que
    # ataca hacia x=0 los tiene en x alto. K-Means agrupa por posición absoluta,
    # así que "cluster 0" (x menor) es la defensa del equipo que va hacia L y a
    # la vez el ataque del que va hacia 0. Para nombrar líneas de forma
    # consistente se clasifica por EQUIPO, orientando cada uno hacia su ataque.
    cluster_labels = _three_lines_per_team(match, field, xmed, L)

    # Central vs abierto dentro de cada línea
    for i in field:
        line = cluster_labels[i]           # "DEF" | "MID" | "ATT"
        ydist = abs(ymed[i] - A / 2)       # distancia al eje longitudinal
        if line == "DEF":
            role = "FB" if ydist > A * _LANE_DEF else "CB"
        elif line == "MID":
            role = "WM" if ydist > A * _LANE_MID else "CM"
        else:  # ATT
            role = "WF" if ydist > A * _LANE_ATT else "CF"
        roles[i] = role

    return roles


def _three_lines_per_team(match, field, xmed, L) -> dict[int, str]:
    """
    Asigna cada jugador de campo a DEF/MID/ATT, orientando por equipo.

    Cada equipo se procesa por separado y se orienta hacia su portería de
    ataque, de modo que "DEF" sea siempre la línea más retrasada de ESE equipo,
    sin importar hacia qué lado juega en el periodo.
    """
    try:
        from sklearn.cluster import KMeans
        have_km = True
    except ImportError:
        have_km = False

    out: dict[int, str] = {}
    for team in (0, 1):
        idx = [i for i in field if match.team_idx[i] == team]
        if not idx:
            continue
        xt = np.array([xmed[i] for i in idx], dtype=float)

        # ¿Hacia dónde ataca? La portería que DEFIENDE es la más cercana a la
        # media de su bloque; ataca hacia la contraria. Si defiende x≈0, ataca
        # hacia L y su defensa está en x bajo (orientación natural). Si defiende
        # x≈L, se invierte el eje para que "x bajo" siga siendo su defensa.
        defends_left = xt.mean() < L / 2
        x_oriented = xt if defends_left else (L - xt)

        k = min(3, len(idx))
        if have_km and len(idx) >= 3:
            km = KMeans(n_clusters=3, random_state=42, n_init=10)
            lab = km.fit_predict(x_oriented.reshape(-1, 1))
            order = np.argsort(km.cluster_centers_.ravel())  # menor x = DEF
        elif len(idx) >= 3:
            q = np.quantile(x_oriented, [1 / 3, 2 / 3])
            lab = np.searchsorted(q, x_oriented)
            order = np.array([0, 1, 2])
        else:
            # Muy pocos jugadores de campo (partido degenerado / prueba): por
            # tercios directos del eje orientado.
            lab = np.searchsorted(
                np.quantile(x_oriented, [1 / 3, 2 / 3]) if len(idx) > 1 else [0, 0],
                x_oriented,
            )
            order = np.array([0, 1, 2])

        rank = {int(c): r for r, c in enumerate(order)}
        names = {0: "DEF", 1: "MID", 2: "ATT"}
        for j, i in enumerate(idx):
            out[i] = names[min(rank.get(int(lab[j]), int(lab[j])), 2)]
    return out


def infer_all_roles(
    match: Match, view: ViewportResult
) -> dict[int, dict[int, str]]:
    """
    Roles para todos los periodos del partido.

    Returns
    -------
    dict : {period: {player_idx: role_string}}
    """
    return {
        int(p): infer_roles_by_geometry(match, view, int(p))
        for p in np.unique(match.period)
    }


def role_at(roles_by_period: dict[int, dict[int, str]],
            period: int, player_idx: int) -> str:
    """Consulta segura: rol de un jugador en un periodo, '?' si no está."""
    return roles_by_period.get(int(period), {}).get(int(player_idx), "?")
