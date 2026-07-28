"""
Cargadores: proveedor -> esquema canónico.

Toda la dependencia de kloppy vive aquí. El resto del proyecto no importa
kloppy en ninguna parte, de modo que añadir un proveedor nuevo (o los datos
propios del club) es escribir una función en este archivo y nada más.

Proveedores soportados
----------------------
- sportec  : 7 partidos abiertos de Bundesliga 1 y 2 (IDSSE / Bassek et al.
             2025). Tracking TRACAB gen-5 a 25 fps, los 22 jugadores + balón.
             Es el conjunto de ENTRENAMIENTO y VALIDACIÓN principal.
- metrica  : 3 partidos abiertos. Es el conjunto que usa Choi (2026), por lo
             que sirve como TEST EXTERNO directamente comparable con el paper.
- synthetic: partido simulado, sin red. Sirve para probar el pipeline completo
             en segundos y para los tests unitarios.

SkillCorner NO se carga aquí como ground truth: es tracking de broadcast, ya
tiene huecos y no contiene las posiciones de los jugadores ausentes. Su rol es
calibrar el simulador de cámara (ver scripts/03_calibrate_camera.py).
"""

from __future__ import annotations

import warnings

import numpy as np

from .schema import Match, PITCH_LENGTH, PITCH_WIDTH

# Los 7 partidos abiertos de Sportec/DFL (IDSSE).
SPORTEC_OPEN_MATCHES: dict[str, str] = {
    "J03WMX": "1. FC Köln vs FC Bayern München",
    "J03WN1": "VfL Bochum 1848 vs Bayer 04 Leverkusen",
    "J03WPY": "Fortuna Düsseldorf vs 1. FC Nürnberg",
    "J03WOH": "Fortuna Düsseldorf vs SSV Jahn Regensburg",
    "J03WQQ": "Fortuna Düsseldorf vs FC St. Pauli",
    "J03WOY": "Fortuna Düsseldorf vs F.C. Hansa Rostock",
    "J03WR9": "Fortuna Düsseldorf vs 1. FC Kaiserslautern",
}

METRICA_OPEN_MATCHES = ["1", "2"]  # el juego 3 usa formato EPTS distinto




def _pitch_and_scalers(dataset):
    """
    Deriva del sistema de coordenadas del dataset las funciones que llevan
    cualquier proveedor al esquema canónico.

    ADVERTENCIA: cada proveedor usa un sistema distinto y NO son
    intercambiables:

        Sportec : métrico centrado en el origen,  x en [-52.5, +52.5]
        Metrica : normalizado,                    x en [0, 1]
        Tracab  : centímetros centrados,          x en [-5250, +5250]

    Suponer uno solo produce coordenadas catastróficamente equivocadas que aun
    así "parecen" números de cancha. Por eso el rango nunca se asume: se lee de
    los metadatos y se normaliza explícitamente.

    Returns
    -------
    (L, A, to_x, to_y) donde to_x/to_y mapean coordenada nativa -> metros con
    origen en la esquina inferior izquierda.
    """
    cs = getattr(dataset.metadata, "coordinate_system", None)
    pdim = getattr(cs, "pitch_dimensions", None) or dataset.metadata.pitch_dimensions

    x0, x1 = float(pdim.x_dim.min), float(pdim.x_dim.max)
    y0, y1 = float(pdim.y_dim.min), float(pdim.y_dim.max)

    L = float(getattr(pdim, "pitch_length", None) or PITCH_LENGTH)
    A = float(getattr(pdim, "pitch_width", None) or PITCH_WIDTH)
    if not (90 <= L <= 120):
        L = PITCH_LENGTH
    if not (55 <= A <= 80):
        A = PITCH_WIDTH

    span_x, span_y = (x1 - x0), (y1 - y0)
    if abs(span_x) < 1e-9 or abs(span_y) < 1e-9:
        raise ValueError(
            f"Sistema de coordenadas degenerado: x={x0}..{x1}, y={y0}..{y1}"
        )

    return L, A, (lambda v: (v - x0) / span_x * L), (lambda v: (v - y0) / span_y * A)


def _ensure_static_orientation(dataset):
    """
    Garantiza una orientación fija a lo largo del partido.

    Si la orientación depende del equipo en posesión (BALL_OWNING_TEAM,
    ACTION_EXECUTING_TEAM), las coordenadas se voltean cada vez que cambia la
    posesión. Eso destruiría el modelo de cámara: el balón "saltaría" de un
    lado al otro varias veces por minuto y el EMA de paneo perseguiría un
    fantasma. La cámara sigue al balón en coordenadas absolutas de estadio.
    """
    from kloppy.domain import Orientation

    orient = getattr(dataset.metadata, "orientation", None)
    static = {
        Orientation.STATIC_HOME_AWAY, Orientation.STATIC_AWAY_HOME,
        Orientation.HOME_AWAY, Orientation.AWAY_HOME,
    }
    if orient in static:
        return dataset
    warnings.warn(
        f"Orientación '{orient}' depende de la posesión; se transforma a "
        "STATIC_HOME_AWAY para que el modelo de cámara sea válido.",
        stacklevel=2,
    )
    return dataset.transform(to_orientation=Orientation.STATIC_HOME_AWAY)


def _is_goalkeeper(player) -> bool:
    """
    Detección de portero desde los metadatos.

    Solo se consulta `starting_position`. El atributo `position` está
    deprecado en kloppy >=3.17 y emite avisos incluso dentro de
    `catch_warnings`, así que no se toca: si los metadatos no bastan, el
    fallback geométrico de `_infer_goalkeepers` resuelve el caso.
    """
    pos = getattr(player, "starting_position", None)
    if pos is None:
        return False
    name = str(getattr(pos, "name", pos)).lower()
    return "goalkeeper" in name or name in {"gk", "tw"}


def _infer_goalkeepers(positions, team_idx, is_gk, pitch, period, min_frames=500):
    """
    Fallback: si los metadatos no marcan portero, se infiere por geometría.

    POR PERIODO, y esto no es un detalle
    ------------------------------------
    Los equipos cambian de lado en el descanso. Un portero defiende x≈5 en el
    primer tiempo y x≈100 en el segundo, así que su posición media sobre el
    PARTIDO COMPLETO es ~52.5 m: el centro de la cancha, el valor menos
    parecido a un portero que existe. Promediar sobre todo el partido descarta
    exactamente al jugador que se busca y selecciona a otro cualquiera —
    típicamente un suplente, cuya media está sesgada por el único tiempo que
    jugó.

    El síntoma es silencioso y venenoso: `validate()` pasa (hay dos "porteros"),
    las métricas corren, y los porteros reales quedan evaluados como jugadores
    de campo con ancla de centroide. Si además el suplente elegido no jugó el
    tramo analizado, la bandera `--include-gk` no cambia ni una muestra y
    parece robustez.

    Criterio correcto: por cada periodo se toma la MEDIANA de la distancia a la
    línea de meta más cercana (mediana y no media: es robusta a las salidas
    puntuales del portero y a los córners a favor). El portero es quien
    minimiza esa distancia en algún periodo.
    """
    if is_gk.any():
        return is_gk

    warnings.warn(
        "Los metadatos no identifican porteros; se infieren por geometría "
        "(mediana por periodo de la distancia a meta).",
        stacklevel=2,
    )
    L = pitch[0]
    N = positions.shape[1]
    out = np.zeros_like(is_gk)

    # Distancia a la línea de meta más cercana, por frame y jugador.
    x = positions[:, :, 0]
    dist_goal = np.minimum(x, L - x)

    # Mejor (menor) mediana por jugador, tomada sobre los periodos por separado.
    best = np.full(N, np.inf)
    for pid in np.unique(period):
        sel = period == pid
        d = dist_goal[sel]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(d, axis=0)
        # Solo cuentan jugadores con presencia real en ese periodo.
        n_obs = np.isfinite(d).sum(axis=0)
        med = np.where(n_obs >= min_frames, med, np.inf)
        best = np.minimum(best, np.nan_to_num(med, nan=np.inf))

    for t in (0, 1):
        idx = np.where(team_idx == t)[0]
        if idx.size == 0:
            continue
        k = idx[int(np.argmin(best[idx]))]
        out[k] = True
        if best[k] > 20.0:
            warnings.warn(
                f"El portero inferido del equipo {t} tiene mediana de "
                f"{best[k]:.1f} m a la meta, demasiado lejos para ser portero. "
                "Revisa los datos: probablemente la inferencia falló.",
                stacklevel=2,
            )
    return out


def _from_kloppy(dataset, match_id: str, provider: str) -> Match:
    """Convierte un TrackingDataset de kloppy al esquema canónico."""
    dataset = _ensure_static_orientation(dataset)

    frames = dataset.frames
    if not frames:
        raise ValueError(f"El dataset de {match_id} no tiene frames")

    L, A, to_x, to_y = _pitch_and_scalers(dataset)

    # Inventario: todo el roster, no solo los titulares (hay sustituciones).
    teams = dataset.metadata.teams
    players = [(p, ti) for ti, team in enumerate(teams) for p in team.players]
    player_index = {p.player_id: k for k, (p, _) in enumerate(players)}

    T, N = len(frames), len(players)
    positions = np.full((T, N, 2), np.nan, dtype=np.float32)
    ball = np.full((T, 2), np.nan, dtype=np.float32)
    on_pitch = np.zeros((T, N), dtype=bool)
    period = np.ones(T, dtype=np.int8)
    ball_alive = np.zeros(T, dtype=bool)

    for t, fr in enumerate(frames):
        bc = fr.ball_coordinates
        if bc is not None and bc.x is not None and bc.y is not None:
            ball[t] = (to_x(bc.x), to_y(bc.y))
        if fr.period is not None:
            period[t] = fr.period.id
        bs = getattr(fr, "ball_state", None)
        ball_alive[t] = bs is not None and "alive" in str(bs).lower()

        for player, pdata in fr.players_data.items():
            k = player_index.get(player.player_id)
            if k is None:
                continue
            c = pdata.coordinates
            if c is None or c.x is None or c.y is None:
                continue
            positions[t, k] = (to_x(c.x), to_y(c.y))
            on_pitch[t, k] = True

    team_idx = np.array([t for _, t in players], dtype=np.int8)
    is_gk = np.array([_is_goalkeeper(p) for p, _ in players], dtype=bool)
    is_gk = _infer_goalkeepers(positions, team_idx, is_gk, (L, A), period)

    if not ball_alive.any():
        warnings.warn(
            "El proveedor no expone ball_state; se asume balón vivo en todos "
            "los frames. Las métricas incluirán balón parado.",
            stacklevel=2,
        )
        ball_alive[:] = True

    m = Match(
        match_id=match_id,
        positions=positions,
        ball=ball,
        on_pitch=on_pitch,
        team_idx=team_idx,
        is_gk=is_gk,
        player_ids=[p.player_id for p, _ in players],
        period=period,
        ball_alive=ball_alive,
        fps=float(getattr(dataset.metadata, "frame_rate", 0) or 25.0),
        pitch=(L, A),
        provider=provider,
        meta={
            "teams": [t.name for t in teams],
            "coordinate_system": type(
                getattr(dataset.metadata, "coordinate_system", None)
            ).__name__,
        },
    )
    m.validate()
    return m




def load_sportec(match_id: str) -> Match:
    """Carga uno de los 7 partidos abiertos de Sportec/DFL vía kloppy."""
    from kloppy import sportec

    if match_id not in SPORTEC_OPEN_MATCHES:
        raise ValueError(
            f"'{match_id}' no está en el conjunto abierto. "
            f"Disponibles: {list(SPORTEC_OPEN_MATCHES)}"
        )
    ds = sportec.load_open_tracking_data(match_id=match_id, only_alive=False)
    return _from_kloppy(ds, match_id=match_id, provider="sportec")


def load_metrica(match_id: str) -> Match:
    """Carga un partido abierto de Metrica Sports vía kloppy."""
    from kloppy import metrica

    ds = metrica.load_open_data(match_id=str(match_id))
    return _from_kloppy(ds, match_id=f"metrica_{match_id}", provider="metrica")


def load(provider: str, match_id: str) -> Match:
    """Despachador genérico."""
    fn = {
        "sportec": load_sportec,
        "metrica": load_metrica,
        # La semilla deriva del match_id para que partidos distintos
        # sean efectivamente distintos y aun así reproducibles.
        "synthetic": lambda mid: synthetic_match(
            match_id=mid, seed=_stable_seed(mid)
        ),
    }.get(provider)
    if fn is None:
        raise ValueError(f"Proveedor desconocido: {provider}")
    return fn(match_id)


def _stable_seed(text: str) -> int:
    """
    Semilla determinista a partir de una cadena.

    No se usa hash(): está aleatorizado por proceso salvo que se fije
    PYTHONHASHSEED, así que los "datos sintéticos reproducibles" no lo serían
    entre ejecuciones distintas.
    """
    import hashlib

    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)




def synthetic_match(
    match_id: str = "SYNTH01",
    minutes: float = 10.0,
    fps: float = 25.0,
    seed: int = 42,
) -> Match:
    """
    Genera un partido plausible sin necesidad de red.

    NO es un sustituto de datos reales y sus números no deben reportarse como
    resultados. Existe para dos cosas: (a) que el pipeline sea ejecutable en
    segundos mientras se descargan los datos verdaderos, y (b) tests unitarios
    deterministas.

    Modelo generativo
    -----------------
    El balón sigue un paseo aleatorio suavizado, con sesgo hacia el centro para
    que no se pegue a las bandas. Cada jugador tiene una posición de rol fija
    (formación 4-4-2 espejada) y se mueve hacia un objetivo que es una
    combinación convexa de su rol y de la posición del balón, con un peso de
    atracción distinto por línea (los delanteros persiguen más el balón que
    los centrales). Los porteros se quedan cerca de su portería.

    Esto reproduce las dos propiedades que importan para probar el simulador
    de cámara: los equipos se desplazan en bloque siguiendo al balón, y los
    jugadores mantienen un desplazamiento aproximadamente constante respecto
    al centroide de su equipo (que es justo la estructura que explota el
    baseline B4).
    """
    rng = np.random.default_rng(seed)
    L, A = PITCH_LENGTH, PITCH_WIDTH
    T = int(minutes * 60 * fps)
    dt = 1.0 / fps

    # Paseo aleatorio suavizado con retorno a la media
    ball = np.zeros((T, 2), dtype=np.float64)
    pos = np.array([L / 2, A / 2])
    vel = np.zeros(2)
    for t in range(T):
        pull = (np.array([L / 2, A / 2]) - pos) * 0.0008
        vel = 0.985 * vel + pull + rng.normal(0, 0.09, 2)
        speed = np.linalg.norm(vel)
        if speed > 0.55:  # tope ~13.7 m/s a 25 fps
            vel *= 0.55 / speed
        pos = np.clip(pos + vel, [1.0, 1.0], [L - 1.0, A - 1.0])
        ball[t] = pos

    # Formación 4-4-2 en fracciones de cancha
    formation = [
        (0.05, 0.50),  # portero
        (0.20, 0.15), (0.20, 0.38), (0.20, 0.62), (0.20, 0.85),  # defensa
        (0.42, 0.15), (0.42, 0.38), (0.42, 0.62), (0.42, 0.85),  # medio
        (0.62, 0.35), (0.62, 0.65),                              # ataque
    ]
    # Cuánto persigue cada línea al balón
    chase = np.array([0.02, 0.16, 0.16, 0.16, 0.16, 0.30, 0.30, 0.30, 0.30, 0.40, 0.40])

    N = 22
    home_roles = np.array([(fx * L, fy * A) for fx, fy in formation])
    away_roles = np.array([((1 - fx) * L, (1 - fy) * A) for fx, fy in formation])
    roles = np.vstack([home_roles, away_roles])
    chase_all = np.concatenate([chase, chase])

    period = np.where(np.arange(T) < T // 2, 1, 2).astype(np.int8)

    # Los equipos cambian de lado en el descanso. No es cosmético: sin este
    # cambio, toda la lógica que depende de qué portería defiende cada equipo
    # (inferencia de portero, anclaje del portero en B4, reinicio de offsets)
    # quedaría sin probar, y un bug ahí es silencioso.
    roles_p2 = roles.copy()
    roles_p2[:, 0] = L - roles_p2[:, 0]
    roles_p2[:, 1] = A - roles_p2[:, 1]

    positions = np.zeros((T, N, 2), dtype=np.float64)
    cur = roles + rng.normal(0, 1.5, roles.shape)
    pvel = np.zeros((N, 2))

    for t in range(T):
        # Objetivo: rol (del lado que toque en este periodo) + atracción al balón
        R = roles if period[t] == 1 else roles_p2
        target = R + chase_all[:, None] * (ball[t] - R)
        accel = (target - cur) * 1.4 * dt + rng.normal(0, 0.05, (N, 2))
        pvel = 0.90 * pvel + accel
        sp = np.linalg.norm(pvel, axis=1, keepdims=True)
        too_fast = (sp > 0.36).ravel()  # ~9 m/s a 25 fps
        pvel[too_fast] *= (0.36 / sp[too_fast])
        cur = np.clip(cur + pvel, [0.5, 0.5], [L - 0.5, A - 0.5])
        positions[t] = cur

    # Porteros anclados a su portería, que cambia en el descanso.
    gk_idx = np.array([0, 11])
    is_gk = np.zeros(N, dtype=bool)
    is_gk[gk_idx] = True
    wobble = 3.0 * np.sin(np.linspace(0, 12 * np.pi, T))
    for k, frac in zip(gk_idx, (0.06, 0.94)):
        gx = np.where(period == 1, frac * L, (1.0 - frac) * L)
        positions[:, k, 0] = gx + wobble
        positions[:, k, 1] = A / 2 + (ball[:, 1] - A / 2) * 0.18

    # ~85% de balón en juego, en rachas
    alive = np.ones(T, dtype=bool)
    n_stops = int(minutes * 1.2)
    for _ in range(n_stops):
        s = rng.integers(0, max(1, T - 1))
        alive[s : s + int(rng.integers(2, 12) * fps)] = False

    m = Match(
        match_id=match_id,
        positions=positions.astype(np.float32),
        ball=ball.astype(np.float32),
        on_pitch=np.ones((T, N), dtype=bool),
        team_idx=np.array([0] * 11 + [1] * 11, dtype=np.int8),
        is_gk=is_gk,
        player_ids=[f"H{i:02d}" for i in range(11)] + [f"A{i:02d}" for i in range(11)],
        period=period,
        ball_alive=alive,
        fps=fps,
        pitch=(L, A),
        provider="synthetic",
        meta={"seed": seed, "note": "DATOS SIMULADOS - no reportar como resultado"},
    )
    m.validate()
    return m
