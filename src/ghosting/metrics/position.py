"""
Métricas de error de posición, estratificadas y con incertidumbre.

Dos decisiones de protocolo que NO son opcionales:

1. ESTRATIFICACIÓN POR GAP DE OCLUSIÓN.
   Un modelo de velocidad constante gana a cualquier red neuronal cuando el
   jugador lleva 2 segundos oculto. El promedio global mezcla ese régimen
   trivial con el difícil y oculta dónde está realmente el mérito. Los bins
   son los de Choi (2026): <=2 s, 2-9.6 s, >9.6 s. El corte en 9.6 s no es
   arbitrario: es la longitud de ventana del Graph Imputer de DeepMind, más
   allá de la cual ese modelo no está definido, y donde caen 50-57% de las
   muestras ocultas reales.

2. BLOCK BOOTSTRAP.
   A 25 fps, frames consecutivos están fuertemente autocorrelacionados: la
   posición en t y en t+1 difieren en centímetros. Remuestrear frames
   individuales trataría 25 observaciones casi idénticas como 25 muestras
   independientes e inflaría artificialmente la precisión, produciendo
   intervalos de confianza demasiado estrechos (anticonservadores). Se
   remuestrean bloques contiguos de 1 minuto.

Se reporta MEDIANA y no media. La distribución del error tiene cola derecha
pesada (un jugador oculto 3 minutos puede estar a 40 m), y la media sería
dominada por esa cola. Choi también reporta mediana; usar media rompería la
comparabilidad.
"""

from __future__ import annotations

import numpy as np

from ..io.schema import Match
from ..camera.viewport import ViewportResult, LONG_OCCLUSION_S


BIN_EDGES = [0.0, 2.0, LONG_OCCLUSION_S, np.inf]
BIN_LABELS = ["<=2s", "2-9.6s", ">9.6s"]


def position_errors(
    match: Match,
    view: ViewportResult,
    estimate: np.ndarray,
    alive_only: bool = True,
    include_gk: bool = False,
) -> dict[str, np.ndarray]:
    """
    Extrae los errores de posición de los jugadores ocultos.

    Población puntuable: pares (frame, jugador) tales que el jugador está en
    cancha, NO es visible, ya fue observado al menos una vez en el periodo
    (gap finito), y el método produjo una estimación.

    Los cold start (nunca vistos) se excluyen porque no hay observación previa
    de la cual partir; ningún método causal puede posicionarlos y puntuarlos
    penalizaría por igual a todos sin discriminar.

    Parameters
    ----------
    include_gk : bool
        Si False (por defecto) se excluyen porteros. Reportar ambas variantes
        es obligatorio: la dinámica del portero es cualitativamente distinta y
        mezclarla contamina la comparación.

    Returns
    -------
    dict con:
        err   : (K,) error euclidiano en metros
        gap   : (K,) gap de oclusión en segundos
        frame : (K,) índice de frame, necesario para el block bootstrap
    """
    T, N = match.n_frames, match.n_players

    scorable = (
        match.on_pitch
        & ~view.visible
        & np.isfinite(view.gap_s)
        & np.isfinite(estimate[..., 0])
        & np.isfinite(match.positions[..., 0])
    )
    if alive_only:
        scorable &= match.ball_alive[:, None]
    if not include_gk:
        scorable &= ~match.is_gk[None, :]

    fi, pi = np.where(scorable)
    if fi.size == 0:
        return {
            "err": np.array([], dtype=np.float64),
            "gap": np.array([], dtype=np.float64),
            "frame": np.array([], dtype=np.int64),
        }

    diff = estimate[fi, pi] - match.positions[fi, pi]
    return {
        "err": np.linalg.norm(diff, axis=1).astype(np.float64),
        "gap": view.gap_s[fi, pi].astype(np.float64),
        "frame": fi,
    }


def stratified_median(errors: dict[str, np.ndarray]) -> dict[str, float]:
    """
    Mediana del error, global y por bin de oclusión, más el peso de cada bin.

    Las claves `share_*` son tan importantes como las de error: dicen qué
    fracción del problema real vive en cada régimen. Un modelo excelente en
    <=2 s que cubre el 19% de los casos vale menos que uno decente en >9.6 s
    que cubre el 41%.
    """
    err, gap = errors["err"], errors["gap"]
    if err.size == 0:
        return {"n": 0}

    out: dict[str, float] = {
        "n": int(err.size),
        "median_all": float(np.median(err)),
        "mean_all": float(np.mean(err)),
        "p90_all": float(np.percentile(err, 90)),
    }
    for lo, hi, label in zip(BIN_EDGES[:-1], BIN_EDGES[1:], BIN_LABELS):
        sel = (gap > lo) & (gap <= hi) if lo > 0 else (gap >= lo) & (gap <= hi)
        out[f"share_{label}"] = float(sel.mean())
        out[f"median_{label}"] = float(np.median(err[sel])) if sel.any() else np.nan
    return out


def block_bootstrap_ci(
    errors: dict[str, np.ndarray],
    fps: float,
    statistic=np.median,
    block_seconds: float = 60.0,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Intervalo de confianza por bootstrap de bloques móviles.

    Se agrupan las observaciones en bloques contiguos de `block_seconds`, se
    remuestrean bloques COMPLETOS con reemplazo, y se recalcula el estadístico.
    Esto respeta la autocorrelación temporal: dentro de un bloque las
    observaciones siguen juntas, como en los datos originales.

    Returns
    -------
    (estimado_puntual, límite_inferior, límite_superior)
    """
    err, frame = errors["err"], errors["frame"]
    if err.size == 0:
        return (np.nan, np.nan, np.nan)

    point = float(statistic(err))

    block_len = max(1, int(round(block_seconds * fps)))
    block_id = frame // block_len
    uniq = np.unique(block_id)
    if uniq.size < 2:
        return (point, np.nan, np.nan)

    by_block = [err[block_id == b] for b in uniq]
    rng = np.random.default_rng(seed)
    n_blocks = len(by_block)

    stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        pick = rng.integers(0, n_blocks, size=n_blocks)
        sample = np.concatenate([by_block[j] for j in pick])
        stats[i] = statistic(sample)

    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (point, lo, hi)


def stratified_bootstrap_ci(
    errors: dict[str, np.ndarray],
    fps: float,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """
    Intervalos de confianza por bin de oclusión, no solo global.

    POR QUÉ ES OBLIGATORIO
    ----------------------
    Sin IC por bin, comparar un valor puntual contra un rango publicado invita
    al error de leer ese rango como una frontera dura. No lo es: el rango
    15.6-16.9 m de Choi (2026) son tres estimaciones puntuales de tres
    partidos, no un intervalo de confianza.

    Un bin de oclusión larga contiene menos observaciones independientes de las
    que sugiere su n: son pocas rachas largas, cada una con cientos de frames
    casi idénticos. La incertidumbre real es grande, y declarar "fuera de
    rango" una diferencia de 1 m sin medirla es exactamente el error que el
    block bootstrap existe para evitar.

    El remuestreo se hace sobre bloques de un minuto de la muestra COMPLETA y
    luego se estratifica dentro de cada réplica. Así se preserva la
    autocorrelación y también la variabilidad de cuántas observaciones caen en
    cada bin, que es una fuente de incertidumbre real.
    """
    err, gap, frame = errors["err"], errors["gap"], errors["frame"]
    out: dict[str, tuple[float, float]] = {}
    if err.size == 0:
        return out

    block_len = max(1, int(round(60.0 * fps)))
    block_id = frame // block_len
    uniq = np.unique(block_id)
    if uniq.size < 2:
        return out

    idx_by_block = [np.where(block_id == b)[0] for b in uniq]
    rng = np.random.default_rng(seed)
    n_blocks = len(idx_by_block)

    stats = {label: np.full(n_boot, np.nan) for label in BIN_LABELS}
    for i in range(n_boot):
        pick = rng.integers(0, n_blocks, size=n_blocks)
        sel = np.concatenate([idx_by_block[j] for j in pick])
        e, g = err[sel], gap[sel]
        for lo, hi, label in zip(BIN_EDGES[:-1], BIN_EDGES[1:], BIN_LABELS):
            m = (g > lo) & (g <= hi) if lo > 0 else (g >= lo) & (g <= hi)
            if m.any():
                stats[label][i] = np.median(e[m])

    for label in BIN_LABELS:
        v = stats[label][np.isfinite(stats[label])]
        if v.size:
            out[label] = (
                float(np.percentile(v, 100 * alpha / 2)),
                float(np.percentile(v, 100 * (1 - alpha / 2))),
            )
    return out


def evaluate(
    match: Match,
    view: ViewportResult,
    estimate: np.ndarray,
    method_name: str,
    n_boot: int = 500,
    include_gk: bool = False,
    bin_ci: bool = True,
    alive_only: bool = True,
) -> dict:
    """
    Evaluación completa de un método sobre un partido. Devuelve una fila.

    Parameters
    ----------
    bin_ci : bool
        Calcular también IC por bin de oclusión. Cuesta n_boot pasadas extra
        pero es lo que permite comparar honestamente contra rangos publicados.
    """
    errs = position_errors(match, view, estimate,
                           include_gk=include_gk, alive_only=alive_only)
    row = {
        "match_id": match.match_id,
        "method": method_name,
        "width_m": float(view.config.width_m),
        "include_gk": include_gk,
        "alive_only": alive_only,
    }
    row.update(stratified_median(errs))
    if errs["err"].size:
        pt, lo, hi = block_bootstrap_ci(errs, match.fps, n_boot=n_boot)
        row["median_ci_lo"] = lo
        row["median_ci_hi"] = hi
        if bin_ci:
            for label, (blo, bhi) in stratified_bootstrap_ci(
                errs, match.fps, n_boot=n_boot
            ).items():
                row[f"ci_lo_{label}"] = blo
                row[f"ci_hi_{label}"] = bhi
    return row




def paired_position_errors(
    match: Match,
    view: ViewportResult,
    estimate_a: np.ndarray,
    estimate_b: np.ndarray,
    alive_only: bool = True,
    include_gk: bool = False,
) -> dict[str, np.ndarray]:
    """
    Errores de dos estimadores sobre EXACTAMENTE la misma población.

    Un par (frame, jugador) entra solo si ambos estimadores produjeron una
    estimación finita. Así la diferencia mide la calidad relativa y no un
    cambio en qué se está evaluando.

    Returns
    -------
    dict con err_a, err_b, gap, frame — todos alineados índice a índice.
    """
    scorable = (
        match.on_pitch
        & ~view.visible
        & np.isfinite(view.gap_s)
        & np.isfinite(estimate_a[..., 0])
        & np.isfinite(estimate_b[..., 0])
        & np.isfinite(match.positions[..., 0])
    )
    if alive_only:
        scorable &= match.ball_alive[:, None]
    if not include_gk:
        scorable &= ~match.is_gk[None, :]

    fi, pi = np.where(scorable)
    if fi.size == 0:
        z = np.array([], dtype=np.float64)
        return {"err_a": z, "err_b": z, "gap": z, "frame": np.array([], dtype=np.int64)}

    truth = match.positions[fi, pi]
    return {
        "err_a": np.linalg.norm(estimate_a[fi, pi] - truth, axis=1).astype(np.float64),
        "err_b": np.linalg.norm(estimate_b[fi, pi] - truth, axis=1).astype(np.float64),
        "gap": view.gap_s[fi, pi].astype(np.float64),
        "frame": fi,
    }


def paired_block_bootstrap_ci(
    paired: dict[str, np.ndarray],
    fps: float,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    """
    IC sobre la DIFERENCIA de medianas entre dos estimadores.

    POR QUÉ NO BASTAN LOS IC MARGINALES
    -----------------------------------
    Comparar dos intervalos marginales y concluir "se solapan, no hay
    diferencia" es un error clásico y conservador hasta la inutilidad. Cuando
    las dos estimaciones vienen de la MISMA muestra —mismos frames, mismos
    jugadores, misma cámara, cambiando solo el método— la mayor parte de la
    incertidumbre es común a ambas y se cancela al restar.

    Un partido con muchas fases de ataque sostenido dará error alto en los dos
    estimadores a la vez. Esa variabilidad infla los IC marginales pero no
    afecta a la diferencia. El bootstrap pareado la elimina remuestreando
    bloques y calculando la diferencia DENTRO de cada réplica.

    Es el procedimiento que usa Choi (2026) para contrastar B4 contra B2.

    Returns
    -------
    dict etiqueta -> (diferencia_puntual, lo, hi), con la clave "global" más
    una por bin de oclusión. Diferencia = mediana(A) - mediana(B); negativa
    significa que A es mejor. Si el intervalo excluye 0, la diferencia es
    creíble al nivel dado.
    """
    ea, eb = paired["err_a"], paired["err_b"]
    gap, frame = paired["gap"], paired["frame"]
    out: dict[str, tuple[float, float, float]] = {}
    if ea.size == 0:
        return out

    block_len = max(1, int(round(60.0 * fps)))
    block_id = frame // block_len
    uniq = np.unique(block_id)
    if uniq.size < 2:
        return out

    idx_by_block = [np.where(block_id == b)[0] for b in uniq]
    n_blocks = len(idx_by_block)
    rng = np.random.default_rng(seed)

    groups: dict[str, np.ndarray] = {"global": np.ones(ea.size, dtype=bool)}
    for lo, hi, label in zip(BIN_EDGES[:-1], BIN_EDGES[1:], BIN_LABELS):
        groups[label] = (gap > lo) & (gap <= hi) if lo > 0 else (gap >= lo) & (gap <= hi)

    point = {
        k: (float(np.median(ea[m])) - float(np.median(eb[m]))) if m.any() else np.nan
        for k, m in groups.items()
    }

    stats = {k: np.full(n_boot, np.nan) for k in groups}
    for i in range(n_boot):
        pick = rng.integers(0, n_blocks, size=n_blocks)
        sel = np.concatenate([idx_by_block[j] for j in pick])
        a_s, b_s, g_s = ea[sel], eb[sel], gap[sel]
        for k, m_full in groups.items():
            if k == "global":
                m = np.ones(sel.size, dtype=bool)
            else:
                lo, hi = dict(zip(BIN_LABELS, zip(BIN_EDGES[:-1], BIN_EDGES[1:])))[k]
                m = (g_s > lo) & (g_s <= hi) if lo > 0 else (g_s >= lo) & (g_s <= hi)
            if m.any():
                stats[k][i] = np.median(a_s[m]) - np.median(b_s[m])

    for k in groups:
        v = stats[k][np.isfinite(stats[k])]
        if v.size and np.isfinite(point[k]):
            out[k] = (
                point[k],
                float(np.percentile(v, 100 * alpha / 2)),
                float(np.percentile(v, 100 * (1 - alpha / 2))),
            )
    return out


def pool_paired(pieces: list[dict[str, np.ndarray]], fps: float) -> dict[str, np.ndarray]:
    """
    Agrupa comparaciones pareadas de varios partidos en una sola muestra.

    POR QUÉ HACE FALTA
    ------------------
    Con un solo partido —o medio— la potencia estadística por bin es baja: el
    régimen de oclusión larga contiene pocas rachas independientes, así que su
    intervalo sale ancho aunque el efecto sea real. El síntoma típico es
    exactamente lo que se observa al comparar dos partidos: el signo del efecto
    coincide en todos los bins, pero cada partido declara "creíble" bins
    distintos. Eso no es contradicción, es falta de muestra.

    Agrupar multiplica el número de bloques independientes y estrecha el
    intervalo sin tocar el estimador puntual.

    EL DETALLE QUE HAY QUE CUIDAR
    -----------------------------
    Los índices de frame se reinician en cada partido, así que concatenar sin
    más fundiría el minuto 3 del partido A con el minuto 3 del partido B en un
    mismo bloque de remuestreo. Serían observaciones de partidos distintos
    tratadas como contiguas. Aquí se desplaza cada partido por un offset mayor
    que su duración, de modo que los bloques nunca crucen la frontera.

    Parameters
    ----------
    pieces : lista de salidas de `paired_position_errors`, una por partido.
    fps : frecuencia común de muestreo.

    Returns
    -------
    dict con el mismo esquema, listo para `paired_block_bootstrap_ci`.
    """
    if not pieces:
        return {k: np.array([]) for k in ("err_a", "err_b", "gap", "frame")}

    block_len = max(1, int(round(60.0 * fps)))
    errs_a, errs_b, gaps, frames = [], [], [], []
    offset = 0
    for p in pieces:
        if p["err_a"].size == 0:
            continue
        errs_a.append(p["err_a"])
        errs_b.append(p["err_b"])
        gaps.append(p["gap"])
        frames.append(p["frame"] + offset)
        # Offset alineado a bloque y con un bloque de separación: garantiza que
        # ningún bloque contenga frames de dos partidos distintos.
        span = int(p["frame"].max()) + 1
        offset += ((span // block_len) + 2) * block_len

    if not errs_a:
        return {k: np.array([]) for k in ("err_a", "err_b", "gap", "frame")}

    return {
        "err_a": np.concatenate(errs_a),
        "err_b": np.concatenate(errs_b),
        "gap": np.concatenate(gaps),
        "frame": np.concatenate(frames),
    }
