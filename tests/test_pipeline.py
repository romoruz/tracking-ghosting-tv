"""
Tests de las invariantes que, si se rompen, invalidan todos los resultados.

Cada test corresponde a un error que costaría semanas detectar a mano y que
produciría números plausibles pero falsos. Córrelos antes de reportar
cualquier cifra.
"""

import numpy as np
import pytest

from ghosting.io import synthetic_match, Match
from ghosting.camera import simulate, occlusion_stats, ViewportConfig
from ghosting.camera.viewport import _alpha_for_fps, compute_gap_seconds
from ghosting.baselines import run_ladder, ALL_METHODS
from ghosting.metrics import position_errors, stratified_median, block_bootstrap_ci


@pytest.fixture(scope="module")
def match():
    return synthetic_match(minutes=6.0, fps=25.0, seed=7)


@pytest.fixture(scope="module")
def match5(match):
    return match.resample(5.0)


@pytest.fixture(scope="module")
def view(match5):
    return simulate(match5, ViewportConfig(width_m=44.0))


# Esquema


def test_schema_valida(match):
    match.validate()


def test_roundtrip_disco(match, tmp_path):
    p = match.save(tmp_path / "m.npz")
    back = Match.load(p)
    assert np.allclose(back.positions, match.positions, equal_nan=True)
    assert back.fps == match.fps and back.match_id == match.match_id


def test_resample_es_decimacion(match):
    m5 = match.resample(5.0)
    assert m5.fps == 5.0
    # Decimar, no interpolar: los frames resultantes son frames originales.
    assert np.allclose(m5.positions[0], match.positions[0], equal_nan=True)
    assert np.allclose(m5.positions[1], match.positions[5], equal_nan=True)


def test_resample_rechaza_subir_fps(match):
    with pytest.raises(ValueError):
        match.resample(50.0)


# Cámara


def test_alpha_preserva_constante_de_tiempo():
    """
    Un EMA a 5 fps debe tener la misma inercia física que uno a 25 fps.
    Sin esto, evaluar a 5 fps daría una cámara 5x más lenta y las
    estadísticas de oclusión no coincidirían con las publicadas.
    """
    a25, fps = 0.06, 5.0
    a5 = _alpha_for_fps(a25, fps)
    assert (1 - a5) ** fps == pytest.approx((1 - a25) ** 25, rel=1e-9)
    assert a5 > a25  # menos pasos por segundo => cada paso pesa más


def test_visibles_en_rango_publicado(match5):
    """
    Con W=44 m Choi reporta 14.6-15.0 visibles de 22 sobre Metrica.
    Fuera de 12-17 significa que las coordenadas o el simulador están mal.
    """
    s = occlusion_stats(match5, simulate(match5, ViewportConfig(width_m=44.0)))
    assert 12.0 <= s["visible_mean"] <= 17.0, s["visible_mean"]


def test_visibilidad_monotona_en_ancho(match5):
    """Ensanchar el viewport nunca puede reducir el número de visibles."""
    means = [
        occlusion_stats(match5, simulate(match5, ViewportConfig(width_m=w)))["visible_mean"]
        for w in (36, 44, 52, 60)
    ]
    assert all(b >= a - 1e-6 for a, b in zip(means, means[1:])), means


def test_fuera_de_cancha_nunca_visible(match5, view):
    assert not (view.visible & ~match5.on_pitch).any()


def test_politica_portero(match5):
    """`always_visible` debe hacer visible al portero siempre; `occlude`, no."""
    v_occ = simulate(match5, ViewportConfig(gk_policy="occlude"))
    v_vis = simulate(match5, ViewportConfig(gk_policy="always_visible"))
    gk = match5.is_gk
    assert v_vis.visible[:, gk].all()
    assert not v_occ.visible[:, gk].all()
    # La política de portero no debe alterar a los jugadores de campo.
    assert (v_occ.visible[:, ~gk] == v_vis.visible[:, ~gk]).all()


def test_gap_cero_si_visible(match5, view):
    assert np.allclose(view.gap_s[view.visible], 0.0)


def test_gap_reinicia_entre_periodos(match5, view):
    """Tras el descanso los equipos cambian de lado: el historial no aplica."""
    p = match5.period
    if len(np.unique(p)) < 2:
        pytest.skip("el partido tiene un solo periodo")
    t0 = int(np.argmax(p == 2))
    # En el primer frame del segundo periodo nadie puede arrastrar gap previo.
    g = view.gap_s[t0]
    assert np.all(np.isnan(g) | (g == 0.0))


def test_gap_crece_de_a_un_paso():
    """Un jugador continuamente oculto acumula exactamente dt por frame."""
    T, N, fps = 40, 3, 5.0
    vis = np.ones((T, N), dtype=bool)
    vis[10:, 1] = False  # el jugador 1 desaparece en t=10
    gap = compute_gap_seconds(vis, np.ones((T, N), bool), np.ones(T, np.int8), fps)
    assert gap[10, 1] == pytest.approx(1 / fps)
    assert gap[20, 1] == pytest.approx(11 / fps)
    assert gap[39, 1] == pytest.approx(30 / fps)


def test_cold_start_es_nan():
    """Nunca visto antes => no puntuable, no cero."""
    T, N, fps = 20, 2, 5.0
    vis = np.zeros((T, N), dtype=bool)
    vis[15:, 0] = True  # el jugador 0 aparece tarde; el 1 nunca
    gap = compute_gap_seconds(vis, np.ones((T, N), bool), np.ones(T, np.int8), fps)
    assert np.isnan(gap[:15, 0]).all()
    assert np.isnan(gap[:, 1]).all()


# Baselines


@pytest.mark.parametrize("method", ALL_METHODS)
def test_metodos_producen_forma_correcta(match5, view, method):
    est = run_ladder(match5, view, method)
    assert est.shape == (match5.n_frames, match5.n_players, 2)


@pytest.mark.parametrize("method", [m for m in ALL_METHODS if m != "B0"])
def test_visibles_se_copian_exactos(match5, view, method):
    """Un método de imputación no debe alterar lo que la cámara sí ve."""
    est = run_ladder(match5, view, method)
    v = view.visible
    assert np.allclose(est[v], match5.positions[v], equal_nan=True)


@pytest.mark.parametrize("method", [m for m in ALL_METHODS if m != "B0"])
def test_nadie_se_imputa_fuera_de_la_cancha(match5, view, method):
    est = run_ladder(match5, view, method)
    L, A = match5.pitch
    f = np.isfinite(est[..., 0])
    assert est[..., 0][f].min() >= -1e-6 and est[..., 0][f].max() <= L + 1e-6
    assert est[..., 1][f].min() >= -1e-6 and est[..., 1][f].max() <= A + 1e-6


def test_b0_no_imputa(match5, view):
    est = run_ladder(match5, view, "B0")
    hidden = match5.on_pitch & ~view.visible
    assert np.isnan(est[hidden][:, 0]).all()


def test_b4_gana_a_b1_y_b2(match5, view):
    """
    Réplica cualitativa del hallazgo central de Choi (2026): el voto de
    centroide anclado a roles domina a la última-vista y al ancla simple.
    Si esto falla, la implementación de B4 está mal.
    """
    med = {}
    for m in ("B1", "B2", "B4"):
        e = run_ladder(match5, view, m)
        med[m] = stratified_median(position_errors(match5, view, e))["median_all"]
    assert med["B4"] < med["B2"] < med["B1"], med


def test_determinismo(match5, view):
    a = run_ladder(match5, view, "B4")
    b = run_ladder(match5, view, "B4")
    assert np.allclose(a, b, equal_nan=True)


def test_causalidad(match5):
    """
    Ningún método puede usar el futuro. Se verifica truncando el partido:
    las estimaciones sobre el prefijo deben ser idénticas a las del partido
    completo restringidas a ese prefijo. Si un método mirara hacia adelante,
    estos dos cálculos diferirían.
    """
    T = match5.n_frames
    cut = T // 2
    full = run_ladder(match5, simulate(match5, ViewportConfig()), "B4")

    trunc = Match(
        match_id=match5.match_id, positions=match5.positions[:cut],
        ball=match5.ball[:cut], on_pitch=match5.on_pitch[:cut],
        team_idx=match5.team_idx, is_gk=match5.is_gk,
        player_ids=match5.player_ids, period=match5.period[:cut],
        ball_alive=match5.ball_alive[:cut], fps=match5.fps,
        pitch=match5.pitch, provider=match5.provider,
    )
    part = run_ladder(trunc, simulate(trunc, ViewportConfig()), "B4")
    assert np.allclose(full[:cut], part, equal_nan=True, atol=1e-4)


# Métricas


def test_error_cero_con_verdad_perfecta(match5, view):
    """Alimentar la verdad de terreno como estimación debe dar error nulo."""
    errs = position_errors(match5, view, match5.positions.copy())
    assert errs["err"].size > 0
    assert errs["err"].max() < 1e-4


def test_porteros_excluidos_por_defecto(match5, view):
    est = run_ladder(match5, view, "B4")
    n_sin = position_errors(match5, view, est, include_gk=False)["err"].size
    n_con = position_errors(match5, view, est, include_gk=True)["err"].size
    assert n_con > n_sin


def test_los_bins_suman_uno(match5, view):
    est = run_ladder(match5, view, "B4")
    s = stratified_median(position_errors(match5, view, est))
    total = s["share_<=2s"] + s["share_2-9.6s"] + s["share_>9.6s"]
    assert total == pytest.approx(1.0, abs=1e-6)


def test_bootstrap_contiene_el_estimado(match5, view):
    est = run_ladder(match5, view, "B4")
    pt, lo, hi = block_bootstrap_ci(
        position_errors(match5, view, est), match5.fps, n_boot=200
    )
    assert lo <= pt <= hi


def test_bootstrap_de_bloques_es_mas_ancho_que_el_ingenuo(match5, view):
    """
    El punto entero del block bootstrap: remuestrear frames sueltos ignora la
    autocorrelación y produce intervalos artificialmente estrechos. El de
    bloques debe ser más ancho.
    """
    errs = position_errors(match5, view, run_ladder(match5, view, "B4"))
    _, lo_b, hi_b = block_bootstrap_ci(errs, match5.fps, block_seconds=60, n_boot=300)

    rng = np.random.default_rng(0)
    e = errs["err"]
    naive = np.array([np.median(rng.choice(e, e.size)) for _ in range(300)])
    lo_n, hi_n = np.percentile(naive, [2.5, 97.5])

    assert (hi_b - lo_b) > (hi_n - lo_n)


# Cargadores: convergencia entre proveedores


def _fake_kloppy_dataset(provider, n_frames=200, fps=25.0):
    """
    Construye un TrackingDataset de kloppy en memoria, sin red.

    Los jugadores se colocan en posiciones METRICAS conocidas y luego se
    convierten al sistema nativo del proveedor. Así, si el cargador es
    correcto, debe devolver exactamente las coordenadas métricas originales
    sin importar el proveedor.
    """
    from datetime import timedelta
    from kloppy.domain import (
        TrackingDataset, Frame, Team, Player, Period, Point, Ground, Metadata,
        DatasetType, Orientation, PlayerData, build_coordinate_system,
        PositionType, BallState,
    )

    cs = build_coordinate_system(provider, DatasetType.TRACKING,
                                 pitch_length=105, pitch_width=68)
    pdim = cs.pitch_dimensions
    x0, x1 = pdim.x_dim.min, pdim.x_dim.max
    y0, y1 = pdim.y_dim.min, pdim.y_dim.max
    to_x = lambda xm: x0 + (xm / 105.0) * (x1 - x0)   # noqa: E731
    to_y = lambda ym: y0 + (ym / 68.0) * (y1 - y0)    # noqa: E731

    teams = []
    for g, name in ((Ground.HOME, "Home"), (Ground.AWAY, "Away")):
        t = Team(team_id=name, name=name, ground=g)
        t.players = [
            Player(player_id=f"{name}{i}", team=t, jersey_no=i + 1,
                   starting_position=(PositionType.Goalkeeper if i == 0
                                      else PositionType.CenterBack))
            for i in range(11)
        ]
        teams.append(t)

    period = Period(id=1, start_timestamp=timedelta(0),
                    end_timestamp=timedelta(seconds=n_frames / fps))
    frames = []
    for k in range(n_frames):
        pdata = {}
        for ti, t in enumerate(teams):
            for j, p in enumerate(t.players):
                xm = 8 + (j * 8.5) % 88 + (0 if ti == 0 else 6)
                ym = 6 + (j * 5.5) % 55
                pdata[p] = PlayerData(coordinates=Point(to_x(xm), to_y(ym)))
        frames.append(Frame(
            frame_id=k, timestamp=timedelta(seconds=k / fps),
            ball_coordinates=Point(to_x(52.5 + 20 * np.sin(k / 40)), to_y(34.0)),
            players_data=pdata, period=period, ball_state=BallState.ALIVE,
            ball_owning_team=teams[0], other_data={}, statistics=[]))

    meta = Metadata(teams=teams, periods=[period], pitch_dimensions=pdim,
                    score=None, frame_rate=fps,
                    orientation=Orientation.STATIC_HOME_AWAY,
                    flags=None, provider=provider, coordinate_system=cs)
    return TrackingDataset(records=frames, metadata=meta)


def test_proveedores_convergen_al_mismo_esquema():
    """
    REGRESIÓN: Sportec usa métrico centrado (x en [-52.5, 52.5]) y Metrica usa
    normalizado (x en [0, 1]). Un cargador que asuma un solo sistema produce
    coordenadas catastróficamente equivocadas que aun así "parecen" números de
    cancha, y nada aguas abajo lo detecta.

    Este test coloca los mismos jugadores en las mismas posiciones físicas y
    exige que ambos proveedores produzcan tensores idénticos.
    """
    pytest.importorskip("kloppy")
    from kloppy.domain import Provider
    from ghosting.io.loaders import _from_kloppy

    out = {}
    for prov in (Provider.SPORTEC, Provider.METRICA):
        ds = _fake_kloppy_dataset(prov)
        out[prov.name] = _from_kloppy(ds, f"T_{prov.name}", prov.name.lower())

    a, b = out["SPORTEC"], out["METRICA"]
    assert np.nanmax(np.abs(a.positions - b.positions)) < 1e-3
    assert np.nanmax(np.abs(a.ball - b.ball)) < 1e-3
    assert a.pitch == b.pitch == (105.0, 68.0)


def test_cargador_produce_coordenadas_en_metros():
    """Las posiciones deben caer dentro de la cancha, no en [0,1] ni centradas."""
    pytest.importorskip("kloppy")
    from kloppy.domain import Provider
    from ghosting.io.loaders import _from_kloppy

    for prov in (Provider.SPORTEC, Provider.METRICA):
        m = _from_kloppy(_fake_kloppy_dataset(prov), "T", prov.name.lower())
        xs = m.positions[..., 0][np.isfinite(m.positions[..., 0])]
        assert xs.min() >= 0.0 and xs.max() <= 105.0
        assert xs.max() > 10.0, "parece seguir en coordenadas normalizadas [0,1]"


def test_porteros_detectados_por_metadatos():
    pytest.importorskip("kloppy")
    from kloppy.domain import Provider
    from ghosting.io.loaders import _from_kloppy

    m = _from_kloppy(_fake_kloppy_dataset(Provider.SPORTEC), "T", "sportec")
    assert m.is_gk.sum() == 2
    assert (m.is_gk & (m.team_idx == 0)).sum() == 1
    assert (m.is_gk & (m.team_idx == 1)).sum() == 1


def test_semilla_sintetica_es_estable_entre_procesos():
    """
    REGRESIÓN: hash() de Python está aleatorizado por proceso salvo que se fije
    PYTHONHASHSEED. Usarlo como semilla haría que los "datos sintéticos
    reproducibles" cambiaran entre ejecuciones.
    """
    from ghosting.io.loaders import _stable_seed

    assert _stable_seed("SYNTH01") == _stable_seed("SYNTH01")
    assert _stable_seed("SYNTH01") != _stable_seed("SYNTH02")
    # Valor fijo: si esto cambia, los datasets sintéticos previos ya no son
    # reproducibles y hay que regenerarlos.
    assert _stable_seed("SYNTH01") == 1682713977


def test_head_minutes_recorta_correctamente(match5):
    """El recorte a los primeros N minutos es exacto y no toca los ejes de jugador."""
    h = match5.head_minutes(2.0)
    assert h.n_frames == int(2 * 60 * match5.fps)
    assert h.n_players == match5.n_players
    assert np.allclose(h.positions, match5.positions[: h.n_frames], equal_nan=True)
    assert h.meta["truncated_to_minutes"] == 2.0


def test_head_minutes_no_desborda(match5):
    h = match5.head_minutes(10_000)
    assert h.n_frames == match5.n_frames


def test_ic_por_bin_contiene_el_estimado(match5, view):
    """El IC de cada bin debe contener su propia estimación puntual."""
    from ghosting.metrics.position import stratified_bootstrap_ci
    from ghosting.metrics import BIN_LABELS

    errs = position_errors(match5, view, run_ladder(match5, view, "B4"))
    point = stratified_median(errs)
    cis = stratified_bootstrap_ci(errs, match5.fps, n_boot=200)
    for label in BIN_LABELS:
        if label in cis:
            lo, hi = cis[label]
            assert lo <= point[f"median_{label}"] <= hi, label


def test_ic_del_bin_largo_es_el_mas_ancho(match5, view):
    """
    El bin de oclusión larga tiene menos rachas independientes, así que su
    incertidumbre debe ser mayor. Si sale más estrecho que el de gaps cortos,
    el bootstrap no está respetando la estructura de bloques.
    """
    from ghosting.metrics.position import stratified_bootstrap_ci

    errs = position_errors(match5, view, run_ladder(match5, view, "B4"))
    cis = stratified_bootstrap_ci(errs, match5.fps, n_boot=300)
    if "<=2s" in cis and ">9.6s" in cis:
        ancho = {k: hi - lo for k, (lo, hi) in cis.items()}
        assert ancho[">9.6s"] > ancho["<=2s"]


# Inferencia de portero (regresión: cambio de lado en el descanso)


def test_sintetico_cambia_de_lado_en_el_descanso(match):
    """
    Sin cambio de lado, toda la lógica que depende de qué portería defiende
    cada equipo queda sin probar. Este test garantiza que el generador lo
    reproduce.
    """
    L = match.pitch[0]
    for k in np.where(match.is_gk)[0]:
        x1 = np.nanmedian(match.positions[match.period == 1, k, 0])
        x2 = np.nanmedian(match.positions[match.period == 2, k, 0])
        assert abs(x1 - x2) > L / 2, f"el portero {k} no cambió de lado"


def test_inferencia_de_portero_con_cambio_de_lado(match):
    """
    REGRESIÓN CRÍTICA.

    Un portero defiende x≈5 en el primer tiempo y x≈100 en el segundo, así que
    su posición media sobre el partido completo es ~52.5 m: el centro de la
    cancha. Inferir el portero como "el de media más cercana a una meta" sobre
    todo el partido descarta exactamente al jugador buscado y elige a otro,
    normalmente un suplente.

    El fallo es silencioso: validate() pasa, las métricas corren, y los
    porteros reales quedan evaluados como jugadores de campo. Si además el
    suplente elegido no jugó el tramo analizado, `--include-gk` no cambia ni
    una muestra y el bug se disfraza de robustez.
    """
    from ghosting.io.loaders import _infer_goalkeepers

    # La media global de un portero real cae en el centro: esa es la trampa.
    for k in np.where(match.is_gk)[0]:
        media = np.nanmean(match.positions[:, k, 0])
        assert abs(media - match.pitch[0] / 2) < 10.0

    sin_meta = np.zeros(match.n_players, dtype=bool)
    inferido = _infer_goalkeepers(
        match.positions, match.team_idx, sin_meta,
        match.pitch, match.period, min_frames=100,
    )
    assert (inferido == match.is_gk).all(), (
        f"inferidos {np.where(inferido)[0].tolist()}, "
        f"reales {np.where(match.is_gk)[0].tolist()}"
    )


def test_inferencia_respeta_metadatos_existentes(match):
    """Si los metadatos ya marcan portero, la inferencia no debe pisarlos."""
    from ghosting.io.loaders import _infer_goalkeepers

    out = _infer_goalkeepers(
        match.positions, match.team_idx, match.is_gk, match.pitch, match.period
    )
    assert (out == match.is_gk).all()


def test_include_gk_cambia_la_poblacion(match5, view):
    """
    REGRESIÓN: si `--include-gk` no altera el número de muestras puntuables,
    la bandera no está tocando a nadie — señal de que `is_gk` señala a
    jugadores que no aportan observaciones (p. ej. suplentes mal inferidos).
    """
    est = run_ladder(match5, view, "B4")
    n_sin = position_errors(match5, view, est, include_gk=False)["err"].size
    n_con = position_errors(match5, view, est, include_gk=True)["err"].size
    assert n_con > n_sin, "include_gk no añadió ninguna muestra"


def test_ancla_de_portero_cambia_su_estimacion(match5, view):
    """
    Las dos políticas de anclaje deben producir estimaciones distintas para el
    portero, e idénticas para los jugadores de campo.
    """
    from ghosting.baselines import LadderConfig

    a = run_ladder(match5, view, "B4", LadderConfig(gk_anchor="goal"))
    b = run_ladder(match5, view, "B4", LadderConfig(gk_anchor="team"))
    gk = match5.is_gk

    dif_gk = np.nanmax(np.abs(a[:, gk] - b[:, gk]))
    assert dif_gk > 1.0, "el anclaje del portero no tuvo efecto"
    assert np.allclose(a[:, ~gk], b[:, ~gk], equal_nan=True), \
        "el anclaje del portero alteró a los jugadores de campo"


# Comparación pareada


def test_pareado_usa_poblacion_identica(match5, view):
    """Los dos estimadores deben evaluarse sobre exactamente los mismos pares."""
    from ghosting.metrics import paired_position_errors
    from ghosting.baselines import LadderConfig

    a = run_ladder(match5, view, "B4", LadderConfig(gk_anchor="goal"))
    b = run_ladder(match5, view, "B4", LadderConfig(gk_anchor="team"))
    pr = paired_position_errors(match5, view, a, b, include_gk=True)
    assert pr["err_a"].size == pr["err_b"].size == pr["gap"].size == pr["frame"].size
    assert pr["err_a"].size > 0


def test_pareado_contra_si_mismo_da_cero(match5, view):
    """Comparar un estimador consigo mismo debe dar diferencia exactamente 0."""
    from ghosting.metrics import paired_position_errors, paired_block_bootstrap_ci

    est = run_ladder(match5, view, "B4")
    pr = paired_position_errors(match5, view, est, est)
    ci = paired_block_bootstrap_ci(pr, match5.fps, n_boot=100)
    for _, (d, lo, hi) in ci.items():
        assert abs(d) < 1e-9 and abs(lo) < 1e-9 and abs(hi) < 1e-9


def test_pareado_es_mas_estrecho_que_marginales(match5, view):
    """
    El punto entero del bootstrap pareado: al restar dentro de cada réplica se
    cancela la incertidumbre común a ambos estimadores. El intervalo de la
    diferencia debe ser MUCHO más estrecho que la suma de los marginales.
    """
    from ghosting.metrics import (
        paired_position_errors, paired_block_bootstrap_ci, block_bootstrap_ci,
    )
    from ghosting.baselines import LadderConfig

    a = run_ladder(match5, view, "B4", LadderConfig(gk_anchor="goal"))
    b = run_ladder(match5, view, "B2", LadderConfig(gk_anchor="goal"))

    pr = paired_position_errors(match5, view, a, b)
    d, lo, hi = paired_block_bootstrap_ci(pr, match5.fps, n_boot=300)["global"]
    ancho_pareado = hi - lo

    ea = position_errors(match5, view, a)
    eb = position_errors(match5, view, b)
    _, la, ha = block_bootstrap_ci(ea, match5.fps, n_boot=300)
    _, lb, hb = block_bootstrap_ci(eb, match5.fps, n_boot=300)
    ancho_marginal = (ha - la) + (hb - lb)

    assert ancho_pareado < ancho_marginal


def test_pareado_detecta_una_diferencia_conocida(match5, view):
    """
    B4 le gana a B1 por un margen grande y consistente. El intervalo pareado
    tiene que excluir el cero; si no, el procedimiento no sirve para nada.
    """
    from ghosting.metrics import paired_position_errors, paired_block_bootstrap_ci

    b1 = run_ladder(match5, view, "B1")
    b4 = run_ladder(match5, view, "B4")
    pr = paired_position_errors(match5, view, b1, b4)
    d, lo, hi = paired_block_bootstrap_ci(pr, match5.fps, n_boot=300)["global"]
    assert d > 0, "B1 debería tener más error que B4"
    assert lo > 0, f"el intervalo pareado [{lo:.2f},{hi:.2f}] no excluye el cero"


def test_pool_paired_no_mezcla_bloques_entre_partidos():
    """
    REGRESIÓN: los índices de frame se reinician en cada partido. Concatenar
    sin desplazar fundiría el minuto 3 del partido A con el minuto 3 del B en
    un mismo bloque de remuestreo — observaciones de partidos distintos
    tratadas como contiguas.
    """
    from ghosting.metrics import pool_paired

    fps = 5.0
    block = int(60 * fps)
    piezas = []
    for _ in range(3):
        n = 400
        piezas.append({
            "err_a": np.ones(n), "err_b": np.ones(n), "gap": np.ones(n),
            "frame": np.arange(n),
        })
    pooled = pool_paired(piezas, fps)

    assert pooled["err_a"].size == 1200
    block_id = pooled["frame"] // block
    # Cada bloque debe pertenecer a un único partido de origen
    origen = np.repeat([0, 1, 2], 400)
    for b in np.unique(block_id):
        assert len(set(origen[block_id == b])) == 1, f"el bloque {b} mezcla partidos"


def test_pool_paired_estrecha_el_intervalo(match5, view):
    """
    Agrupar partidos reduce la incertidumbre, pero NO por debajo del partido
    más afortunado.

    Si los partidos tienen efectos genuinamente distintos (heterogeneidad), el
    agrupado incorpora esa varianza entre partidos además de la de muestreo. Un
    intervalo estrecho salido de un solo partido es sobreconfiado respecto a la
    generalización; el agrupado lo corrige. Por eso la comparación correcta es
    contra el ANCHO PROMEDIO, no contra el mínimo.
    """
    from ghosting.metrics import (
        paired_position_errors, paired_block_bootstrap_ci, pool_paired,
    )
    from ghosting.io import synthetic_match
    from ghosting.camera import simulate, ViewportConfig

    piezas = []
    for seed in (11, 12, 13, 14):
        m = synthetic_match(minutes=6.0, seed=seed).resample(5.0)
        v = simulate(m, ViewportConfig(width_m=44.0))
        piezas.append(paired_position_errors(
            m, v, run_ladder(m, v, "B1"), run_ladder(m, v, "B4")
        ))

    anchos = []
    for p in piezas:
        _, lo, hi = paired_block_bootstrap_ci(p, 5.0, n_boot=300)["global"]
        anchos.append(hi - lo)

    pooled = pool_paired(piezas, 5.0)
    _, lo_p, hi_p = paired_block_bootstrap_ci(pooled, 5.0, n_boot=300)["global"]

    promedio = sum(anchos) / len(anchos)
    assert (hi_p - lo_p) < promedio, (
        f"agrupado {hi_p - lo_p:.2f} no es más estrecho que el promedio "
        f"individual {promedio:.2f}"
    )


def test_pool_paired_lista_vacia():
    from ghosting.metrics import pool_paired
    out = pool_paired([], 5.0)
    assert out["err_a"].size == 0


# Modelo residual


def test_modelo_arranca_siendo_exactamente_b4(match5, view):
    """
    INVARIANTE CENTRAL del diseño residual: con la cabeza inicializada a cero,
    el modelo predice exactamente B4. Eso acota el riesgo a la baja — no puede
    ser catastróficamente peor que el baseline — y es lo que hace viable
    entrenar con 7 partidos.
    """
    torch = pytest.importorskip("torch")
    from ghosting.models import build_windows, ResidualImputer

    b4 = run_ladder(match5, view, "B4")
    w = build_windows(match5, view, b4)
    assert len(w) > 0

    model = ResidualImputer(dim=32, n_blocks=1)
    b = {k: torch.from_numpy(np.stack([x[k] for x in w[:4]])) for k in w[0]}
    with torch.no_grad():
        pred = model.predict_positions(b["feats"], b["player_mask"], b["base"])
    assert torch.allclose(pred, b["base"], atol=1e-6)


def test_modelo_es_equivariante_a_permutaciones(match5, view):
    """
    Los jugadores son un CONJUNTO. Reordenar la entrada debe reordenar la
    salida igual. Si esto falla, el modelo memoriza el orden del roster en vez
    de aprender estructura de juego, y no generaliza a otro partido.

    SE COMPRUEBA EN FLOAT64, Y NO ES UN DETALLE
    -------------------------------------------
    En float32 la atención suma 26 términos; hacerlo en distinto orden cambia
    el último bit y produce discrepancias de ~1e-5 que no tienen nada que ver
    con la equivarianza (la aritmética de punto flotante no es asociativa).
    Con el umbral en 1e-5 este test fallaba de forma intermitente según la
    permutación que tocara.

    La solución perezosa sería aflojar la tolerancia, pero eso enmascararía una
    violación real: si el modelo dependiera del orden, la discrepancia sería de
    orden 1, no de 1e-5. En float64 el ruido numérico baja a ~1e-14, así que un
    umbral estricto distingue ambas cosas sin ambigüedad.
    """
    torch = pytest.importorskip("torch")
    from ghosting.models import build_windows, ResidualImputer

    b4 = run_ladder(match5, view, "B4")
    w = build_windows(match5, view, b4)
    model = ResidualImputer(dim=32, n_blocks=2)
    # Cabeza no nula: con pesos cero la equivarianza sería trivial.
    torch.nn.init.normal_(model.head.weight, std=0.1)
    model.double().eval()

    b = {k: torch.from_numpy(np.stack([x[k] for x in w[:2]])) for k in w[0]}
    feats = b["feats"].double()
    with torch.no_grad():
        r1 = model(feats, b["player_mask"])

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(feats.shape[2], generator=g)
    with torch.no_grad():
        r2 = model(feats[:, :, perm], b["player_mask"][:, perm])

    dif = float((r1[:, :, perm] - r2).abs().max())
    assert dif < 1e-10, f"discrepancia {dif:.2e}: el modelo depende del orden"


def test_mascara_causal_impide_ver_el_futuro(match5, view):
    """
    En modo causal, alterar los frames futuros no debe cambiar la predicción
    del frame t. Si cambia, el modelo está interpolando y no puede venderse
    como tiempo real.
    """
    torch = pytest.importorskip("torch")
    from ghosting.models import build_windows, ResidualImputer

    b4 = run_ladder(match5, view, "B4")
    w = build_windows(match5, view, b4)
    model = ResidualImputer(dim=32, n_blocks=2, causal=True)
    torch.nn.init.normal_(model.head.weight, std=0.1)
    model.eval()

    b = {k: torch.from_numpy(np.stack([x[k] for x in w[:2]])) for k in w[0]}
    T = b["feats"].shape[1]
    cut = T // 2

    with torch.no_grad():
        r1 = model(b["feats"], b["player_mask"])
        alterado = b["feats"].clone()
        alterado[:, cut:] = torch.randn_like(alterado[:, cut:])
        r2 = model(alterado, b["player_mask"])

    assert torch.allclose(r1[:, :cut], r2[:, :cut], atol=1e-5), \
        "el modelo causal está usando información del futuro"


def test_perdida_penaliza_jitter_dentro_de_la_region_factible():
    """
    Las bisagras cinemáticas tienen gradiente NULO por debajo de v_max, así que
    no castigan oscilación de alta frecuencia. El término de suavidad existe
    justo para eso: una trayectoria que tiembla debe puntuar peor que una
    recta, aunque ninguna viole las cotas.
    """
    torch = pytest.importorskip("torch")
    from ghosting.models import imputer_loss

    B, T, N = 1, 20, 3
    base = torch.zeros(B, T, N, 2)
    target = torch.zeros(B, T, N, 2)
    mask = torch.ones(B, T, N, dtype=torch.bool)

    t = torch.arange(T, dtype=torch.float32)
    suave = torch.zeros(B, T, N, 2)
    suave[..., 0] = (t * 0.1)[None, :, None]

    jitter = suave.clone()
    jitter[..., 1] = (torch.where(t % 2 == 0, 0.4, -0.4))[None, :, None]

    _, p_s = imputer_loss(suave, target, base, mask, fps=5.0)
    _, p_j = imputer_loss(jitter, target, base, mask, fps=5.0)

    # Ninguna supera v_max = 11 m/s, así que la bisagra no las distingue...
    assert p_s["vel"] == pytest.approx(p_j["vel"], abs=1e-6)
    # ...pero la suavidad sí.
    assert p_j["smooth"] > p_s["smooth"] * 10


def test_ventanas_tienen_forma_consistente(match5, view):
    from ghosting.models import build_windows, WindowConfig, N_FEATURES

    cfg = WindowConfig(length=30, stride=15, max_players=24)
    w = build_windows(match5, view, run_ladder(match5, view, "B4"), cfg)
    assert len(w) > 0
    for x in w[:5]:
        assert x["feats"].shape == (30, 24, N_FEATURES)
        assert x["target"].shape == (30, 24, 2)
        assert x["loss_mask"].shape == (30, 24)
        assert x["player_mask"].shape == (24,)
        # El objetivo es cero donde no hay verdad con la que comparar
        assert np.allclose(x["target"][~x["loss_mask"]], 0.0)


# Control de cancha


def test_control_es_simetrico_al_intercambiar_equipos(match5):
    """
    Intercambiar los equipos debe reflejar el mapa: C_local(r) = 1 - C_visitante(r).
    Si no se cumple, hay un sesgo en el modelo que invalidaría la comparación
    entre paneles.
    """
    from ghosting.metrics import make_grid, pitch_control

    _, _, pts = make_grid(match5.pitch, step=5.0)
    h = match5.positions[0, match5.team_idx == 0]
    a = match5.positions[0, match5.team_idx == 1]
    c1 = pitch_control(pts, h, a)
    c2 = pitch_control(pts, a, h)
    assert np.allclose(c1, 1.0 - c2, atol=1e-9)


def test_control_neutro_sin_jugadores(match5):
    """Sin jugadores de un equipo no hay información: 0.5 en todo el campo."""
    from ghosting.metrics import make_grid, pitch_control

    _, _, pts = make_grid(match5.pitch, step=5.0)
    h = match5.positions[0, match5.team_idx == 0]
    c = pitch_control(pts, h, np.zeros((0, 2)))
    assert np.allclose(c, 0.5)


def test_quitar_defensores_desplaza_el_control(match5, view):
    """
    El supuesto central de la figura: ignorar a los jugadores ocultos distorsiona
    el mapa de control. Si quitar defensores no cambiara nada, toda la
    imputación sería irrelevante.
    """
    from ghosting.metrics import make_grid, pitch_control, control_mae

    _, _, pts = make_grid(match5.pitch, step=4.0)
    t = int(np.argmin(view.visible.sum(axis=1)))   # el frame con más ocultos
    on, vis = match5.on_pitch[t], view.visible[t]
    h, a = match5.team_idx == 0, match5.team_idx == 1

    completo = pitch_control(pts, match5.positions[t, on & h],
                             match5.positions[t, on & a])
    ciego = pitch_control(pts, match5.positions[t, vis & h],
                          match5.positions[t, vis & a])
    assert control_mae(completo, ciego) > 1.0
