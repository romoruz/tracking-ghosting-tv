"""
Esquema canónico de un partido.

Todo el proyecto opera sobre esta estructura, sin importar el proveedor de
origen (Sportec, Metrica, SkillCorner). Los cargadores de `loaders.py` son
responsables de traducir a este esquema; nada aguas abajo debe saber de qué
proveedor vinieron los datos.

Convención de coordenadas
-------------------------
Cancha de L x A metros con origen en la esquina inferior izquierda:

    (0, A) ---------------- (L, A)
      |                        |
      |                        |
    (0, 0) ---------------- (L, 0)

El eje x es el eje largo (dirección de ataque). Por defecto L=105, A=68.
Se usa esquina-origen y no centro-origen porque simplifica las grillas de
pitch control y los recortes de viewport.

IMPORTANTE: no se normaliza la dirección de ataque. Los equipos cambian de
lado en el segundo tiempo y eso es correcto: la cámara sigue al balón en
coordenadas absolutas de estadio, no relativas al equipo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Dimensiones por defecto (estándar FIFA / Bundesliga)
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


@dataclass
class Match:
    """
    Un partido en representación canónica.

    Atributos de forma (T, N, ...) donde T = número de frames y
    N = número de jugadores únicos que aparecen en el partido (típicamente
    28-32 por las sustituciones, NO 22).

    Attributes
    ----------
    match_id : str
        Identificador del partido.
    positions : np.ndarray, shape (T, N, 2), dtype float32
        Posiciones en metros. Valores NaN donde el jugador no está en cancha.
    ball : np.ndarray, shape (T, 2), dtype float32
        Posición del balón en metros. NaN si el balón no fue rastreado.
    on_pitch : np.ndarray, shape (T, N), dtype bool
        True si el jugador i está físicamente en cancha en el frame t.
        Cubre titulares/suplentes, expulsiones y el descanso.
        Este es el "ground truth de existencia", distinto de la visibilidad
        de cámara que se calcula en `camera/viewport.py`.
    team_idx : np.ndarray, shape (N,), dtype int8
        0 = equipo local, 1 = equipo visitante.
    is_gk : np.ndarray, shape (N,), dtype bool
        True si el jugador es portero. Usado para anclaje diferenciado
        (ver docs/02_formalismo_matematico.md, sección "Partición del portero").
    player_ids : list[str], len N
        Identificadores originales del proveedor.
    period : np.ndarray, shape (T,), dtype int8
        1 = primer tiempo, 2 = segundo tiempo, etc.
    ball_alive : np.ndarray, shape (T,), dtype bool
        True si el balón está en juego. Los frames muertos se excluyen de
        las métricas pero se conservan para no romper la continuidad temporal.
    fps : float
        Frecuencia de muestreo en frames por segundo.
    pitch : tuple[float, float]
        (largo, ancho) en metros.
    provider : str
        Proveedor de origen, solo para trazabilidad.
    meta : dict
        Metadatos libres (nombres de equipos, fecha, etc.).
    """

    match_id: str
    positions: np.ndarray
    ball: np.ndarray
    on_pitch: np.ndarray
    team_idx: np.ndarray
    is_gk: np.ndarray
    player_ids: list[str]
    period: np.ndarray
    ball_alive: np.ndarray
    fps: float
    pitch: tuple[float, float] = (PITCH_LENGTH, PITCH_WIDTH)
    provider: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)



    @property
    def n_frames(self) -> int:
        return self.positions.shape[0]

    @property
    def n_players(self) -> int:
        return self.positions.shape[1]

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps



    def validate(self) -> None:
        """
        Verifica invariantes del esquema. Falla ruidosamente si algo está mal.

        Se llama al final de cada cargador. Un error aquí significa que el
        cargador está mal, no que los datos estén mal: es una red de seguridad
        contra bugs de traducción entre proveedores.
        """
        T, N = self.n_frames, self.n_players

        assert self.positions.shape == (T, N, 2), (
            f"positions debe ser (T,N,2), es {self.positions.shape}"
        )
        assert self.ball.shape == (T, 2), f"ball debe ser (T,2), es {self.ball.shape}"
        assert self.on_pitch.shape == (T, N), (
            f"on_pitch debe ser (T,N), es {self.on_pitch.shape}"
        )
        assert self.team_idx.shape == (N,), (
            f"team_idx debe ser (N,), es {self.team_idx.shape}"
        )
        assert self.is_gk.shape == (N,), f"is_gk debe ser (N,), es {self.is_gk.shape}"
        assert len(self.player_ids) == N, (
            f"player_ids debe tener N={N} entradas, tiene {len(self.player_ids)}"
        )
        assert self.period.shape == (T,), f"period debe ser (T,), es {self.period.shape}"
        assert self.ball_alive.shape == (T,), (
            f"ball_alive debe ser (T,), es {self.ball_alive.shape}"
        )

        # Los equipos deben ser exactamente {0, 1}
        teams = set(np.unique(self.team_idx).tolist())
        assert teams <= {0, 1}, f"team_idx debe estar en {{0,1}}, encontrado {teams}"

        # Cada equipo debe tener al menos un portero
        for t in (0, 1):
            n_gk = int((self.is_gk & (self.team_idx == t)).sum())
            assert n_gk >= 1, f"El equipo {t} no tiene portero registrado"

        # Coherencia entre on_pitch y NaN: si está en cancha, debe tener posición
        in_pitch_nan = np.isnan(self.positions).any(axis=2) & self.on_pitch
        frac_bad = float(in_pitch_nan.mean())
        assert frac_bad < 0.02, (
            f"{frac_bad:.2%} de los frames tienen on_pitch=True pero posición NaN. "
            "Revisa el cargador."
        )

        # Nadie fuera de los límites de la cancha por un margen absurdo
        L, A = self.pitch
        finite = np.isfinite(self.positions)
        xs = self.positions[..., 0][finite[..., 0]]
        ys = self.positions[..., 1][finite[..., 1]]
        if xs.size:
            assert xs.min() > -10 and xs.max() < L + 10, (
                f"x fuera de rango: [{xs.min():.1f}, {xs.max():.1f}] con L={L}"
            )
        if ys.size:
            assert ys.min() > -10 and ys.max() < A + 10, (
                f"y fuera de rango: [{ys.min():.1f}, {ys.max():.1f}] con A={A}"
            )



    def save(self, path: str | Path) -> Path:
        """Guarda como .npz comprimido."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            match_id=self.match_id,
            positions=self.positions.astype(np.float32),
            ball=self.ball.astype(np.float32),
            on_pitch=self.on_pitch,
            team_idx=self.team_idx.astype(np.int8),
            is_gk=self.is_gk,
            player_ids=np.array(self.player_ids, dtype=object),
            period=self.period.astype(np.int8),
            ball_alive=self.ball_alive,
            fps=self.fps,
            pitch=np.array(self.pitch, dtype=np.float32),
            provider=self.provider,
            meta=np.array([self.meta], dtype=object),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Match":
        """Carga desde .npz."""
        d = np.load(Path(path), allow_pickle=True)
        return cls(
            match_id=str(d["match_id"]),
            positions=d["positions"],
            ball=d["ball"],
            on_pitch=d["on_pitch"],
            team_idx=d["team_idx"],
            is_gk=d["is_gk"],
            player_ids=list(d["player_ids"]),
            period=d["period"],
            ball_alive=d["ball_alive"],
            fps=float(d["fps"]),
            pitch=tuple(d["pitch"].tolist()),
            provider=str(d["provider"]),
            meta=dict(d["meta"][0]),
        )



    def resample(self, target_fps: float) -> "Match":
        """
        Submuestrea por decimación entera. No interpola.

        El benchmark de Choi (2026) evalúa a 5 fps mientras que el tracking
        nativo de Sportec y Metrica es de 25 fps. Decimar (y no promediar)
        preserva posiciones reales en lugar de inventar puntos intermedios.
        """
        if target_fps > self.fps:
            raise ValueError(
                f"No se puede subir de {self.fps} a {target_fps} fps sin interpolar"
            )
        step = int(round(self.fps / target_fps))
        if abs(self.fps / step - target_fps) > 1e-6:
            raise ValueError(
                f"{self.fps} fps no es divisible enteramente entre {target_fps} fps"
            )
        sl = slice(None, None, step)
        return Match(
            match_id=self.match_id,
            positions=self.positions[sl],
            ball=self.ball[sl],
            on_pitch=self.on_pitch[sl],
            team_idx=self.team_idx,
            is_gk=self.is_gk,
            player_ids=self.player_ids,
            period=self.period[sl],
            ball_alive=self.ball_alive[sl],
            fps=self.fps / step,
            pitch=self.pitch,
            provider=self.provider,
            meta={**self.meta, "resampled_from_fps": self.fps},
        )

    def head_minutes(self, minutes: float) -> "Match":
        """
        Recorta a los primeros `minutes` minutos de partido.

        Existe por una razón de protocolo, no de conveniencia: Choi (2026)
        evalúa "los primeros 45 minutos de cada partido". Comparar contra sus
        cifras usando el partido completo introduce una diferencia sistemática:
        el segundo tiempo trae sustituciones, fatiga y cambios tácticos, y las
        tres cosas degradan específicamente a los métodos anclados a formación
        en oclusiones largas. Un suplente que entra no tiene offset almacenado,
        así que B4 cae a B2 y luego a B1, cuyo error en la cola es el doble.
        """
        n = int(round(minutes * 60 * self.fps))
        n = min(n, self.n_frames)
        return Match(
            match_id=self.match_id,
            positions=self.positions[:n],
            ball=self.ball[:n],
            on_pitch=self.on_pitch[:n],
            team_idx=self.team_idx,
            is_gk=self.is_gk,
            player_ids=self.player_ids,
            period=self.period[:n],
            ball_alive=self.ball_alive[:n],
            fps=self.fps,
            pitch=self.pitch,
            provider=self.provider,
            meta={**self.meta, "truncated_to_minutes": minutes},
        )

    def summary(self) -> str:
        """Resumen legible para logs."""
        n_home = int((self.team_idx == 0).sum())
        n_away = int((self.team_idx == 1).sum())
        mean_on = float(self.on_pitch.sum(axis=1).mean())
        alive = float(self.ball_alive.mean())
        return (
            f"Match {self.match_id} [{self.provider}]\n"
            f"  frames    : {self.n_frames:,} @ {self.fps:g} fps "
            f"({self.duration_s / 60:.1f} min)\n"
            f"  jugadores : {self.n_players} "
            f"(local {n_home} / visitante {n_away}, "
            f"{int(self.is_gk.sum())} porteros)\n"
            f"  en cancha : {mean_on:.1f} en promedio\n"
            f"  balón vivo: {alive:.1%} de los frames\n"
            f"  cancha    : {self.pitch[0]:g} x {self.pitch[1]:g} m"
        )
