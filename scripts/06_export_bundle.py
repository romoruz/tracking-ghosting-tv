#!/usr/bin/env python3
"""
Exporta un paquete mínimo para entrenar en Colab / Kaggle.

POR QUÉ NO HAY QUE DESCARGAR NADA EN LA NUBE
La intuición natural es "habrá que bajar la base de datos otra vez y eso se
come la sesión de GPU". Es al revés: los datos crudos ya están procesados en
`data/processed/`, y el entrenamiento solo usa una fracción de ellos.

El .npz que genera el pipeline guarda el partido entero a 25 fps (~42 MB por
partido de Sportec). Pero el entrenamiento corre a 5 fps sobre los primeros
45 minutos, y descarta a los jugadores que nunca pisan el campo. Eso es ~4 MB
por partido: **unos 36 MB para los nueve**, que suben en segundos.

Descargar Sportec desde HuggingFace dentro de la sesión de GPU sería tirar
entre 15 y 30 minutos de cómputo alquilado en trabajo de red y de parseo XML
que ya hiciste en tu laptop.

Uso:
    python scripts/06_export_bundle.py
    python scripts/06_export_bundle.py --fps 5 --minutes 45 --out bundle.zip
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from ghosting.io import Match  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def compactar(m: Match, fps: float, minutes: float) -> Match:
    """
    Reduce a lo que el entrenamiento realmente consume.

    Tres recortes, en este orden:
      1. Submuestreo a `fps` (decimación, no interpolación).
      2. Truncado a los primeros `minutes`.
      3. Eliminación de jugadores que nunca están en cancha en ese tramo
         (suplentes que entraron en el segundo tiempo, sobre todo).

    El tercero importa más de lo que parece: los rosters de Sportec listan 40
    jugadores y solo ~22 juegan el primer tiempo.
    """
    if m.fps > fps:
        m = m.resample(fps)
    if minutes:
        m = m.head_minutes(minutes)

    usados = m.on_pitch.any(axis=0)
    if usados.sum() < 22:      # salvaguarda: nunca dejar menos de 22
        usados[:] = True

    return Match(
        match_id=m.match_id,
        positions=m.positions[:, usados],
        ball=m.ball,
        on_pitch=m.on_pitch[:, usados],
        team_idx=m.team_idx[usados],
        is_gk=m.is_gk[usados],
        player_ids=[p for p, k in zip(m.player_ids, usados) if k],
        period=m.period,
        ball_alive=m.ball_alive,
        fps=m.fps,
        pitch=m.pitch,
        provider=m.provider,
        meta={**m.meta, "compacted": True},
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--minutes", type=float, default=45.0)
    ap.add_argument("--out", default="ghosting_bundle.zip")
    ap.add_argument("--split", action="store_true",
                    help="genera DOS zips: datos (no cambia nunca) y código "
                         "(cambia en cada iteración, ~200 KB). Recomendado si vas "
                         "a iterar: subir 200 KB tarda segundos frente a 15 MB")
    ap.add_argument("--include-synthetic", action="store_true",
                    help="incluir partidos sintéticos (por defecto se omiten)")
    args = ap.parse_args()

    files = sorted(PROC.glob("*.npz"))
    if not args.include_synthetic:
        files = [f for f in files if "SYNTH" not in f.stem.upper()]
    if not files:
        print("No hay partidos en data/processed/. Corre antes scripts/00_download.py")
        return 1

    tmp = ROOT / "data" / "_bundle"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "data" / "processed").mkdir(parents=True)

    total_ori = total_new = 0
    print(f"{'partido':<16}{'original':>11}{'compacto':>11}{'jugadores':>11}")
    print("-" * 49)
    for f in files:
        m = Match.load(f)
        n0 = m.n_players
        c = compactar(m, args.fps, args.minutes)
        dest = tmp / "data" / "processed" / f.name
        c.save(dest)
        o, n = f.stat().st_size, dest.stat().st_size
        total_ori += o
        total_new += n
        print(f"{m.match_id:<16}{o/1e6:>10.1f}M{n/1e6:>10.1f}M"
              f"{f'{n0}->{c.n_players}':>11}")

    # El código fuente viaja con los datos: la sesión de nube queda autocontenida.
    for sub in ("src", "scripts", "tests"):
        shutil.copytree(ROOT / sub, tmp / sub,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in ("pyproject.toml", "requirements.txt", "README.md"):
        if (ROOT / f).exists():
            shutil.copy(ROOT / f, tmp / f)

    if args.split:
        # Dos paquetes: los datos son estables y pesan; el código es ligero y
        # cambia en cada iteración. Separarlos evita resubir 15 MB para
        # cambiar tres archivos.
        d_out = ROOT / "ghosting_data.zip"
        c_out = ROOT / "ghosting_code.zip"
        with zipfile.ZipFile(d_out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in (tmp / "data").rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(tmp))
        with zipfile.ZipFile(c_out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in tmp.rglob("*"):
                if p.is_file() and not str(p.relative_to(tmp)).startswith("data/"):
                    z.write(p, p.relative_to(tmp))
        shutil.rmtree(tmp)
        print("-" * 49)
        print(f"{'datos  ':<16}{d_out.stat().st_size/1e6:>10.1f}M  (sube una vez)")
        print(f"{'código ':<16}{c_out.stat().st_size/1e6:>10.1f}M  (sube en cada iteración)")
        print(f"\n-> {d_out}\n-> {c_out}")
        print("\nCrea DOS datasets en Kaggle: 'ghosting-data' y 'ghosting-code'.")
        print("A partir de ahí solo actualizas el segundo, que tarda segundos.")
        return 0

    out = ROOT / args.out
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in tmp.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(tmp))
    shutil.rmtree(tmp)

    print("-" * 49)
    print(f"{'total datos':<16}{total_ori/1e6:>10.1f}M{total_new/1e6:>10.1f}M"
          f"{f'{total_ori/max(total_new,1):.0f}x':>11}")
    print(f"\npaquete (datos + código): {out.stat().st_size/1e6:.1f} MB")
    print(f"-> {out}")
    print("\nSúbelo a Kaggle como Dataset privado, o a Google Drive para Colab.")
    print("La subida se hace ANTES de arrancar la GPU: no consume tiempo alquilado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
