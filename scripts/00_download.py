#!/usr/bin/env python3
"""
Descarga y convierte los datos abiertos al esquema canónico.

Uso:
    python scripts/00_download.py --provider sportec           # los 7 partidos
    python scripts/00_download.py --provider sportec --match J03WMX
    python scripts/00_download.py --provider metrica
    python scripts/00_download.py --provider synthetic         # sin red

Los .npz quedan en data/processed/. La descarga de Sportec pasa por
HuggingFace vía kloppy: la primera vez tarda varios minutos por partido.
"""
import argparse, sys, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ghosting.io import load, SPORTEC_OPEN_MATCHES  # noqa: E402
from ghosting.io.loaders import METRICA_OPEN_MATCHES  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "processed"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", required=True,
                    choices=["sportec", "metrica", "synthetic"])
    ap.add_argument("--match", default=None, help="ID concreto; por defecto, todos")
    ap.add_argument("--force", action="store_true",
                    help="rehacer aunque exista (OBLIGATORIO tras cambiar el "
                         "cargador: is_gk queda guardado dentro del .npz)")
    args = ap.parse_args()

    if args.match:
        ids = [args.match]
    elif args.provider == "sportec":
        ids = list(SPORTEC_OPEN_MATCHES)
    elif args.provider == "metrica":
        ids = list(METRICA_OPEN_MATCHES)
    else:
        ids = ["SYNTH01", "SYNTH02", "SYNTH03"]

    OUT.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for mid in ids:
        dest = OUT / f"{args.provider}_{mid}.npz"
        if dest.exists() and not args.force:
            print(f"[  ok  ] {dest.name} (ya existe)")
            ok += 1
            continue
        print(f"[ .... ] {args.provider}:{mid}", flush=True)
        try:
            m = load(args.provider, mid)
            m.save(dest)
            print(f"[  ok  ] {dest.name}")
            print("\n".join("         " + l for l in m.summary().splitlines()))
            ok += 1
        except Exception as e:
            print(f"[ FALLO] {mid}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            fail += 1

    print(f"\n{ok} correctos, {fail} fallidos -> {OUT}")
    return 1 if fail and not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
