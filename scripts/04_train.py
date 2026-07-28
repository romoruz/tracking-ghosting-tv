#!/usr/bin/env python3
"""
Entrena el imputador residual sobre B4.

Uso:
    # prueba rápida sin datos reales
    python scripts/04_train.py --provider synthetic --epochs 3

    # entrenamiento real
    python scripts/04_train.py --epochs 40

    # variante bidireccional (interpolación; NO es tiempo real)
    python scripts/04_train.py --epochs 40 --bidirectional

Partición POR PARTIDO (nunca por ventana): a 5 fps con solapamiento, dos
ventanas vecinas comparten casi todos sus frames.

    train : J03WMX J03WN1 J03WOH J03WOY J03WPY
    val   : J03WQQ
    test  : J03WR9
    externo congelado : metrica_1 metrica_2   (no se toca hasta el final)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from ghosting.io import Match  # noqa: E402
from ghosting.camera import simulate, ViewportConfig  # noqa: E402
from ghosting.baselines import run_ladder, LadderConfig  # noqa: E402
from ghosting.models import (  # noqa: E402
    build_windows, WindowConfig, ResidualImputer, imputer_loss, median_error_m,
)

ROOT = Path(__file__).resolve().parents[1]
PROC, OUT = ROOT / "data" / "processed", ROOT / "reports"

SPLITS = {
    "train": ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY"],
    "val": ["J03WQQ"],
    "test": ["J03WR9"],
}


class Windows(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return {k: torch.from_numpy(v) for k, v in self.items[i].items()}


def load_split(names, args, cfg_w):
    """Carga partidos, calcula B4 sobre el partido completo y trocea."""
    out = []
    for stem in names:
        hits = list(PROC.glob(f"*{stem}*.npz"))
        if not hits:
            print(f"  [aviso] no encontrado: {stem}")
            continue
        m = Match.load(hits[0])
        if m.fps > args.fps:
            m = m.resample(args.fps)
        if args.minutes:
            m = m.head_minutes(args.minutes)
        view = simulate(m, ViewportConfig(width_m=args.width))
        # B4 sobre el partido COMPLETO: sus offsets son recursivos y
        # calcularlo por ventana reiniciaría el estado.
        b4 = run_ladder(m, view, "B4", LadderConfig())
        w = build_windows(m, view, b4, cfg_w)
        out.extend(w)
        print(f"  {m.match_id:<12} {m.n_frames:>7,} frames -> {len(w):>5} ventanas")
    return out


def _progress(iterable, desc, total, enabled):
    """Barra de progreso si tqdm está disponible; si no, iterador pelado."""
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total, leave=False,
                    ncols=78, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    except ImportError:
        return iterable


def _bins_desde_features(feats):
    """
    Recupera el gap de oclusión en segundos desde la feature `gap_log`.

    El dataset guarda log1p(gap / GAP_SCALE); se invierte para poder
    estratificar la validación por régimen sin arrastrar tensores extra.
    """
    from ghosting.models.dataset import GAP_SCALE, FEATURES
    return torch.expm1(feats[..., FEATURES.index("gap_log")]) * GAP_SCALE


def run_epoch(model, loader, opt, fps, train: bool, desc="", bar=True,
              accumulate: int = 1):
    """
    Una pasada completa.

    Devuelve (componentes de la pérdida, dict de errores por régimen).

    El error se acumula sobre TODAS las observaciones y se toma la mediana al
    final. Calcular la mediana de cada lote y luego la mediana de esas medianas
    sería un estimador distinto y sesgado, porque los lotes tienen distinto
    número de pares ocultos.

    ESTRATIFICACIÓN EN VALIDACIÓN
    No basta con la mediana global. El objetivo del preset de horizonte largo
    es el bin >9.6 s, y una mejora global puede venir entera de las oclusiones
    cortas mientras la larga no se mueve — que es exactamente lo que pasó con
    la ventana de 10 s. Si el early stopping vigila solo el global, puede
    detenerse en un punto bueno en promedio y malo justo donde importa.
    """
    model.train(train)
    agg, n = {}, 0
    errs, gaps = [], []

    dev = next(model.parameters()).device
    it = _progress(loader, desc, len(loader), bar)
    for i, batch in enumerate(it):
        batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
        pred = model.predict_positions(
            batch["feats"], batch["player_mask"], batch["base"]
        )
        loss, parts = imputer_loss(
            pred, batch["target"], batch["base"], batch["loss_mask"], fps
        )
        if train:
            (loss / accumulate).backward()
            if (i + 1) % accumulate == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1

        with torch.no_grad():
            m = batch["loss_mask"]
            if m.any():
                e = torch.linalg.norm(
                    pred.detach() - (batch["base"] + batch["target"]), dim=-1
                )[m]
                errs.append(e)
                gaps.append(_bins_desde_features(batch["feats"])[m])

        if bar and hasattr(it, "set_postfix"):
            it.set_postfix(loss=f"{parts['total']:.2f}")

    out = {"global": float("nan"), "<=2s": float("nan"),
           "2-9.6s": float("nan"), ">9.6s": float("nan")}
    if errs:
        e = torch.cat(errs)
        g = torch.cat(gaps)
        out["global"] = float(e.median())
        for lab, lo, hi in [("<=2s", 0.0, 2.0), ("2-9.6s", 2.0, 9.6),
                            (">9.6s", 9.6, float("inf"))]:
            sel = (g > lo) & (g <= hi) if lo > 0 else (g >= lo) & (g <= hi)
            if sel.any():
                out[lab] = float(e[sel].median())
    return {k: v / max(n, 1) for k, v in agg.items()}, out


def guardar(path, model, args, best, base_err, hist, epoch, opt=None, sched=None):
    """
    Guarda el checkpoint. Se llama CADA VEZ que mejora la validación, no solo
    al final: un entrenamiento de horas que se interrumpe no debe perderse.
    """
    torch.save({
        "state_dict": model.state_dict(),
        "optim": opt.state_dict() if opt else None,
        "sched": sched.state_dict() if sched else None,
        "args": vars(args),
        "epoch": epoch,
        "val_err_m": best,
        "b4_val_err_m": base_err,
        "history": hist,
    }, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="sportec", choices=["sportec", "synthetic"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--stride", type=int, default=25,
                    help="solapamiento entre ventanas. 25 sobre 50 = 50%% de\n                          solape; bajarlo multiplica el coste sin aportar\n                          informacion nueva")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--width", type=float, default=44.0)
    ap.add_argument("--minutes", type=float, default=45.0)
    ap.add_argument("--bidirectional", action="store_true",
                    help="ver futuro dentro de la ventana. Es INTERPOLACIÓN, "
                         "mucho más fácil; nunca lo reportes como tiempo real")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0, help="0 = todos los nucleos")
    ap.add_argument("--resume", action="store_true",
                    help="continuar desde el ultimo checkpoint")
    ap.add_argument("--no-bar", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="preset rápido para una primera señal: ventanas cortas, "
                         "sin solape, modelo estrecho")
    ap.add_argument("--long", action="store_true",
                    help="preset de horizonte largo: ventana de 30 s en vez de 10 s. "
                         "Es el único preset capaz de atacar el bin >9.6 s, porque es "
                         "el único donde el jugador oculto llega a haber sido visible "
                         "dentro de la ventana")
    ap.add_argument("--accumulate", type=int, default=1,
                    help="acumular gradientes durante K lotes antes del step. "
                         "Simula un lote K veces mayor sin más memoria. Con 32 GB "
                         "de RAM y el preset --long no hace falta")
    ap.add_argument("--monitor", default="global",
                    choices=["global", "<=2s", "2-9.6s", ">9.6s"],
                    help="métrica de validación que dirige el early stopping y el "
                         "checkpoint. Usa '>9.6s' si el objetivo del experimento es "
                         "el régimen de oclusión larga")
    ap.add_argument("--train-ids", nargs="+", default=None,
                    help="sobrescribe la partición por defecto (para validación cruzada)")
    ap.add_argument("--val-id", default=None)
    ap.add_argument("--test-id", default=None)
    ap.add_argument("--ckpt", default=None, help="ruta del checkpoint")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cuda para GPU. En GPU sube tambien --batch: la\n                          memoria sobra y el paralelismo se aprovecha")
    ap.add_argument("--dry-run", type=float, default=0.0, metavar="SEGUNDOS",
                    help="mide el ritmo REAL durante N segundos y extrapola el "
                         "coste total, sin entrenar. Úsalo siempre antes de dejar "
                         "una corrida larga: una laptop con throttling térmico va "
                         "mucho más lenta en la hora 3 que en el minuto 1")
    ap.add_argument("--scheduler", default="plateau", choices=["plateau", "cosine"],
                    help="plateau (por defecto) reduce la tasa cuando la validación "
                         "se estanca. cosine anela según --epochs, así que si el "
                         "early stopping corta antes, la tasa nunca llega a bajar y "
                         "el modelo no afina")
    args = ap.parse_args()

    torch.set_num_threads(args.threads or (os.cpu_count() or 4))
    if args.quick:
        args.window, args.stride, args.dim, args.blocks = 40, 40, 96, 3
    if args.long:
        # 150 frames a 5 fps = 30 s. Cubre gaps de hasta 30 s, es decir buena
        # parte del bin >9.6 s. El lote baja para que quepa en memoria; el
        # coste por segundo de vídeo cubierto sube ~1.6x, no más.
        args.window, args.stride, args.batch = 150, 75, 10
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg_w = WindowConfig(length=args.window, stride=args.stride)
    if args.train_ids:
        splits = {"train": args.train_ids,
                  "val": [args.val_id] if args.val_id else [],
                  "test": [args.test_id] if args.test_id else []}
    else:
        splits = (SPLITS if args.provider == "sportec"
                  else {"train": ["SYNTH01"], "val": ["SYNTH02"],
                        "test": ["SYNTH03"]})

    data = {}
    for name, ids in splits.items():
        print(f"\n[{name}]")
        data[name] = load_split(ids, args, cfg_w)
    if not data["train"]:
        print("\nSin datos de entrenamiento. Corre antes scripts/00_download.py")
        return 1

    loaders = {
        k: DataLoader(Windows(v), batch_size=args.batch, shuffle=(k == "train"))
        for k, v in data.items() if v
    }

    dev = torch.device(args.device)
    model = ResidualImputer(dim=args.dim, n_blocks=args.blocks,
                            causal=not args.bidirectional).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=3
        )
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    if args.dry_run:
        print(f"\n[dry-run] midiendo ritmo real durante {args.dry_run:.0f} s...\n")
        model.train(True)
        n_pasos, t0 = 0, time.time()
        tiempos = []
        for batch in loaders["train"]:
            t1 = time.time()
            pred = model.predict_positions(batch["feats"], batch["player_mask"],
                                           batch["base"])
            loss, _ = imputer_loss(pred, batch["target"], batch["base"],
                                   batch["loss_mask"], args.fps)
            loss.backward()
            opt.step(); opt.zero_grad()
            tiempos.append(time.time() - t1)
            n_pasos += 1
            if time.time() - t0 > args.dry_run:
                break
        if not tiempos:
            print("no se completó ningún paso"); return 1
        # Se usa la mediana de la SEGUNDA MITAD: los primeros pasos van a turbo
        # y no representan el ritmo sostenido.
        est = float(np.median(tiempos[len(tiempos) // 2:]))
        pasos_ep = len(loaders["train"]) + len(loaders["val"])
        seg_ep = est * pasos_ep
        print(f"  pasos medidos      : {n_pasos}")
        print(f"  primer paso        : {tiempos[0]:.2f} s")
        print(f"  ritmo sostenido    : {est:.2f} s/paso")
        print(f"  pasos por época    : {pasos_ep}")
        print(f"  ESTIMADO POR ÉPOCA : {seg_ep / 60:.1f} min")
        print(f"  ESTIMADO {args.epochs} ÉPOCAS : {seg_ep * args.epochs / 3600:.1f} horas")
        if seg_ep * args.epochs / 3600 > 8:
            print("\n  *** No cabe en una noche. Reduce --epochs, usa --quick,")
            print("      o exporta el paquete y entrena en GPU:")
            print("      python scripts/06_export_bundle.py")
        return 0

    modo = "BIDIRECCIONAL (interpolación)" if args.bidirectional else "CAUSAL (online)"
    print(f"\nmodelo: {model.n_params:,} parámetros | modo {modo}")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"hilos CPU: {torch.get_num_threads()} | ventanas train {len(data['train']):,}\n")

    ckpt = (Path(args.ckpt) if args.ckpt else
            OUT / f"imputer_{'bidir' if args.bidirectional else 'causal'}.pt")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if ckpt.exists() and not args.resume:
        # Un checkpoint validado es un activo. Cambiar el número de features o
        # la ventana lo vuelve incargable, así que se archiva antes de pisarlo.
        prev = ckpt.with_suffix(f".prev{int(time.time())}.pt")
        ckpt.rename(prev)
        print(f"checkpoint anterior archivado en {prev.name}")
    ep0, hist_prev = 1, []
    if args.resume and ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(st["state_dict"])
        if st.get("optim"):
            opt.load_state_dict(st["optim"])
        if st.get("sched"):
            sched.load_state_dict(st["sched"])
        ep0, hist_prev = st["epoch"] + 1, st.get("history", [])
        print(f"reanudando desde la época {st['epoch']} "
              f"(err val {st['val_err_m']:.2f} m)")

    # Error de partida de B4, por régimen. No hace falta modelo ni pasada hacia
    # adelante: el modelo residual con la cabeza a cero predice exactamente
    # `base`, así que el error de B4 se lee directo de los datos.
    from ghosting.models.dataset import GAP_SCALE, FEATURES as _F
    _e, _g = [], []
    for w in data["val"]:
        m = w["loss_mask"]
        if not m.any():
            continue
        _e.append(np.linalg.norm(w["target"], axis=-1)[m])
        _g.append((np.expm1(w["feats"][..., _F.index("gap_log")]) * GAP_SCALE)[m])
    _e, _g = np.concatenate(_e), np.concatenate(_g)
    base_bins = {"global": float(np.median(_e))}
    for lab, lo, hi in [("<=2s", 0.0, 2.0), ("2-9.6s", 2.0, 9.6),
                        (">9.6s", 9.6, np.inf)]:
        sel = (_g > lo) & (_g <= hi) if lo > 0 else (_g >= lo) & (_g <= hi)
        base_bins[lab] = float(np.median(_e[sel])) if sel.any() else float("nan")
    base_err = base_bins[args.monitor]

    print("error mediano de B4 en validación (el piso a batir):")
    for k, v in base_bins.items():
        marca = "  <- métrica vigilada" if k == args.monitor else ""
        print(f"    {k:<8}{v:>8.2f} m{marca}")
    print()

    hdr = (f"{'época':>6}{'train':>9}{'global':>9}{'≤2s':>8}{'2-9.6s':>9}"
           f"{'>9.6s':>8}{'vs B4':>9}{'lr':>9}{'seg':>7}{'ETA':>8}")
    print(hdr); print("-" * len(hdr))

    best = min([h["val_err_m"] for h in hist_prev], default=float("inf"))
    best_state, sin_mejora, hist = None, 0, list(hist_prev)
    t_ini = time.time()
    for ep in range(ep0, args.epochs + 1):
        t0 = time.time()
        tr, _ = run_epoch(model, loaders["train"], opt, args.fps, True,
                          f"ep {ep} train", not args.no_bar, args.accumulate)
        with torch.no_grad():
            va, bins = run_epoch(model, loaders["val"], None, args.fps, False,
                                 f"ep {ep} val", not args.no_bar)
        err = bins[args.monitor]
        if args.scheduler == "plateau":
            sched.step(err)
        else:
            sched.step()
        dt = time.time() - t0
        delta = base_err - err
        eta = (args.epochs - ep) * dt
        lr = opt.param_groups[0]["lr"]
        print(f"{ep:>6}{tr['total']:>9.3f}{bins['global']:>9.2f}"
              f"{bins['<=2s']:>8.2f}{bins['2-9.6s']:>9.2f}{bins['>9.6s']:>8.2f}"
              f"{delta:>+9.2f}{lr:>9.1e}{dt:>7.1f}{f'{eta/60:.0f}m':>8}")
        hist.append({"epoch": ep, "train": tr["total"], "val_loss": va["total"],
                     "val_err_m": err, "delta_vs_b4": delta, "lr": lr, **bins})

        if err < best - 1e-4:
            best, sin_mejora = err, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            # Guardar YA: un entrenamiento largo interrumpido no debe perderse.
            guardar(ckpt, model, args, best, base_err, hist, ep, opt, sched)
        else:
            sin_mejora += 1
            if sin_mejora >= args.patience:
                print(f"\nearly stopping en la época {ep}")
                break

    if best_state:
        model.load_state_dict(best_state)

    print(f"\nmejor error en validación [{args.monitor}]: {best:.2f} m "
          f"(B4: {base_err:.2f} m, delta {base_err - best:+.2f} m)")

    OUT.mkdir(parents=True, exist_ok=True)
    guardar(ckpt, model, args, best, base_err, hist, args.epochs)
    (OUT / "tables").mkdir(exist_ok=True)
    with open(OUT / "tables" / "train_history.json", "w") as f:
        json.dump(hist, f, indent=2)
    print(f"-> {ckpt}")
    print("\nSiguiente: python scripts/05_evaluate_model.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
