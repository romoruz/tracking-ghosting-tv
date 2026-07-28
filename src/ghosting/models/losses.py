"""
Función de pérdida del imputador residual.

    L = L_rec + λ_s·L_smooth + λ_v·L_vel + λ_a·L_acc

Por qué hacen falta las dos familias de términos
------------------------------------------------
Las bisagras cinemáticas (L_vel, L_acc) tienen **gradiente nulo** en la región
factible: no penalizan nada mientras el modelo se mantenga por debajo de
v_max. Eso deja pasar oscilación de alta frecuencia — un jugador que tiembla a
3 m/s frame a frame es físicamente absurdo pero no viola ninguna cota. Los
transformers producen exactamente ese artefacto.

L_smooth (segunda derivada discreta) lo castiga en TODO el dominio; las
bisagras ponen el techo duro. Se necesitan ambas.

Nota de nomenclatura: esto no es una "PINN". No hay ecuación diferencial que
gobierne el sistema. Son penalizaciones de factibilidad física, y conviene
llamarlas así en vez de inflar el vocabulario.

Calibración de los pesos
------------------------
Los términos crudos tienen magnitudes muy dispares. Evaluados sobre la salida
de B4 (es decir, con residuo cero), sobre datos sintéticos:

    rec = 13.5   smooth = 0.35   vel = 0.51   acc = 86.0

El término de aceleración es ~6x el de reconstrucción **antes de que el modelo
haga nada**: está midiendo los saltos del propio B4, que salta cada vez que un
jugador reaparece y su estimación se engancha a la posición observada. Con
pesos iguales, el modelo dedicaría casi todo su esfuerzo a suavizar las
discontinuidades del baseline en lugar de a acertar posiciones.

Los pesos por defecto están calibrados para que cada término físico aporte
~3-5% de la reconstrucción en la inicialización. Si cambias el baseline o la
escala de las features, recalíbralos: no son constantes universales.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .dataset import V_MAX, A_MAX


def imputer_loss(
    pred_pos: torch.Tensor,
    true_res: torch.Tensor,
    base: torch.Tensor,
    loss_mask: torch.Tensor,
    fps: float,
    lambda_smooth: float = 2.0,
    lambda_vel: float = 1.3,
    lambda_acc: float = 0.005,
    huber_delta: float = 5.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Parameters
    ----------
    pred_pos : (B,T,N,2)  posición predicha (= base + residuo)
    true_res : (B,T,N,2)  residuo verdadero (verdad - base)
    base     : (B,T,N,2)  estimación de B4
    loss_mask: (B,T,N)    dónde hay verdad con la que comparar
    fps      : frecuencia de muestreo, para derivar velocidad y aceleración

    Returns
    -------
    (pérdida total, dict de componentes para el log)
    """
    m = loss_mask
    n = m.sum().clamp(min=1)


    # Huber y no MSE: la distribución del error tiene cola pesada (un jugador
    # oculto 3 minutos puede estar a 40 m). Con MSE, esos pocos casos dominan
    # el gradiente y el modelo deja de aprender el caso típico.
    true_pos = base + true_res
    err = torch.linalg.norm(pred_pos - true_pos, dim=-1)
    rec = (F.huber_loss(err, torch.zeros_like(err), reduction="none",
                        delta=huber_delta) * m).sum() / n


    if pred_pos.shape[1] >= 3:
        d2 = pred_pos[:, 2:] - 2 * pred_pos[:, 1:-1] + pred_pos[:, :-2]
        mm = (m[:, 2:] & m[:, 1:-1] & m[:, :-2]).float()
        smooth = ((d2 ** 2).sum(-1) * mm).sum() / mm.sum().clamp(min=1)
    else:
        smooth = pred_pos.new_zeros(())


    if pred_pos.shape[1] >= 2:
        v = (pred_pos[:, 1:] - pred_pos[:, :-1]) * fps
        sp = torch.linalg.norm(v, dim=-1)
        mv = (m[:, 1:] & m[:, :-1]).float()
        vel = ((F.relu(sp - V_MAX) ** 2) * mv).sum() / mv.sum().clamp(min=1)
    else:
        v, vel = None, pred_pos.new_zeros(())

    if v is not None and v.shape[1] >= 2:
        a = (v[:, 1:] - v[:, :-1]) * fps
        ac = torch.linalg.norm(a, dim=-1)
        ma = (m[:, 2:] & m[:, 1:-1] & m[:, :-2]).float()
        acc = ((F.relu(ac - A_MAX) ** 2) * ma).sum() / ma.sum().clamp(min=1)
    else:
        acc = pred_pos.new_zeros(())

    total = rec + lambda_smooth * smooth + lambda_vel * vel + lambda_acc * acc
    return total, {
        "total": float(total.detach()),
        "rec": float(rec.detach()),
        "smooth": float(smooth.detach()),
        "vel": float(vel.detach()),
        "acc": float(acc.detach()),
    }


@torch.no_grad()
def median_error_m(pred_pos, true_res, base, loss_mask) -> float:
    """Mediana del error de posición en metros. La métrica que de verdad importa."""
    m = loss_mask
    if m.sum() == 0:
        return float("nan")
    err = torch.linalg.norm(pred_pos - (base + true_res), dim=-1)[m]
    return float(err.median())
