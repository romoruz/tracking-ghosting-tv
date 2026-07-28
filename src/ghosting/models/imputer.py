"""
Imputador residual espacio-temporal.

Predice la corrección sobre B4, no la posición absoluta. Ver
`dataset.py` para la justificación.

Estructura
----------
Bloques alternados de atención espacial y temporal:

    entrada (B, T, N, F)
        │  proyección lineal a D
        ▼
    [ atención ESPACIAL  ] sobre los N jugadores, dentro de cada frame
    [ atención TEMPORAL  ] sobre los T frames, dentro de cada jugador
        │  x n_blocks
        ▼
    cabeza lineal -> (B, T, N, 2)  residuo en metros

Por qué alternar en vez de aplanar
----------------------------------
Aplanar (T·N) tokens y hacer atención completa sería O((TN)²) = O(50²·26²) por
ventana, y sobre todo perdería las dos simetrías del problema:

- **Equivarianza a permutaciones entre jugadores.** Los jugadores de un equipo
  son un CONJUNTO, no una lista. La atención espacial es equivariante por
  construcción: reordenar la entrada reordena la salida igual. Un MLP sobre el
  vector concatenado de 2N coordenadas rompería esto y el modelo memorizaría el
  orden del roster en vez de aprender estructura de juego.
- **Causalidad opcional en el eje temporal.** Separar los ejes permite aplicar
  una máscara causal solo en el temporal, y por tanto entrenar el mismo modelo
  en modo online (solo pasado) o bidireccional (pasado y futuro) cambiando una
  bandera.

Modos causal y bidireccional
----------------------------
`causal=True` -> el frame t solo atiende a t' <= t. Es lo que un club puede
usar en vivo, y es el régimen en el que está medido todo el benchmark.

`causal=False` -> atiende a toda la ventana. Es un problema de INTERPOLACIÓN,
sustancialmente más fácil, y es lo que hace el Graph Imputer de DeepMind con
ventanas de 9.6 s. Se implementa para poder reportar ambos, nunca para
presentar el número bidireccional como si fuera de tiempo real.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import N_FEATURES


class Attention(nn.Module):
    """
    Atención multi-cabeza sobre el último eje agrupado.

    Usa `F.scaled_dot_product_attention`, que en CPU es ~2.8x más rápida que
    calcular QK^T, softmax y AV a mano: PyTorch la despacha a un kernel fusionado
    que evita materializar la matriz de atención completa. Con las formas de
    este modelo eso son ~60 minutos por época frente a ~20.
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0, "dim debe ser divisible entre heads"
        self.h = heads
        self.dk = dim // heads
        self.p = dropout
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, key_padding_mask=None, causal=False):
        """
        x : (B, L, D)
        key_padding_mask : (B, L) bool -- True = posición válida
        """
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.h, self.dk).transpose(1, 2)
        k = k.view(B, L, self.h, self.dk).transpose(1, 2)
        v = v.view(B, L, self.h, self.dk).transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            # SDPA con máscara booleana: True = participa en la atención.
            attn_mask = key_padding_mask[:, None, None, :].expand(B, 1, L, L).clone()
            if causal:
                tri = torch.tril(
                    torch.ones(L, L, dtype=torch.bool, device=x.device)
                )
                attn_mask = attn_mask & tri[None, None]
            # Una fila completamente enmascarada produce NaN. Se garantiza la
            # diagonal: la salida de esas posiciones se descarta igualmente,
            # pero el NaN se propagaría por la conexión residual.
            eye = torch.eye(L, dtype=torch.bool, device=x.device)
            attn_mask = attn_mask | eye[None, None]

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.p if self.training else 0.0,
            is_causal=(causal and attn_mask is None),
        )
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(torch.nan_to_num(out, nan=0.0))


class Block(nn.Module):
    """Un bloque = atención espacial + atención temporal + feed-forward."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n1, self.spatial = nn.LayerNorm(dim), Attention(dim, heads, dropout)
        self.n2, self.temporal = nn.LayerNorm(dim), Attention(dim, heads, dropout)
        self.n3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x, player_mask, causal):
        """x : (B, T, N, D) | player_mask : (B, N) bool"""
        B, T, N, D = x.shape

        # Atención espacial
        h = self.n1(x).reshape(B * T, N, D)
        pm = player_mask[:, None, :].expand(B, T, N).reshape(B * T, N)
        x = x + self.spatial(h, key_padding_mask=pm).reshape(B, T, N, D)

        # Atención temporal
        h = self.n2(x).permute(0, 2, 1, 3).reshape(B * N, T, D)
        x = x + self.temporal(h, causal=causal).reshape(B, N, T, D).permute(0, 2, 1, 3)

        return x + self.ff(self.n3(x))


class ResidualImputer(nn.Module):
    """
    Modelo completo.

    Parameters
    ----------
    dim : int
        Anchura del modelo. 128 basta: con 7 partidos de entrenamiento, subir
        de ahí solo compra sobreajuste.
    n_blocks : int
        Bloques espacio-temporales.
    causal : bool
        Modo online (True) o bidireccional (False). Ver la nota del módulo.
    max_residual_m : float
        Cota dura sobre el residuo, en metros, aplicada con tanh. Impide que
        una salida disparatada mueva a un jugador a la tribuna y estabiliza el
        arranque, cuando los pesos aún son ruido.
    """

    def __init__(
        self,
        dim: int = 128,
        n_blocks: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        causal: bool = True,
        max_residual_m: float = 30.0,
    ):
        super().__init__()
        self.causal = causal
        self.max_residual_m = max_residual_m

        self.inp = nn.Sequential(nn.Linear(N_FEATURES, dim), nn.GELU())
        self.time_emb = nn.Parameter(torch.zeros(1, 512, 1, dim))
        nn.init.trunc_normal_(self.time_emb, std=0.02)

        self.blocks = nn.ModuleList(
            [Block(dim, heads, dropout) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2)
        # Arrancar en cero => el modelo empieza siendo exactamente B4 y solo
        # puede mejorar desde ahí. Con pocos datos esto importa mucho.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feats: torch.Tensor, player_mask: torch.Tensor) -> torch.Tensor:
        """
        feats : (B, T, N, F)
        player_mask : (B, N) bool
        returns : (B, T, N, 2) residuo en metros
        """
        B, T, N, _ = feats.shape
        x = self.inp(feats) + self.time_emb[:, :T]
        for blk in self.blocks:
            x = blk(x, player_mask, self.causal)
        r = self.head(self.norm(x))
        return torch.tanh(r) * self.max_residual_m

    def predict_positions(self, feats, player_mask, base) -> torch.Tensor:
        """Posición final = B4 + residuo."""
        return base + self(feats, player_mask)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
