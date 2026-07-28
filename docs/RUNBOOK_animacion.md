# Animación "ghosting en acción" — runbook

Un solo archivo nuevo: `scripts/10_animate_ghosting.py`. No toca nada del
proyecto existente. Sigue el principio del repo: un script, un artefacto.

## Instalar (dentro de tu carpeta `ghosting/`)

```bash
tar xzf ghosting_animacion.tar.gz     # deja scripts/10_animate_ghosting.py en su sitio
source venv/bin/activate               # tu venv de siempre
# requiere el build de torch CPU (setup.sh --torch) y ffmpeg para mp4:
#   sudo pacman -S ffmpeg
```

Necesita, ya presentes en tu repo:
- `data/processed/*J03WR9*.npz`         (el partido)
- `reports/cv/fold_J03WR9.pt`           (el checkpoint de ese fold)

## Correr

```bash
# 1) revisar un frame estático antes de animar (segundos)
python scripts/10_animate_ghosting.py --match J03WR9 --preview

# 2) el clip completo (elige solo el mejor segmento; ~2 min en CPU)
python scripts/10_animate_ghosting.py --match J03WR9 --clip-seconds 12

# 3) forzar un inicio concreto (segundos de partido) si quieres otra jugada
python scripts/10_animate_ghosting.py --match J03WR9 --start-s 1756 --clip-seconds 12
```

Salidas en `reports/figures/`: `ghosting_J03WR9.mp4` y `ghosting_preview_J03WR9.png`.
Si no hay ffmpeg, exporta `.gif` automáticamente.

## Qué se ve

- Banda amarilla = viewport (lo que la cámara muestra). Paneá siguiendo el balón.
- Círculo lleno = jugador visible. Círculo hueco = jugador oculto, posición REAL.
- Rombo gris hueco = fantasma de B4 (heurística). Rombo naranja = fantasma del modelo.
- Línea de color = error del modelo a la posición real, con su valor en metros.
  Color por tiempo oculto: verde ≤2 s, ámbar 2–9.6 s, rojo >9.6 s (el régimen abierto).
- Cabecera = mediana del error acumulada en el clip, B4 → modelo.
- Franja inferior = cómo evoluciona esa mediana a medida que corre el vídeo.

## Decisiones que importan (y que un revisor preguntará)

- **Causal**: cada frame usa solo el pasado. Nada bidireccional entra aquí.
- **5 fps de modelo, 25 fps de vídeo por HOLD** (cada predicción se sostiene 5
  frames). No se interpolan las posiciones del fantasma: se mostrarían puntos
  que el modelo nunca predijo.
- **Segmento elegido por criterio**, no a mano: la ventana con más oclusión
  larga y alguna reaparición limpia. Ahí es donde el problema se ve.
- **In-domain, held-out**: `fold_J03WR9.pt` se entrenó sin J03WR9. Evaluar sobre
  J03WR9 es honesto.
- Reproduce la tabla del paper: global 7.93 → 5.60 m, bin >9.6 s 10.85 → 8.71 m.

## Bug corregido de `05_evaluate_model.py`

La reconstrucción por ventanas de aquel script recompone la predicción usando
`feats[..., -1]` (índice 17 = `d_last`) para decidir qué jugadores conserva,
pero `pad_players` los seleccionó con el índice de `on_pitch` (14). Cuando
difieren, las predicciones se escriben en columnas de jugador equivocadas y el
modelo aparece PEOR que B4. Este script usa el índice correcto. Conviene
arreglarlo también en `05_evaluate_model.py` (línea del `on = per["feats"][s:e, :, -1]...`).
