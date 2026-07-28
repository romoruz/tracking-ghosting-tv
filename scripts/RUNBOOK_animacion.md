# Animacion "ghosting en accion" — runbook (v2, limpia y por modos)

Un solo archivo: `scripts/10_animate_ghosting.py`. No toca nada del repo.

## Instalar
```bash
tar xzf ghosting_animacion.tar.gz     # deja scripts/10_animate_ghosting.py
source venv/bin/activate               # torch CPU (setup.sh --torch) + ffmpeg
```
Necesita ya presentes: `data/processed/*J03WR9*.npz` y `reports/cv/fold_J03WR9.pt`.

## Tres modos, tres carpetas
```bash
python scripts/10_animate_ghosting.py --match J03WR9 --mode attack    # -> reports/video/ataque/
python scripts/10_animate_ghosting.py --match J03WR9 --mode defense   # -> reports/video/defensa/
python scripts/10_animate_ghosting.py --match J03WR9 --mode full --slow 1.6  # -> reports/video/completo/
```
- **attack**: fantasmas SOLO del equipo en posesion (los que se estiran).
- **defense**: fantasmas SOLO del equipo sin balon (la linea que se repliega).
- **full**: ambos equipos, en camara lenta.

Revisar un frame antes de animar: agrega `--preview` (saca un PNG).

## Que cambio respecto a la v1 (por que ya no se ve amontonado)
- Color por equipo en TODO: rombos, lineas de error y texto (azul local / rojo visitante).
- B4 (baseline) muy tenue en gris; en attack/defense solo se dibuja para el PEOR jugador.
- Transparencia de la linea proporcional al error: casi perfecto -> invisible; error grande -> opaco.
- Etiqueta de metros solo cuando el error es apreciable (>=6 m).
- Rombos semitransparentes para no tapar a los reales.
- Camara lenta por DILATACION TEMPORAL (--slow), no interpolacion: los saltos de 5 Hz se conservan.

## Portero
Se atenua visualmente (mas transparente, sin etiqueta) pero NO se falsea a "visible".
Forzarlo rompe la fisica del broadcast y la comparabilidad con Choi (ver 03_modelo_de_camara.md).

## Banderas utiles
- `--clip-seconds N`   duracion del tramo de juego (def. 12)
- `--slow F`           camara lenta (2.0 = mitad de velocidad); honesta, no interpola
- `--start-s S`        forzar inicio en el segundo S de partido
- `--fps-video N`      fps de salida (def. 25)

## Reproduce el paper
global 7.93 -> 5.60 m ; bin >9.6 s 10.85 -> 8.71 m. Todo causal, held-out (fold sin J03WR9).

## Nota: bug de 05_evaluate_model.py
Su reensamblado usa `feats[..., -1]` (d_last) en vez del indice de `on_pitch` (14) para
elegir jugadores; scatterea predicciones a columnas equivocadas y el modelo parece peor
que B4. Este script usa el indice correcto. Conviene arreglarlo tambien alla.
