# Roadmap

## Estado

| Paso | Contenido | Estado | Hardware |
|---|---|---|---|
| 0 | Esquema canónico + cargadores | ✅ | CPU |
| 1 | Simulador de cámara + reporte | ✅ | CPU |
| 2 | Escalera B0–B4 + métricas con IC | ✅ | CPU |
| 3 | Modelo aprendido residual | ⬜ | CPU o GPU gratuita |
| 4 | Modelo generativo | ⬜ | GPU |
| 5 | Pipeline de visión sobre video | ⬜ | GPU |

## Siguientes tres tareas, en orden

### A. Correr sobre datos reales

```bash
./run.sh sportec     # 7 partidos de Bundesliga
./run.sh metrica     # 3 partidos, comparables con Choi
```

Primero Metrica: si tus B0–B4 reproducen la tabla del paper, el pipeline está
correcto. Solo entonces pasa a Sportec.

**Entregable:** tabla de la escalera sobre 10 partidos, dos proveedores, dos
ligas. Ya es un resultado: nadie ha verificado si los hallazgos de Choi
generalizan fuera de Metrica.

### B. Pitch control y métricas de decisión

Implementar Spearman (2017) sobre grilla de 3 m y añadir el MAE de la zona
oculta y el error de control-share.

**Entregable:** la figura de tres paneles — control real, control solo con
visibles, control con fantasmas. Es la imagen que se entiende sin matemáticas y
la que va en la presentación.

### C. Calibración del viewport con SkillCorner

Ajustar zoom y tilt contra estadísticas de broadcast real. Ver
`docs/03_modelo_de_camara.md`, sección "Calibración".

**Entregable:** benchmark en dos versiones, la del paper y la calibrada. Es la
contribución metodológica propia.

## Cómo lanzar una corrida nocturna

```bash
python scripts/04_train.py --long --epochs 100 --monitor ">9.6s" --patience 12
```

Cuatro decisiones, y la segunda es la que suele olvidarse:

- **`--epochs 100` en vez de 40.** El límite lo pone el early stopping, no el
  contador. Fijar 40 solo garantiza cortar antes de tiempo si el modelo aún
  estaba aprendiendo.
- **`--monitor ">9.6s"`.** El checkpoint y el early stopping vigilan la métrica
  que es objetivo del experimento. Con `global`, una mejora que venga entera de
  las oclusiones cortas detendría la corrida en un punto bueno en promedio y
  malo justo donde importa — que es exactamente lo que ocurrió con la ventana
  de 10 s.
- **`--scheduler plateau`** (por defecto). El coseno anela según `--epochs`; si
  pides 100 y el early stopping corta en 30, la tasa nunca llega a bajar y el
  modelo no afina. `ReduceLROnPlateau` reacciona al progreso real.
- **`--patience 12`** en vez de 8: con ventanas largas hay menos pasos por
  época, así que la mejora llega más despacio en unidades de época.

La consola imprime las cuatro métricas por época, no solo la global, así que la
lectura de la mañana es inmediata.

## Paso 3: el modelo aprendido

### Presupuesto de cómputo

7 partidos × 90 min × 5 fps × 22 jugadores ≈ **4.2 M** muestras jugador-frame.
Ventanas de 10 s a 5 fps = 50 frames. Modelo con `hidden_dim=128`: unos pocos
millones de parámetros.

Eso entrena en **1–2 h en GPU gratuita** (Colab o Kaggle, que da ~30 h
semanales), o en unas 8 h en el i7-1165G7. No hace falta comprar nada.

Instala PyTorch CPU con `./setup.sh --torch`. La laptop tiene gráficos Intel
Iris Xe: no hay CUDA, y el build CPU evita descargar ~2 GB de librerías de
NVIDIA que no se usarían.

### Arquitectura

Ver `docs/02_formalismo_matematico.md` §3. Resumen:

- Predice el **residuo sobre B4**, no la posición absoluta.
- Entradas absolutas **y** relativas (balón y centroide).
- Bloques alternados de atención espacial (sobre los 22, equivariante) y
  temporal.
- Pérdida: reconstrucción + suavidad + bisagras cinemáticas.
- Causal por defecto; variante bidireccional etiquetada aparte.

### Criterio de éxito

Bajar el bin de **>9.6 s** por debajo de los 15.6–16.9 m que reporta B4, con
intervalos de confianza que no se solapen. Cualquier otra mejora es secundaria.

### Riesgo

Con 7 partidos, es perfectamente posible que el modelo aprendido **no le gane**
a B4. Es un resultado legítimo y hay que reportarlo como tal: *"con datos
abiertos, las heurísticas cierran la mayor parte del margen; superarlas
requiere más datos"*. Eso, además, es exactamente el argumento para pedirle
datos a un club.

## Contribución

Sobre el trabajo previo, este repositorio aporta:

1. **Réplica independiente** de la escalera de Choi (2026), con código propio.
2. **Generalización a otro proveedor y otra liga**: Sportec/DFL (TRACAB gen-5,
   Bundesliga) además de Metrica. El paper lista como limitación explícita usar
   tres partidos de un solo proveedor.
3. **Partición del portero**: anclaje a portería en vez de al centroide del
   equipo, con reporte separado. La escalera original trata a los 22 jugadores
   de forma homogénea, lo que introduce un error sistemático en la posición del
   portero durante fases de ataque.
4. **Modelo de cámara calibrado empíricamente** contra broadcast real
   (SkillCorner), en vez de un viewport rectangular supuesto. El paper señala
   la ausencia de zoom y tilt como limitación.
5. **Modelo aprendido para oclusiones largas** (paso 3), el régimen que el
   paper identifica como abierto.

Los puntos 1–3 están hechos o son cuestión de correr los scripts. El 4 es una
semana. El 5 es el trabajo real.

## Lo que NO está en este repositorio

- **Pipeline de visión sobre video** (detección, re-ID, calibración de cancha).
  Es un proyecto aparte, con su propia literatura (SoccerNet-GSR, PnLCalib,
  BoT-SORT). Este repositorio asume tracking ya extraído.
- **Datos propios de ningún club.** Todo es abierto y redistribuible bajo las
  licencias de cada proveedor.
