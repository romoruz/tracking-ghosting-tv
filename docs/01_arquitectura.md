# Arquitectura del proyecto

## Principio rector

**Cada script escribe un artefacto en disco y ninguno depende del estado de un
notebook.** Los notebooks sirven para explorar; los números que van a un
reporte o a una presentación salen siempre de `scripts/`, con semilla fija y
versiones congeladas. Si un resultado no se puede regenerar con un comando, no
es un resultado.

## Flujo de datos

```
                    kloppy
   Sportec / Metrica ──────┐
                           ├──► [00_download.py] ──► data/processed/*.npz
   generador sintético ────┘                              (esquema canónico)
                                                                 │
                                                                 ▼
                                                    [camera/viewport.py]
                                                     máscara M[t,i] + gaps
                                                                 │
                          ┌──────────────────────────────────────┤
                          ▼                                      ▼
             [01_viewport_report.py]                  [baselines/ladder.py]
              validación del simulador                     B0 … B4
                          │                                      │
                          ▼                                      ▼
              reports/tables/                          [metrics/position.py]
              viewport_stats.csv                    error estratificado + IC
              figures/occlusion_*.png                          │
                                                                ▼
                                                      reports/tables/ladder.csv
                                                      figures/ladder_*.png
                                                      figures/ghosts_*.png
```

## Estructura

```
ghosting/
├── setup.sh                   instala venv y dependencias
├── run.sh                     orquestador (demo | sportec | metrica | test)
├── requirements.txt
├── pyproject.toml
│
├── src/ghosting/
│   ├── io/
│   │   ├── schema.py          Match: la estructura de datos central
│   │   └── loaders.py         proveedor -> esquema canónico (única dep. de kloppy)
│   ├── camera/
│   │   └── viewport.py        simulador de cámara, máscara M, gaps de oclusión
│   ├── baselines/
│   │   └── ladder.py          B0–B5, causales, sin entrenamiento
│   ├── metrics/
│   │   └── position.py        error estratificado + block bootstrap
│   └── viz/
│       └── pitch.py           cancha, fantasmas, histogramas, comparativas
│
├── scripts/
│   ├── 00_download.py         paso 0
│   ├── 01_viewport_report.py  paso 1
│   └── 02_run_baselines.py    paso 2
│
├── tests/test_pipeline.py     44 tests de invariantes
├── docs/
├── data/{raw,processed}/      .npz generados (gitignored)
└── reports/{tables,figures}/  salidas (gitignored)
```

## Decisiones de diseño

### El esquema canónico aísla al proveedor

Toda dependencia de kloppy vive en `io/loaders.py`. El resto del proyecto no
importa kloppy en ninguna parte. Consecuencia práctica: **añadir los datos
propios del club es escribir una función en ese archivo y nada más.** Ni el
simulador, ni los baselines, ni las métricas, ni las figuras cambian.

### $N$ no es 22

Con sustituciones aparecen 28–32 jugadores distintos a lo largo de un partido.
El esquema guarda a todos y añade una máscara `on_pitch[t, i]` que indica
quién está físicamente en cancha en cada frame. Eso resuelve limpiamente
sustituciones, expulsiones y el descanso.

Fijar $N=22$ obligaría a decidir arbitrariamente qué jugador ocupa cada
"casilla" tras un cambio, y ese es un bug que produce números plausibles pero
falsos.

### Las coordenadas se normalizan, nunca se asumen

Cada proveedor usa un sistema distinto y **no son intercambiables**:

| Proveedor | rango de x | Tipo |
|---|---|---|
| Sportec | −52.5 … +52.5 | métrico, origen al centro |
| Metrica | 0 … 1 | normalizado |
| Tracab | −5250 … +5250 | centímetros, origen al centro |

Asumir uno solo produce coordenadas catastróficamente equivocadas que **aun así
parecen números de cancha**, y ninguna validación aguas abajo lo detecta.
`_pitch_and_scalers` lee el rango de
`dataset.metadata.coordinate_system.pitch_dimensions` y normaliza
explícitamente. Cubierto por `test_proveedores_convergen_al_mismo_esquema`.

También se fuerza orientación estática: si fuera `BALL_OWNING_TEAM`, las
coordenadas se voltearían en cada cambio de posesión y el EMA de paneo
perseguiría un fantasma.

### La identidad del portero se infiere por periodo

Si el proveedor no marca porteros (Metrica no lo hace), se infieren por
geometría. El criterio ingenuo —el jugador cuya x media está más cerca de una
línea de meta— **falla siempre**: los equipos cambian de lado en el descanso,
así que un portero real promedia x ≈ 52.5 m, el centro de la cancha, que es el
valor menos parecido a un portero que existe.

El fallo es silencioso. `validate()` pasa porque hay dos "porteros", las
métricas corren, y los porteros reales quedan evaluados como jugadores de campo
con ancla de centroide. Si el jugador mal elegido resulta ser un suplente que no
jugó el tramo analizado, la bandera `--include-gk` no cambia ni una muestra y
el bug se disfraza de robustez.

El criterio correcto usa la **mediana por periodo** de la distancia a la meta
más cercana, y toma el mínimo entre periodos. Mediana y no media: es robusta a
las salidas del portero y a los córners a favor. Cubierto por
`test_inferencia_de_portero_con_cambio_de_lado` y
`test_include_gk_cambia_la_poblacion`.

> `is_gk` se guarda dentro del `.npz`. Si cambias el cargador, hay que
> regenerar los datos con `--force`; si no, sigues usando la inferencia vieja.

### `on_pitch` no es lo mismo que `visible`

Son dos máscaras distintas y confundirlas invalida todo:

- `on_pitch[t,i]` — verdad de existencia: el jugador está en el campo.
- `visible[t,i]` — verdad de observación: la cámara lo muestra.

Un jugador puede estar en cancha y no ser visible (ese es el problema entero).
No puede ser visible sin estar en cancha.

### Reescalado de constantes de tiempo

El paper define $\alpha=0.06$ para el paneo **a 25 fps** y el peso del EMA de
offsets **a 5 fps**. Evaluar a otra frecuencia sin reescalar cambiaría la
inercia física de la cámara y del suavizado.

Ambos módulos reescalan resolviendo $(1-a_{\text{fps}})^{\text{fps}} = (1-a_{\text{ref}})^{\text{ref}}$.
Está cubierto por `test_alpha_preserva_constante_de_tiempo`.

### Decimar, no interpolar

`Match.resample()` submuestrea por decimación entera. Promediar frames
inventaría posiciones que nadie ocupó nunca y contaminaría la verdad de
terreno.

### Baselines iterativos, no vectorizados

`run_ladder` recorre frame por frame. Es deliberado: los métodos son
recursivos (el offset en $t$ depende del centroide votado en $t$, que depende
de los offsets en $t-1$) y la claridad del código importa más que la velocidad.
Un partido de 90 min a 5 fps son 27k frames: menos de un segundo por método.

## Partición de datos

**Por partido, nunca por frame ni por ventana.**

A 25 fps los frames consecutivos son casi idénticos. Partir por frame pondría
observaciones casi iguales en entrenamiento y en test, y el modelo parecería
excelente sin haber generalizado nada.

| Conjunto | Partidos | Uso |
|---|---|---|
| Entrenamiento | 5 de Sportec | ajuste de parámetros |
| Validación | 1 de Sportec | selección de modelo |
| Test interno | 1 de Sportec | evaluación final |
| Test externo | 3 de Metrica | comparabilidad con Choi (2026) |

El test externo se **congela** hasta el final. Es lo que hace que los números
sean creíbles frente a terceros.

## Reproducibilidad

- Todas las semillas fijadas (`default_rng(42)`, `seed` derivada del `match_id`).
- Backend de matplotlib `Agg`: funciona por SSH y sin pantalla.
- Sin estado global: cada script se puede correr aislado.
- `./run.sh test` verifica 44 invariantes antes de confiar en cualquier cifra.
