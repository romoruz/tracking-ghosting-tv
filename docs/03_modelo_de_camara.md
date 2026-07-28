# Modelo de cámara

Si la máscara de oclusión simulada no se parece a la real, todos los números
del proyecto son ficción. Este es el cimiento estadístico y merece más
escrutinio que el modelo aprendido.

## Por qué no se puede enmascarar al azar

Una máscara aleatoria haría el problema artificialmente fácil y no
representativo. La cámara de televisión ocluye de forma **sistemática**:

- Sigue al balón, con retraso.
- Los jugadores lejos del balón desaparecen.
- Los defensores del lado lejano durante un ataque sostenido desaparecen
  durante **minutos**, no segundos.

Esa última propiedad es la que define el problema difícil, y una máscara
aleatoria uniforme nunca la produciría.

## Modelo base (réplica de Choi 2026)

Cámara que paneá horizontalmente siguiendo una versión suavizada del balón:

$$c(t)=\alpha\,x_b(t)+(1-\alpha)\,c(t-1),\qquad \alpha=0.06 \text{ por frame a 25 fps}$$

$$V_t=\Big\{\,i\;:\;\big|x_i(t)-c(t)\big|\le \tfrac{W}{2}\,\Big\},\qquad W=44\ \text{m}$$

El EMA modela el retraso del camarógrafo: la cámara nunca está exactamente
sobre el balón, va detrás. La región visible es una ventana vertical que abarca
todo el alto de la cancha.

Detalles de implementación:

- El centro se **reinicia en cada periodo**: entre tiempos la cámara no
  arrastra su posición previa.
- Los NaN del balón se puentean manteniendo el último centro válido: si el
  sistema perdió la referencia, la cámara no se mueve.
- El coeficiente se reescala automáticamente si el partido tiene otro fps, para
  preservar la constante de tiempo física del paneo.

## Validación

Con $W=44$ m deben verse **entre 14 y 16 jugadores de 22**. Referencias:

| Fuente | Visibles de 22 |
|---|---|
| Choi (2026), simulado sobre Metrica, $W=44$ | 14.6 – 15.0 |
| Omidshafiei et al. (2022), simulado sobre EPL | 12.8 ± 3.7 |
| Ochin et al. (2025), broadcast real | 9 – 12 |
| Choi (2026), clips reales de Mundial | 10 – 16 |

`scripts/01_viewport_report.py` compara automáticamente y marca desviaciones
mayores a 1.6 jugadores con `<-- revisar`.

Barrido de sensibilidad publicado, que debes reproducir:

| $W$ (m) | Visibles (Choi) |
|---|---|
| 36 | 13.0 – 13.1 |
| 44 | 14.6 – 15.0 |
| 52 | 16.1 – 16.5 |
| 60 | 17.2 – 17.7 |

## Gaps de oclusión

Para cada par (frame, jugador) se calcula $\Delta_i(t)$, el tiempo desde la
última observación dentro del periodo actual:

- $0$ si el jugador es visible.
- Positivo si está oculto pero fue visto antes en el mismo periodo.
- `NaN` si nunca fue visto (cold start) o si no está en cancha.

**Esta es la variable de estratificación de todo el proyecto.** Sin ella, el
promedio global mezcla el régimen trivial (2 s de oclusión) con el irresuelto
(30 s) y no dice nada útil.

Choi reporta que **50–57% de las muestras ocultas superan los 9.6 s**, que es
precisamente el límite de la ventana del Graph Imputer. Más de la mitad del
problema real cae fuera del alcance del estado del arte aprendido.

## Extensiones sobre el modelo base

Choi señala explícitamente como limitación que su viewport *"paneá pero no
hace zoom ni tilt; el zoom cambia el número de jugadores visibles, así que las
estadísticas de oclusión de un broadcast real pueden diferir de las
simuladas."*

Este repositorio implementa dos extensiones, **desactivadas por defecto** para
preservar la comparabilidad numérica:

### Zoom: $W(t)$ variable

```python
ViewportConfig(enable_zoom=True, zoom_min_m=32, zoom_max_m=60, zoom_speed_ref=12)
```

El ancho interpola linealmente con la rapidez del balón (suavizada a 1 s):
balón lento, plano abierto; balón rápido, plano cerrado. Es el comportamiento
del realizador de televisión.

### Tilt: banda vertical

```python
ViewportConfig(enable_tilt=True, height_m=52)
```

Añade la condición $|y_i(t)-y_b(t)|\le H/2$, ocluyendo a los jugadores de la
banda lejana.

## Calibración contra broadcast real

Las extensiones anteriores tienen parámetros libres. Suponerlos sería repetir
el problema que se pretende corregir. **SkillCorner opendata contiene la
respuesta**: son 10 partidos de tracking de broadcast genuino, donde la máscara
de oclusión no es simulada sino **observada**.

Procedimiento:

1. Cargar SkillCorner y extraer, por frame, qué jugadores fueron detectados.
2. Estimar empíricamente: distribución de $|V_t|$, distribución de duraciones
   de oclusión, y dependencia de la visibilidad respecto a la posición del
   balón.
3. Ajustar $(W_{\min}, W_{\max}, v_{\text{ref}}, H, \alpha)$ minimizando la
   divergencia entre las estadísticas simuladas sobre Sportec y las observadas
   en SkillCorner.
4. Re-correr todo el benchmark con el viewport calibrado y **reportar ambas
   versiones**: la del paper (comparable) y la calibrada (más realista).

Esto es una contribución metodológica pequeña pero legítima, y es barata: no
requiere GPU ni datos privados.

> Advertencia: SkillCorner no es verdad de terreno. Ya tiene los huecos y no
> contiene las posiciones de los jugadores ausentes. Sirve para calibrar la
> *máscara*, nunca para medir error de imputación.

## Limitaciones conocidas

1. **Solo cámara principal.** No modela repeticiones, primeros planos ni
   cámaras tácticas. Por eso `alive_only=True` restringe el análisis a balón
   en juego.
2. **Sin oclusión mutua.** Un jugador tapado por otro dentro del cuadro se
   considera visible. En un pipeline de visión real, no lo sería.
3. **Sin fallos de detección.** Un sistema real pierde jugadores por
   iluminación, uniformes similares o aglomeraciones. Aquí la visibilidad es
   geométrica y perfecta.

Las tres hacen que el benchmark sea **optimista**. El error real de un pipeline
sobre video será mayor. Decirlo antes de que lo pregunten es parte del trabajo.
