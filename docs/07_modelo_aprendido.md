# 07 · El modelo aprendido: imputador residual

## 1. Qué predice, y por qué eso y no otra cosa

El modelo **no predice la posición absoluta**. Predice la **corrección sobre
B4**, el mejor método sin entrenamiento del benchmark:

$$\hat p_j(t) \;=\; \underbrace{\hat c(t) + \text{off}_j}_{\text{B4, forma cerrada}} \;+\; \underbrace{f_\theta\big(X_{\text{obs}},M,b\big)_j(t)}_{\text{residuo aprendido}}$$

Tres razones, y la primera es la que manda con 5 partidos de entrenamiento:

1. **Converge mucho más rápido.** B4 ya explica la mayor parte de la señal
   ("el equipo se desplazó y el jugador conservó su rol"); el modelo solo tiene
   que aprender en qué se equivoca.
2. **Acota el riesgo a la baja.** Si $f_\theta \to 0$ se recupera B4 exactamente.
   La cabeza se inicializa a cero, así que el modelo **arranca siendo B4** y solo
   puede mejorar desde ahí.
3. **La comparación es directa.** El residuo *es* la mejora, y se mide con el
   mismo bootstrap pareado ya validado.

> B4 se calcula sobre el **partido completo** antes de trocear en ventanas. Sus
> offsets son recursivos; calcularlo por ventana reiniciaría el estado y
> produciría un baseline peor, inflando artificialmente la mejora aparente.

## 2. Entrada

El tensor de entrada es **el mejor estado conocido** en cada instante:

- jugador **visible** → su posición real
- jugador **oculto** → la estimación de B4

más la máscara y el gap, que le dicen al modelo cuáles de esas entradas son
observación y cuáles conjetura, y desde hace cuánto.

### Las 18 features, en orden

| # | Feature | Descripción |
|---|---|---|
| 0–1 | `x_norm`, `y_norm` | posición conocida, normalizada a la cancha |
| 2–3 | `vx_norm`, `vy_norm` | velocidad desde la última observación, topada a 11 m/s |
| 4 | `visible` | 1 = observación, 0 = conjetura de B4 |
| 5 | `gap_log` | $\log(1+\Delta/10)$; 0 si visible |
| 6–7 | `dx_ball`, `dy_ball` | posición relativa al balón |
| 8–9 | `dx_centroid`, `dy_centroid` | relativa al centroide visible del propio equipo |
| 10 | `team` | 0 local / 1 visitante |
| 11 | `is_gk` | portero |
| 12–13 | `ball_x`, `ball_y` | balón en absoluto |
| 14 | `on_pitch` | 0 = relleno o fuera de cancha |
| **15–16** | **`last_x`, `last_y`** | **última posición REALMENTE observada** |
| **17** | **`d_last`** | **distancia entre esa última vista y la estimación de B4** |

**Las tres últimas son las que resolvieron el bin de oclusión larga**, y merecen
explicación aparte (§4).

Sobre las coordenadas: se alimentan **absolutas y relativas a la vez**. Centrar
en el balón mejora la generalización, pero sustituir las absolutas perdería
información que sí importa —distancia a la portería propia, línea de fuera de
juego, cercanía a la banda—.

El gap se comprime logarítmicamente porque los huecos van de 0.2 s a varios
minutos; sin comprimir, un gap de 300 s dominaría la entrada.

## 3. Arquitectura

```
entrada (B, T, N, 18)
    │  proyección lineal a D=128  +  embedding temporal
    ▼
┌─ bloque ×4 ───────────────────────────────────┐
│  atención ESPACIAL   sobre los N jugadores,   │
│                      dentro de cada frame     │
│  atención TEMPORAL   sobre los T frames,      │
│                      dentro de cada jugador   │
│  feed-forward                                 │
└───────────────────────────────────────────────┘
    │  LayerNorm + cabeza lineal
    ▼
tanh(·) × 30 m  →  (B, T, N, 2)  residuo en metros
```

**863,618 parámetros.** `D=128`, 4 bloques, 4 cabezas.

### Por qué alternar los ejes en vez de aplanar

Aplanar $T\cdot N$ tokens y hacer atención completa sería $O((TN)^2)$ y, sobre
todo, perdería las dos simetrías del problema:

- **Equivarianza a permutaciones.** Los jugadores de un equipo son un
  **conjunto**, no una lista. La atención espacial es equivariante por
  construcción. Un MLP sobre el vector concatenado de $2N$ coordenadas rompería
  esto y el modelo memorizaría el orden del roster.
- **Causalidad selectiva.** Separar los ejes permite aplicar la máscara causal
  **solo** en el temporal, y entrenar el mismo modelo en modo online o
  bidireccional con una bandera.

Nota de coste: la atención temporal es $O(T^2)$ pero solo supone el 9% del
cómputo a $T=50$ y el 22% a $T=150$. **El feed-forward domina y es lineal en
$T$**, por eso duplicar la ventana cuesta ~1.6× por segundo de vídeo cubierto,
no 9×.

### La cota del residuo

La salida pasa por $\tanh(\cdot)\times 30$ m. Impide que una salida disparatada
mueva a un jugador a la tribuna y estabiliza el arranque, cuando los pesos aún
son ruido.

## 4. La pieza que resolvió el bin largo

El bin `>9.6 s` está definido como los casos en que el jugador lleva más de
9.6 s oculto. Con ventana de **10 s exactos**, para casi todo ese bin el jugador
**no aparece visible ni una sola vez dentro de la ventana**: el modelo solo ve
la estimación de B4 repetida.

El primer modelo, sin las features 15–17, no encontraba efecto ahí:
`+0.67 m [−1.33, +2.67]`.

La hipótesis fue que hacía falta alargar la ventana a 30 s. **Resultó
innecesaria.** Bastó con darle al modelo la **última posición realmente
observada** del jugador, arrastrada hacia adelante y reiniciada por periodo. B4
ya ha "suavizado" esa información en su estimación; las features crudas son el
único enlace directo con lo que de verdad se vio.

Con ellas, el bin largo pasó a `+4.04 m [+3.46, +4.66]` agrupado sobre 7
partidos.

**Lección:** antes de escalar la arquitectura, comprobar si falta información en
la entrada. Era el arreglo barato y fue suficiente.

## 5. Función de pérdida

$$\mathcal{L} = \mathcal{L}_{\text{rec}} + \lambda_s\mathcal{L}_{\text{smooth}} + \lambda_v\mathcal{L}_{\text{vel}} + \lambda_a\mathcal{L}_{\text{acc}}$$

**Reconstrucción** (Huber, δ=5 m), solo sobre pares ocultos con verdad conocida.
Huber y no MSE porque la distribución del error tiene cola pesada: con MSE, los
pocos casos de 40 m dominarían el gradiente.

**Suavidad**, segunda derivada discreta:
$\big\|\hat p(t{+}1) - 2\hat p(t) + \hat p(t{-}1)\big\|^2$

**Bisagras cinemáticas**: $\max(0, \|\hat v\| - 11)^2$ y $\max(0, \|\hat a\| - 7)^2$

### Por qué hacen falta las dos familias

Las bisagras tienen **gradiente nulo** en la región factible. No penalizan nada
mientras el modelo se mantenga bajo $v_{\max}$, así que dejan pasar oscilación
de alta frecuencia — un jugador que tiembla a 3 m/s es físicamente absurdo pero
no viola ninguna cota, y los transformers producen justo ese artefacto.
`L_smooth` lo castiga en todo el dominio; las bisagras ponen el techo duro.

### Pesos: calibrados, no elegidos

Evaluados sobre la salida de B4 (residuo cero), los términos crudos valen:

```
rec = 13.5    smooth = 0.35    vel = 0.51    acc = 86.0
```

El de aceleración es **6× el de reconstrucción antes de que el modelo haga
nada**: mide los saltos del propio B4, que salta cuando un jugador reaparece y
su estimación se engancha a la posición observada. Con pesos iguales, el modelo
dedicaría casi todo su esfuerzo a suavizar el baseline.

Por defecto: `λ_smooth = 2.0`, `λ_vel = 1.3`, `λ_acc = 0.005`, de modo que cada
término físico aporte ~3–5% de la reconstrucción en la inicialización.
**Si cambias el baseline o la escala de las features, recalíbralos.**

## 6. Entrenamiento

| Parámetro | Valor | Nota |
|---|---|---|
| ventana | 50 frames (10 s a 5 fps) | |
| stride | 25 | 50% de solape; 80% no aporta y cuesta igual |
| lote | 64 (GPU) / 16 (CPU) | |
| optimizador | AdamW, lr 3e-4, wd 1e-4 | |
| scheduler | `ReduceLROnPlateau`, factor 0.5, paciencia 3 | coseno falla con early stopping |
| early stopping | paciencia 12 sobre la métrica vigilada | |
| clip de gradiente | 1.0 | |
| semilla | 1000 + índice de tanda | |

El early stopping y el checkpoint vigilan la métrica que elijas con
`--monitor` (`global`, `<=2s`, `2-9.6s`, `>9.6s`). **Vigilar el global cuando
el objetivo es el bin largo puede detener la corrida en un punto bueno en
promedio y malo donde importa.**

Épocas típicas hasta early stopping: 51–79. Salvo la tanda J03WQQ, que cortó en
16 con una trayectoria de optimización fallida (ver `06_resultados.md`).

## 7. Inferencia sobre un partido completo

El modelo ve ventanas de $T$ frames. Para reconstruir el partido se usan
ventanas solapadas al 50% y se toma de cada una **solo su segunda mitad**: así
todo frame evaluado tiene al menos $T/2$ frames de contexto por detrás.

Tomar la primera mitad mezclaría predicciones hechas casi sin historia con
predicciones bien informadas, y en modo causal eso inflaría el error de forma
artificial.

## 8. Invariantes verificadas por tests

| Invariante | Test |
|---|---|
| Con cabeza a cero, la salida es exactamente B4 | `test_modelo_arranca_siendo_exactamente_b4` |
| Permutar jugadores permuta la salida (float64) | `test_modelo_es_equivariante_a_permutaciones` |
| En modo causal, alterar el futuro no cambia el pasado | `test_mascara_causal_impide_ver_el_futuro` |
| La suavidad castiga jitter que las bisagras dejan pasar | `test_perdida_penaliza_jitter_dentro_de_la_region_factible` |
