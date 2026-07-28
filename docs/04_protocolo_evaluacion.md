# Protocolo de evaluación

Tres decisiones que no son opcionales. Saltarse cualquiera produce números
plausibles y falsos.

## 1. Estratificación por gap de oclusión

**Un modelo de velocidad constante gana a cualquier red neuronal cuando el
jugador lleva 2 segundos oculto.** El promedio global mezcla ese régimen
trivial con el difícil y oculta dónde está realmente el mérito de un método.

Bins (los de Choi 2026):

| Bin | Régimen | Qué lo resuelve |
|---|---|---|
| ≤ 2 s | trivial | extrapolación lineal |
| 2 – 9.6 s | intermedio | ancla de formación |
| > 9.6 s | **abierto** | nada, todavía |

El corte en 9.6 s no es arbitrario: es la longitud de ventana del Graph
Imputer de DeepMind, más allá de la cual ese modelo no está definido. Y ahí
caen **50–57% de las muestras ocultas**.

Reporta siempre los tres bins **y el peso de cada uno**. Un método excelente en
≤2 s que cubre el 19% de los casos vale menos que uno decente en >9.6 s que
cubre el 41%.

## 2. Mediana, no media

La distribución del error tiene cola derecha pesada: un jugador oculto tres
minutos puede estar a 40 m. La media queda dominada por esa cola y deja de
describir el caso típico.

Choi también reporta mediana; usar media rompería la comparabilidad. Se
reportan además `mean_all` y `p90_all` como diagnóstico de la cola, pero la
cifra de titular es la mediana.

## 3. Block bootstrap

A 25 fps, frames consecutivos están fuertemente autocorrelacionados: la
posición en $t$ y en $t+1$ difieren en centímetros.

**Remuestrear frames individuales trataría 25 observaciones casi idénticas como
25 muestras independientes.** El resultado son intervalos de confianza
artificialmente estrechos (anticonservadores) que te harían declarar
significativa una diferencia que es ruido.

La corrección: agrupar en bloques contiguos de **1 minuto** y remuestrear
bloques completos con reemplazo. Dentro de un bloque las observaciones siguen
juntas, como en los datos originales.

El test `test_bootstrap_de_bloques_es_mas_ancho_que_el_ingenuo` verifica que el
intervalo por bloques sea efectivamente más ancho que el ingenuo. Si algún día
falla, el bootstrap está mal implementado.

## Población puntuable

Se puntúan los pares (frame, jugador) que cumplen **todas** estas condiciones:

- el jugador está en cancha,
- **no** es visible,
- ya fue observado al menos una vez en el periodo (gap finito),
- el método produjo una estimación,
- el balón está en juego (configurable, activado por defecto).

Exclusiones y su razón:

| Excluido | Razón |
|---|---|
| Cold start | Ningún método causal puede posicionarlo; no discrimina |
| Fuera de cancha | No existe |
| Balón muerto | La cámara hace planos que el modelo no representa |
| Porteros (por defecto) | Dinámica cualitativamente distinta; se reporta aparte |

## Métricas

### Implementadas

**Error de posición del jugador oculto** (m): distancia euclidiana entre
posición imputada y real. Mediana global, mediana por bin, p90, e IC 95% por
block bootstrap.

### Pendientes (paso 3)

**MAE del mapa de pitch control** (pp): pitch control sobre grilla de 3 m con
el modelo de Spearman (2017), comparado contra el mapa de observación completa.
Se reporta sobre toda la cancha y sobre la **zona oculta** por separado.

**Error de control-share** (pp): desviación absoluta media por frame del
número de titular *"el equipo A controla el x% de la cancha"*.

Estas dos son las **métricas relevantes para la decisión**: determinan qué
reporta el sistema, no solo dónde coloca a los jugadores. Choi selecciona
métodos sobre ellas y reporta el MAE del mapa solo por completitud.

## Números de referencia

Choi (2026), $W=44$ m, tres partidos de Metrica, **primeros 45 min**, 5 fps.

> **El recorte a 45 minutos no es un detalle.** Comparar contra estas cifras
> usando el partido completo introduce una diferencia sistemática que golpea
> específicamente el bin de oclusión larga. En el segundo tiempo entran
> suplentes, y un suplente **no tiene offset de rol almacenado**: B4 cae a B2 y
> de ahí a B1, cuyo error en la cola larga es aproximadamente el doble. A eso
> se suman fatiga, deriva de rol y cambios tácticos por marcador.
>
> Usa `--minutes 45` para replicar el protocolo. Sin ese flag, espera un bin
> `>9.6 s` inflado en 1–2 m y **no lo interpretes como un fallo de tu
> implementación**.

**Escalera completa** (mediana de error de posición, m, g1/g2/g3):

| Método | Mediana (m) | Hidden ctrl MAE (pp) | Share err. (pp) |
|---|---|---|---|
| B0 ignorar | — | 26.9 / 25.6 / 25.1 | 13.4 / 12.5 / 11.1 |
| B1 última vista | 19.6 / 17.9 / 18.4 | 22.1 / 20.0 / 19.5 | 10.6 / 9.5 / 8.2 |
| B2 ancla | 13.6 / 12.8 / 12.5 | 15.7 / 14.6 / 14.3 | 6.2 / 5.4 / 4.4 |
| B5 plantilla fija | 22.7 / 20.7 / 21.9 | 23.0 / 18.8 / 19.1 | 10.3 / 7.7 / 6.9 |
| B3E solo EMA | 15.7 / 15.2 / 16.3 | 13.2 / 12.2 / 13.6 | 5.8 / 4.8 / 5.0 |
| B3V solo velocidad | 13.2 / 12.2 / 11.9 | 15.5 / 14.4 / 14.0 | 6.2 / 5.5 / 4.4 |
| B3 EMA + velocidad | 14.6 / 14.0 / 15.1 | 12.8 / 11.8 / 13.2 | 5.7 / 4.6 / 4.7 |
| **B4 voto de centroide** | **11.6 / 10.0 / 9.7** | 13.3 / 12.2 / 13.8 | **4.7 / 4.5 / 4.7** |

**B4 estratificado** — el objetivo real:

| Gap | Mediana B4 | Peso |
|---|---|---|
| ≤ 2 s | 3.3 – 3.7 m | 43–50% (con el siguiente) |
| 2 – 9.6 s | 7.2 – 8.9 m | |
| **> 9.6 s** | **15.6 – 16.9 m** | **50–57%** |

**Ese último renglón es el objetivo del paso 3.** El autor lo dice
explícitamente: es donde queda margen y donde espera que ataquen los modelos
aprendidos de horizonte largo.

## Los rangos publicados no son fronteras

El `15.6 – 16.9 m` de Choi para el bin largo son **tres estimaciones puntuales
de tres partidos**, no un intervalo de confianza. Comparar un valor propio
contra ese rango como si fuera una frontera dura es el error que el block
bootstrap existe para evitar.

Por eso `evaluate()` calcula IC **por bin**, no solo global. El bin de oclusión
larga contiene menos rachas independientes de lo que sugiere su `n` —son pocas
fases de ataque sostenido, cada una con cientos de frames casi idénticos— así
que su incertidumbre real es grande. En Metrica, el IC de la mediana global de
B4 sobre medio partido ya mide ~3 m de ancho; el del bin largo es mayor.

La lectura correcta de un valor de 17.9 m frente al rango publicado no es
"fuera de rango", sino: *¿su intervalo se solapa con el rango de estimaciones
publicadas?* Si se solapa, es la misma cifra medida con ruido.

## Comparar dos métodos: bootstrap pareado

Los IC marginales sirven para reportar *un* método. **No sirven para comparar
dos.** Mirar dos intervalos, ver que se solapan y concluir "no hay diferencia"
es un error conservador hasta la inutilidad.

La razón: cuando los dos estimadores se evalúan sobre la misma muestra —mismos
frames, mismos jugadores, misma cámara— la mayor parte de su incertidumbre es
*común*. Un partido con muchas fases de ataque sostenido dará error alto en
ambos a la vez; esa variabilidad infla los dos intervalos marginales pero no
afecta en nada a la diferencia entre ellos.

`paired_block_bootstrap_ci` remuestrea bloques y calcula la diferencia
**dentro de cada réplica**, con lo que esa varianza común se cancela. El
intervalo resultante es típicamente varias veces más estrecho, y es el único
que responde la pregunta correcta: *¿el método B mejora sobre el A?*

Es el procedimiento que usa Choi (2026) para contrastar B4 contra B2.

Requisitos:
- La población debe ser idéntica: un par (frame, jugador) entra solo si ambos
  estimadores produjeron estimación finita.
- Se reporta el delta global y por bin, porque una mejora puede concentrarse
  entera en el régimen de oclusión larga y desaparecer en el promedio.

### Agrupar partidos

Con un solo partido —o medio— la potencia por bin es baja: el régimen de
oclusión larga contiene pocas rachas independientes, así que su intervalo sale
ancho aunque el efecto sea real. El síntoma es característico: **el signo del
efecto coincide en todos los bins y todos los partidos, pero cada partido
declara "creíble" bins distintos.** Eso no es contradicción, es falta de
muestra.

`pool_paired` agrupa varias comparaciones pareadas en un solo bootstrap. El
detalle que hay que cuidar: los índices de frame se reinician en cada partido,
así que concatenar sin desplazar fundiría el minuto 3 del partido A con el
minuto 3 del B en un mismo bloque de remuestreo. La función desplaza cada
partido por un offset alineado a bloque, de modo que ningún bloque cruce la
frontera.

Un matiz importante: el agrupado **no** es necesariamente más estrecho que el
partido más afortunado. Si los partidos tienen efectos genuinamente distintos,
incorpora también esa varianza entre partidos. Es lo correcto — un intervalo
estrecho salido de un solo partido es sobreconfiado respecto a la
generalización.

## Verificación de la implementación

Reproducir estos números sobre Metrica es la validación del pipeline. Si tus
B0–B4 no coinciden (±10%), el bug está en la carga de datos o en el simulador,
no en tu modelo. **Es la única forma de detectar ese tipo de error antes de
que contamine todo lo demás.**

Sobre datos sintéticos los valores absolutos serán mucho menores (~3 m para B4
frente a los 9.7–11.6 m publicados) porque los autómatas mantienen sus offsets
de rol con varianza casi nula. Lo que sí debe conservarse es el **orden de
mérito**: `B4 < B3V < B2 < B3 < B3E < B1, B5`.

Dos comprobaciones adicionales que confirman que la implementación es fiel y no
solo "da números":

1. **B2 y B3V coinciden exactamente en el bin >9.6 s.** B3V es B2 más
   extrapolación de velocidad mezclada con constante de 1.5 s; a los 9.6 s ese
   término ya decayó a $e^{-6.4}\approx 0.002$. Que salgan idénticos y no
   parecidos verifica la mezcla exponencial.
2. **El peso del bin >9.6 s es menor en sintético (~30%) que en real
   (50–57%).** El balón sintético recorre la cancha de forma más uniforme que
   el real, así que produce menos fases de ataque sostenido y por tanto menos
   oclusiones largas. Al pasar a datos reales, ese peso debe subir. Si no sube,
   revisa el simulador.
