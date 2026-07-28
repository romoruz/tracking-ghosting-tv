# 09 · Decisiones de diseño y bitácora de errores

> Cada entrada de este documento corresponde a un bug real que se cometió, se
> detectó y se corrigió, o a una decisión que parece arbitraria y no lo es.
> **Si estás modificando el código, lee esto antes.** Varios de estos fallos
> son silenciosos: producen números plausibles y falsos.

---

## A. Errores que producían números falsos

### A.1 Los proveedores usan sistemas de coordenadas distintos

**Síntoma:** ninguno. El código "funcionaba".

| Proveedor | rango de x | Tipo |
|---|---|---|
| Sportec | −52.5 … +52.5 | métrico, origen al centro |
| Metrica | 0 … 1 | normalizado |
| Tracab | −5250 … +5250 | centímetros, origen al centro |

El cargador original hacía `coordenada × 105` asumiendo `[0,1]`. Con Metrica
funcionaba; con Sportec habría multiplicado −52.5 × 105 = −5512 m.

**Corrección:** `_pitch_and_scalers` lee el rango de
`dataset.metadata.coordinate_system.pitch_dimensions` y normaliza
explícitamente. **Nunca se asume la escala.**

**Test que lo cubre:** `test_proveedores_convergen_al_mismo_esquema` construye
datasets kloppy de ambos proveedores con los mismos jugadores en las mismas
posiciones físicas y exige tensores idénticos.

---

### A.2 La inferencia de portero fallaba por el cambio de lado

**Síntoma:** `--include-gk` no cambiaba **ni una** muestra puntuable. Se
interpretó como "el método es robusto". Era un no-op.

**Causa:** los equipos cambian de lado en el descanso. Un portero defiende
x≈5 en el primer tiempo y x≈100 en el segundo, así que su posición media sobre
el partido completo es ~52.5 m: **el centro de la cancha, el valor menos
parecido a un portero que existe.** El criterio "el jugador cuya x media está
más cerca de una meta" seleccionaba a otro, típicamente un suplente. Y si ese
suplente no jugó el tramo analizado, aportaba cero muestras.

Consecuencia colateral: **los porteros reales llevaban todas las corridas
evaluados como jugadores de campo**, aportando el 22% de las muestras y ~1.4 m
de error inflado.

**Corrección:** mediana **por periodo** de la distancia a la meta más cercana,
mínimo entre periodos. Mediana y no media, para ser robusta a las salidas del
portero.

**Tests:** `test_inferencia_de_portero_con_cambio_de_lado`,
`test_include_gk_cambia_la_poblacion`. Además el generador sintético ahora
**cambia de lado en el descanso**, porque un test que nunca cambia de lado
jamás habría atrapado esto.

> `is_gk` se guarda dentro del `.npz`. Si cambias el cargador, regenera los
> datos con `--force` o seguirás usando la inferencia vieja.

---

### A.3 Un número inventado presentado como medición

En una fase temprana se generó un script que "medía" el error así:

```python
simulated_generated = all_real_defenders + np.random.normal(0, 2.5, shape)
```

Es decir: tomaba las posiciones reales, les sumaba ruido gaussiano de 2.5 m, y
"medía" que el error era ~2.5 m. **Ese número no salía de ningún modelo.**
Estuvo a punto de acabar en una presentación.

**Regla que se deriva:** ninguna cifra entra en un reporte si no se puede
regenerar ejecutando un script del repositorio sobre datos versionados.

---

### A.4 B4 calculado por ventana en vez de sobre el partido completo

Los offsets de B4 son **recursivos**: el del frame t depende del centroide
votado en t, que depende de los offsets de t−1. Calcularlo dentro de cada
ventana de entrenamiento reiniciaría ese estado y produciría un baseline
artificialmente malo — **inflando la mejora aparente del modelo**.

**Corrección:** `04_train.py` calcula B4 sobre el partido completo *antes* de
trocear. Documentado en `models/dataset.py`.

---

## B. Errores de método estadístico

### B.1 Comparar dos intervalos marginales para juzgar una diferencia

Con IC marginales, `A = 16.5 [13.7, 19.5]` y `B = 15.6 [13.7, 17.7]` se solapan
casi por completo y la conclusión sería "no hay diferencia". Es falsa.

Cuando dos estimadores se evalúan sobre la **misma muestra** —mismos frames,
mismos jugadores, misma cámara— la mayor parte de la incertidumbre es *común* y
se cancela al restar. Un partido con muchas fases de ataque sostenido da error
alto en ambos a la vez; eso infla los dos intervalos y no afecta a la
diferencia.

**Corrección:** `paired_block_bootstrap_ci` remuestrea bloques y calcula la
diferencia **dentro de cada réplica**. Es el procedimiento que usa Choi para
contrastar B4 contra B2.

**Tests:** comparar un estimador consigo mismo da cero exacto; el intervalo
pareado es más estrecho que la suma de los marginales; detecta la diferencia
conocida B1 vs B4.

---

### B.2 Tratar un rango publicado como frontera dura

El `15.6 – 16.9 m` de Choi son **tres estimaciones puntuales de tres partidos**,
no un intervalo de confianza. Un valor propio de 17.9 m no está "fuera de
rango" si su IC lo contiene.

**Corrección:** `stratified_bootstrap_ci` calcula IC **por bin**, no solo
global. El bin largo tiene menos rachas independientes de lo que sugiere su `n`
—pocas fases de ataque sostenido, cada una con cientos de frames casi
idénticos— así que su IC es varias veces más ancho que el del bin corto con el
mismo `n`.

---

### B.3 Agrupar evaluaciones que reutilizan los mismos datos

Siete modelos × dos partidos son catorce mediciones. Meterlas todas en un
bootstrap agrupado haría aparecer los mismos frames de Metrica **siete veces**,
estrechando el intervalo por multiplicar observaciones no independientes.

**Corrección:** `08_external_test.py` reporta las catorce por separado *y*
además un **conjunto** (promedio de las siete predicciones) evaluado una vez
por partido, que sí se puede agrupar.

---

### B.4 El bootstrap ingenuo ignora la autocorrelación

A 25 fps los frames consecutivos difieren en centímetros. Remuestrear frames
individuales trataría 25 observaciones casi idénticas como independientes,
produciendo intervalos anticonservadores.

**Corrección:** bloques contiguos de 1 minuto. Hay un test que verifica que el
intervalo por bloques sea **más ancho** que el ingenuo.

---

### B.5 `pool_paired` debe desplazar los índices entre partidos

Los índices de frame se reinician en cada partido. Concatenar sin desplazar
fundiría el minuto 3 del partido A con el minuto 3 del B en un mismo bloque de
remuestreo. **Test:** `test_pool_paired_no_mezcla_bloques_entre_partidos`.

> Matiz: el agrupado **no** es necesariamente más estrecho que el partido más
> afortunado. Si los partidos tienen efectos genuinamente distintos, incorpora
> también esa varianza entre partidos, y eso es correcto: un intervalo estrecho
> de un solo partido es sobreconfiado respecto a la generalización.

---

### B.6 Sensibilidad de protocolo: el balón muerto

Sportec expone `ball_state`; Metrica no. Filtrar balón parado mueve el peso del
bin `>9.6 s` de 50.8% a 41.3% (media sobre 7 partidos).

Mecanismo: en balón muerto la cámara se queda sobre el punto de la jugada y los
del lado lejano acumulan huecos enormes — justo los que engordan el bin largo.

**Corolario sobre Choi:** usó Metrica, así que su protocolo **incluye balón
parado por limitación del proveedor, no por elección**. No es un error suyo;
es una sensibilidad que hay que igualar para comparar.

Bandera: `--include-dead-ball`.

---

## C. Resultados negativos (mantener, no borrar)

### C.1 La partición del portero no generaliza

Anclar al portero a su portería en vez de al centroide del equipo parecía una
mejora clara sobre 2 partidos de Metrica (+0.94 y +1.32 m en el bin largo, 8/8
deltas positivos). Sobre 7 partidos de Bundesliga **se invierte**:

```
bin >9.6 s agrupado:  -0.42 m [-0.70, -0.13]   EMPEORA de forma creíble
```

**Mecanismo probable:** con τ = 3 s, pasados ~10 s de oclusión el modelo coloca
al portero prácticamente sobre su línea. Pero el portero moderno juega
adelantado 15–25 m durante posesión rival sostenida — exactamente cuando lleva
más tiempo oculto. El ancla se equivoca más justo donde más pesa.

El ancla por defecto volvió a `"team"`. `"goal"` se conserva como opción
documentada, **no como recomendación**.

**Por qué importa:** es exactamente el tipo de hallazgo que un estudio de dos
partidos habría publicado como contribución estando equivocado. Es el
argumento más fuerte a favor de la validación cruzada.

---

## D. Decisiones de diseño que parecen arbitrarias

### D.1 `N` no es 22

Con sustituciones aparecen 26–40 jugadores distintos por partido (Sportec lista
20 por equipo). Fijar N=22 obligaría a decidir arbitrariamente quién ocupa cada
casilla tras un cambio. En su lugar hay una máscara `on_pitch` separada.

**`on_pitch` y `visible` son máscaras distintas y confundirlas invalida todo:**
- `on_pitch[t,i]` — verdad de existencia: el jugador está en el campo.
- `visible[t,i]` — verdad de observación: la cámara lo muestra.

### D.2 Las constantes de tiempo se reescalan con el fps

El paper define α = 0.06 para el paneo **a 25 fps** y el peso del EMA de offsets
**a 5 fps**. Evaluar a otra frecuencia sin reescalar cambiaría la inercia física.
Ambos módulos resuelven
$(1-a_{\text{fps}})^{\text{fps}} = (1-a_{\text{ref}})^{\text{ref}}$.

### D.3 Decimar, no interpolar

`Match.resample()` submuestrea por decimación entera. Promediar frames
inventaría posiciones que nadie ocupó y contaminaría la verdad de terreno.

### D.4 La cabeza del modelo se inicializa a cero

Con la cabeza a cero, el modelo predice **exactamente B4**. Eso acota el riesgo
a la baja —no puede ser catastróficamente peor que el baseline— y es lo que
hace viable entrenar con 5 partidos. **Test:**
`test_modelo_arranca_siendo_exactamente_b4`.

### D.5 Los pesos de la pérdida están calibrados, no elegidos

Medidos sobre la salida de B4 (residuo cero), los términos crudos valían:

```
rec = 13.5    smooth = 0.35    vel = 0.51    acc = 86.0
```

El de aceleración era **6× el de reconstrucción antes de que el modelo hiciera
nada**: medía los saltos del propio B4, que salta cuando un jugador reaparece.
Con pesos iguales el modelo habría dedicado casi todo su esfuerzo a suavizar el
baseline. Los pesos por defecto hacen que cada término físico aporte ~3–5%.

**Si cambias el baseline o la escala de las features, recalibra.**

### D.6 Hacen falta las dos familias de términos físicos

Las bisagras cinemáticas tienen **gradiente nulo** en la región factible: no
penalizan nada por debajo de `v_max`. Eso deja pasar oscilación de alta
frecuencia — un jugador que tiembla a 3 m/s es absurdo pero no viola ninguna
cota, y los transformers producen justo ese artefacto. `L_smooth` lo castiga en
todo el dominio; las bisagras ponen el techo duro.

**Nomenclatura:** esto **no** es una "PINN". No hay ecuación diferencial que
gobierne el sistema. Son penalizaciones de factibilidad. Llamarlas así evita
inflar el vocabulario ante alguien que sepa la diferencia.

### D.7 El test de equivarianza corre en float64

En float32 la atención suma 26 términos; hacerlo en distinto orden cambia el
último bit y produce discrepancias de ~1e-5 sin relación con la equivarianza
(la aritmética de punto flotante no es asociativa). El test fallaba de forma
intermitente.

Aflojar la tolerancia habría enmascarado una violación real, que sería de orden
1 y no de 1e-5. En float64 el ruido baja a ~1e-14 y un umbral estricto
distingue ambas cosas.

### D.8 Partición por partido, nunca por ventana

A 5 fps con solapamiento, dos ventanas vecinas comparten casi todos sus frames.
Separarlas entre train y test daría números de fantasía.

---

## E. Operativa: errores que costaron tiempo

### E.1 El checkpoint solo se guardaba al final

Una corrida nocturna de 8 horas se perdió entera. Ahora se guarda **cada vez
que mejora la validación**, más `--resume`.

### E.2 Estimar el coste sin medirlo

Se estimó "6–7 min por época" extrapolando ráfagas de 2–3 iteraciones. El valor
real en la laptop fue de ~19 min: un i7-1165G7 de 15 W en chasis 2-en-1 sufre
**throttling térmico** severo bajo carga sostenida. Los primeros segundos van a
turbo; la hora catorce, no.

**Corrección:** `--dry-run SEGUNDOS` mide el ritmo real y extrapola, usando la
mediana de la **segunda mitad** de las mediciones. Úsalo siempre antes de una
corrida larga.

### E.3 El scheduler coseno con early stopping

`CosineAnnealingLR(T_max=epochs)` anela según el contador de épocas. Si pides
100 y el early stopping corta en 30, **la tasa nunca llegó a bajar** y el modelo
no afinó. Por defecto ahora es `ReduceLROnPlateau`.

### E.4 El solapamiento excesivo de ventanas

`stride=10` sobre ventanas de 50 es **80% de solape**: dos ventanas vecinas
comparten 40 de 50 frames, casi sin información nueva y al mismo coste.
`stride=25` (50%) reduce el cómputo 2.5× sin pérdida apreciable.

### E.5 En Kaggle, `/kaggle/working` es volátil

Y la celda de reconstrucción hacía `rmtree('/kaggle/working/gh')`, borrando
`gh/reports/` con los checkpoints dentro. Preserva `reports/` antes de
reconstruir, y **descarga los artefactos en cuanto existan**.

---

## F. Cosas que NO se han hecho y por qué

- **`--long` (ventana de 30 s):** implementado pero **nunca ejecutado con
  éxito**. La hipótesis era que el bin largo necesitaba ventanas más grandes.
  Resultó innecesaria: las features de ancla de largo alcance (§D del doc 07)
  resolvieron el problema con ventana de 10 s.
- **Modo bidireccional:** implementado, nunca reportado. Es interpolación, no
  tiempo real.
- **Calibración del viewport con SkillCorner:** pendiente. Es la contribución
  metodológica más barata que queda.
- **Modelo generativo (DDPM/VAE):** no empezado. Daría distribuciones en vez de
  puntos, que es lo que quiere un cuerpo técnico para escenarios.
- **Pipeline de visión sobre video:** fuera de alcance. Este repositorio asume
  tracking ya extraído.
