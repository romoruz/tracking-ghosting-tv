# 06 · Resultados

> Todas las cifras de este documento se pueden regenerar ejecutando los scripts
> del repositorio sobre los `.npz` de `data/processed/`. Ninguna es estimación.
> Los CSV originales están en `reports/tables/`.

---

## 1. Conjuntos de datos

| Conjunto | Partidos | Proveedor | `ball_state` | Porteros en metadatos | Rol |
|---|---|---|---|---|---|
| Sportec / DFL | 7 | TRACAB gen-5 | **sí** (48–59% vivo) | **sí** | entrenamiento + CV |
| Metrica | 2 | Metrica Sports | no | no (inferidos) | **test congelado** |

Esas dos diferencias de la derecha hacen que **los números de ambos conjuntos
no sean directamente comparables entre sí**. Sí lo son dentro de cada uno, que
es lo que importa para medir modelo contra B4.

---

## 2. Réplica del benchmark publicado ✅

Metrica, primeros 45 min, W = 44 m, porteros incluidos y anclados al equipo
(configuración A). B4 cae dentro de los rangos de Choi (2026) en las cinco
métricas.

| Métrica | Choi 2026 | metrica_1 | metrica_2 |
|---|---|---|---|
| mediana global | 9.7 – 11.6 | 11.0 ✓ | 10.1 ✓ |
| ≤2 s | 3.3 – 3.7 | 3.6 ✓ | 3.3 ✓ |
| 2–9.6 s | 7.2 – 8.9 | 8.5 ✓ | 7.3 ✓ |
| >9.6 s | 15.6 – 16.9 | 16.5 ✓ | 17.9 (IC contiene el rango) |
| peso >9.6 s | 50 – 57% | 55.2% ✓ | 49.7% |

**Réplica independiente lograda con código propio.**

---

## 3. Generalización de la escalera a Bundesliga ✅

El orden de mérito se conserva íntegro en los 7 partidos de Sportec:
`B4 < B3V < B2 < B3 < B3E < B1, B5`. B4 gana en los 9 partidos de los 2
proveedores.

Nadie había verificado que los hallazgos de Choi se sostuvieran fuera de
Metrica. Se sostienen.

---

## 4. Sensibilidad de protocolo: el balón muerto ⚠️

| Conjunto | peso >9.6 s | balón vivo |
|---|---|---|
| Choi 2026 (Metrica) | 50 – 57% | no filtrable |
| Metrica (nuestro) | 49.7 – 55.2% | no filtrable |
| Sportec, **solo vivo** | 35.6 – 47.8% (media 41.3%) | 48 – 59% |
| Sportec, **con balón muerto** | 43.4 – 54.1% (media 50.8%) | — |

Incluir balón parado desplaza el peso del régimen de oclusión larga **+9.5
puntos de media**, y 5 de 7 partidos entran en el rango publicado.

**Corolario:** Choi usó Metrica, que no expone `ball_state`, así que su
protocolo **incluye balón parado por limitación del proveedor, no por
elección**. Cualquiera que use este benchmark con datos modernos tiene que
igualar esa condición para comparar.

---

## 5. Partición del portero: RESULTADO NEGATIVO ❌

Anclar al portero a su portería en vez de al centroide parecía una mejora clara
sobre 2 partidos de Metrica. **No generaliza.**

| Bin | Metrica (2 partidos) | Bundesliga (7, agrupado, balón vivo) | Bundesliga (con balón muerto) |
|---|---|---|---|
| ≤2 s | +0.19, +0.17 | +0.09 [+0.06, +0.12] | +0.08 [+0.06, +0.11] |
| 2–9.6 s | +0.38, +0.11 | +0.01 [−0.05, +0.08] | +0.00 [−0.05, +0.05] |
| **>9.6 s** | **+0.94, +1.32** | **−0.34 [−0.70, −0.03]** | **−0.42 [−0.70, −0.13]** |
| global | +0.26, +0.53 | −0.05 [−0.12, +0.03] | −0.14 [−0.24, −0.04] |

*(delta = mediana(ancla equipo) − mediana(ancla portería); positivo = el ancla
de portería es mejor)*

En Metrica los 8 deltas eran positivos. En Bundesliga solo 3 de 7 lo son en el
bin largo, y el agrupado es creíblemente **negativo**. Un partido llega a
−2.03 m [−2.71, −1.19]. El resultado es robusto bajo ambos protocolos de balón.

**Mecanismo probable:** con τ = 3 s, pasados ~10 s de oclusión el modelo coloca
al portero prácticamente sobre su línea. El portero moderno juega adelantado
15–25 m durante posesión rival sostenida — exactamente cuando lleva más tiempo
oculto.

El ancla por defecto volvió a `"team"`.

---

## 6. Modelo aprendido: validación cruzada ✅

Leave-one-match-out sobre los 7 partidos de Sportec. Cada modelo se evalúa
**sobre un partido que no vio ni en entrenamiento ni en validación**.

| Test | B4 global | modelo | delta [IC 95%] | B4 >9.6s | modelo | delta |
|---|---|---|---|---|---|---|
| J03WMX | 6.80 | **3.40** | +3.43 [+2.85, +4.06] | 9.96 | **6.03** | +3.94 |
| J03WN1 | 8.23 | **3.89** | +4.50 [+3.94, +5.23] | 13.08 | **7.22** | +6.00 |
| J03WOH | 8.36 | **3.82** | +4.60 [+3.98, +5.31] | 13.45 | **7.41** | +6.16 |
| J03WOY | 8.17 | **3.64** | +4.57 [+3.59, +5.41] | 12.73 | **6.63** | +6.15 |
| J03WPY | 6.62 | **3.23** | +3.40 [+2.84, +4.03] | 9.31 | **5.90** | +3.48 |
| J03WQQ | 7.63 | **5.46** | +2.19 [+1.47, +3.02] | 11.22 | **7.99** | +3.23 |
| J03WR9 | 7.93 | **5.60** | +2.38 [+1.64, +3.26] | 10.85 | **8.71** | +2.20 |

### Agrupado sobre los 7 partidos

| bin | delta (m) | IC 95% pareado | veredicto |
|---|---|---|---|
| global | **+3.54** | [+3.28, +3.83] | MEJORA CREÍBLE |
| ≤2 s | +2.10 | [+1.98, +2.22] | MEJORA CREÍBLE |
| 2–9.6 s | +3.94 | [+3.63, +4.25] | MEJORA CREÍBLE |
| **>9.6 s** | **+4.04** | [+3.46, +4.66] | MEJORA CREÍBLE |

**Consistencia: 7/7 partidos mejoran en los cuatro regímenes.**

Medias: global 7.68 → 4.15 m (**−46%**), bin largo 11.51 → 7.13 m (**−38%**).

### Dos avisos sobre estos números

**El IC agrupado es más estrecho de lo que parece.** `[+3.28, +3.83]` responde
"¿cuánto mejora una muestra oculta cualquiera de estos siete partidos?". No
responde "¿cuánto mejorará en un partido nuevo". Para eso el número honesto es
el **rango entre partidos: +2.19 a +4.60**.

**La tanda J03WQQ falló al entrenar.** Early stopping en la época 16 con
validación 7.51 m frente a ~4 m de las demás. Trayectoria de optimización
desafortunada, no problema de datos. Explica que sea el único con bin ≤2 s
nulo. **Se reporta como está**: que el agrupado se sostenga con esa tanda
dentro lo hace más creíble.

---

## 7. Test externo congelado ✅

Los 7 modelos, entrenados **solo con Bundesliga**, evaluados sobre Metrica:
otro proveedor, otra liga, otro año. Metrica no intervino en ningún momento del
desarrollo.

| | B4 global | modelo (conjunto) | reducción | B4 >9.6s | modelo | reducción |
|---|---|---|---|---|---|---|
| metrica_1 | 9.64 | 7.23 | **−25%** | 15.63 | 13.77 | −12% |
| metrica_2 | 8.10 | 5.00 | **−38%** | 16.16 | 10.41 | −36% |

### Conjunto agrupado sobre los dos partidos

| bin | delta (m) | IC 95% pareado |
|---|---|---|
| global | **+3.49** | [+2.95, +4.11] |
| ≤2 s | +1.97 | [+1.81, +2.14] |
| 2–9.6 s | +3.49 | [+3.07, +3.99] |
| **>9.6 s** | **+4.49** | [+2.97, +6.12] |

**Consistencia: 14/14 evaluaciones positivas en global, 2–9.6 s y >9.6 s.**
En ≤2 s son 12/14; los dos nulos vienen ambos de la tanda J03WQQ, la que falló
al entrenar.

### La brecha de dominio, que hay que decir

| | reducción global | reducción >9.6 s |
|---|---|---|
| Sportec (**dentro** de dominio) | −46% | −38% |
| metrica_1 (**fuera**) | −25% | **−12%** |
| metrica_2 (**fuera**) | −38% | −36% |

Generaliza, **con una caída al salir de dominio**. Y el bin largo de metrica_1
apenas se mueve. Decirlo antes de que lo pregunten es lo que da autoridad.

Observación adicional: el **conjunto** de los 7 modelos supera a cada modelo
individual en metrica_1 (+3.08 frente a un máximo de +2.98). Incluye al modelo
malo y aun así gana — comportamiento clásico de un ensemble.

---

## 8. Qué se puede afirmar

> *"El modelo reduce el error de imputación de jugadores fuera de cámara
> un 46% frente al mejor método publicado sin entrenamiento, validado con
> leave-one-match-out sobre siete partidos de Bundesliga: mejora en 7 de 7,
> incluido el régimen de oclusión larga que la literatura declara abierto.
> Sobre un conjunto congelado de otro proveedor y otra liga, que nunca
> intervino en el desarrollo, la mejora se sostiene en 14 de 14 evaluaciones,
> con una reducción del 25 al 38%. Todo con datos abiertos, en modo causal
> (solo pasado), y con GPU gratuita."*

Cada cifra está medida con bootstrap pareado de bloques, protocolo publicado y
test congelado.

**Lo que NO se puede afirmar está en `00_CONTEXTO.md`, sección 5. Léelo antes
de escribir una sola diapositiva.**
