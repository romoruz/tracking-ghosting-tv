# 00 · Contexto del proyecto

> **Este documento está escrito para alguien —persona o IA— que llega al
> proyecto sin haberlo visto.** Léelo entero antes que cualquier otro. Contiene
> el porqué, el estado real, y sobre todo **qué NO se puede afirmar**, que es lo
> que más caro sale reconstruir.

---

## 1. Qué hace este proyecto

Reconstruye las posiciones de los jugadores de fútbol que la cámara de
televisión **no muestra**, a partir de los que sí muestra.

La cámara principal de un broadcast paneá siguiendo al balón y enseña entre 10
y 16 de los 22 jugadores en cualquier instante. Toda métrica espacial calculada
sobre ese subconjunto —control de cancha, compacidad del bloque, valor del
espacio— está sesgada por dónde apuntó el camarógrafo, no por lo que pasó en el
campo.

El proyecto **mide ese sesgo y lo corrige**.

## 2. Por qué existe

Es la base técnica de una propuesta comercial: tres estudiantes del ITAM
ofreciendo servicios de análisis a un club de fútbol mexicano. La cadena
lógica del pitch es:

1. Los datos de eventos (StatsBomb, Opta) pierden todo lo que pasa entre el
   saque y el remate de un córner.
2. Un intento previo de modelar defensa a balón parado con datos de eventos
   (el proyecto **xDefense**) demostró que **el dato no alcanza**: ningún
   equipo se distingue del azar. Ese resultado negativo es el argumento.
3. La solución es tracking propio extraído de video, y su cuello de botella son
   los jugadores fuera de cuadro.
4. Este proyecto ataca ese cuello de botella y demuestra que se puede resolver
   con datos abiertos y hardware gratuito.

**El objetivo comercial no es un modelo perfecto. Es una demostración creíble
de capacidad técnica que justifique pedirle datos y financiamiento a un club.**

## 3. Linaje científico

El problema no lo inventamos. Hay tres referencias que hay que conocer:

| Trabajo | Aporta | Limitación |
|---|---|---|
| **Le, Carr, Yue & Lucey (2017)**, MIT Sloan | Define el "ghosting" por aprendizaje por imitación | Datos privados |
| **Omidshafiei et al. (2022)**, *Sci. Rep.* 12:8638 (DeepMind + Liverpool) | Graph Imputer: graph networks + VAE, ventanas de 9.6 s | 105 partidos privados de la Premier; **bidireccional**, no aplicable online |
| **Choi (2026)**, arXiv:2607.11548 | Primer benchmark **abierto y reproducible**; escalera de heurísticas B0–B5 | Sin componente aprendido; 3 partidos de un solo proveedor |

**Este repositorio replica el benchmark de Choi y le añade el modelo aprendido
que él identifica como trabajo pendiente.**

## 4. Estado: qué está establecido

Todo lo de esta tabla está medido con bootstrap pareado de bloques y protocolo
publicado. Ninguna cifra es una estimación ni una extrapolación.

| # | Resultado | Estado |
|---|---|---|
| 1 | Réplica independiente del benchmark de Choi sobre Metrica | ✅ 5/5 métricas en rango |
| 2 | Generalización a Bundesliga (Sportec, 7 partidos) | ✅ orden de mérito conservado |
| 3 | Partición del portero (ancla a portería) | ❌ **resultado negativo**: no generaliza |
| 4 | El balón muerto desplaza el peso de los bins 9.5 pp | ✅ nota metodológica propia |
| 5 | Modelo residual aprendido, CV leave-one-match-out | ✅ 7/7 partidos mejoran |
| 6 | Test externo congelado (Metrica, otro proveedor y liga) | ✅ 14/14 evaluaciones mejoran |

### Las cifras de titular

```
Bundesliga (7 partidos, leave-one-match-out, dentro de dominio):
    error global    7.68 m -> 4.15 m   (-46%)   delta agrupado +3.54 [+3.28, +3.83]
    bin >9.6 s     11.51 m -> 7.13 m   (-38%)   delta agrupado +4.04 [+3.46, +4.66]
    consistencia:  7/7 partidos mejoran en los cuatro regímenes

Metrica (2 partidos, CONGELADOS, otro proveedor y liga, fuera de dominio):
    metrica_1      9.64 m -> 7.23 m   (-25%)
    metrica_2      8.10 m -> 5.00 m   (-38%)
    delta agrupado del conjunto: +3.49 [+2.95, +4.11]
    consistencia:  14/14 evaluaciones positivas en global y 2-9.6 s
```

## 5. Qué NO se puede afirmar

Esta sección es la más importante del documento. Cada punto viene de un error
que se cometió y se corrigió.

**No digas "destrozamos el estado del arte".** Se superó una heurística *sin
entrenamiento* por un 46% dentro de dominio. Es sólido y no necesita
superlativos. Un revisor que conozca el campo castigará la exageración
descontando todo lo demás.

**No digas que DeepMind "no lo resolvió".** Publicaron un modelo bidireccional
a propósito, con 105 partidos privados. Resolvieron un problema distinto (y más
fácil: interpolación en vez de extrapolación).

**No presentes números bidireccionales como tiempo real.** Todo lo medido aquí
es **causal**: el frame t solo usa observaciones de t' ≤ t. El modo
bidireccional existe en el código (`--bidirectional`) pero **nunca se ha
reportado**. Mezclarlos invalidaría la comparación.

**No compares cifras absolutas entre Sportec y Metrica.** Metrica no expone
`ball_state`, así que su evaluación incluye balón parado por necesidad; la de
Sportec lo filtra. Las comparaciones válidas son *modelo contra B4 dentro del
mismo conjunto*.

**No omitas la brecha de dominio.** Dentro de dominio la mejora es del 46%;
fuera, del 25–38%. Y el bin largo de `metrica_1` apenas se mueve (−12%). Decir
esto antes de que lo pregunten es lo que da autoridad.

**No omitas que la tanda J03WQQ falló al entrenar.** Early stopping en la época
16 con validación 7.51 m frente a ~4 m de las demás. Es una trayectoria de
optimización desafortunada. Que el resultado agrupado se sostenga *con* esa
tanda dentro lo hace más creíble, no menos.

**No presentes el pitch control como probabilidad calibrada.** La
implementación es una simplificación cinemática del modelo de Spearman (2017),
sin integrar sobre la trayectoria del balón. Vale para comparar el mismo
instante con distintos conjuntos de jugadores —el sesgo es idéntico en los tres
paneles y se cancela— no para valores absolutos.

## 6. Restricciones del entorno

- **Laptop:** Dell Inspiron 14 5410, i7-1165G7 (4 núcleos / 8 hilos, 15 W),
  32 GB RAM, gráficos Intel Iris Xe → **no hay CUDA**. Los pasos 0–2 corren
  bien en CPU; el entrenamiento no (throttling térmico: ~19 min/época contra
  los ~6 s de una T4).
- **GPU:** Kaggle, 30 h/semana de T4. La validación cruzada completa son
  ~2 horas.
- **Datos:** todos abiertos y redistribuibles. No hay datos de ningún club.

## 7. Mapa de la documentación

| Documento | Contenido |
|---|---|
| `00_CONTEXTO.md` | Este archivo. Empezar aquí. |
| `01_arquitectura.md` | Estructura del código, flujo de datos, decisiones de diseño |
| `02_formalismo_matematico.md` | El problema, los estimadores, las pérdidas |
| `03_modelo_de_camara.md` | Simulador de viewport y su calibración |
| `04_protocolo_evaluacion.md` | Métricas, estratificación, bootstrap pareado |
| `05_roadmap.md` | Qué falta |
| `06_resultados.md` | **Todos los números, con sus intervalos** |
| `07_modelo_aprendido.md` | El imputador residual en detalle |
| `08_como_correr.md` | Runbook completo, local y en Kaggle |
| `09_decisiones_y_errores.md` | **Bitácora de bugs y por qué el código es así** |

Si eres una IA y solo puedes leer dos: este y `09_decisiones_y_errores.md`.
El segundo contiene los errores que ya se cometieron, y evitar repetirlos vale
más que cualquier explicación de lo que funciona.
