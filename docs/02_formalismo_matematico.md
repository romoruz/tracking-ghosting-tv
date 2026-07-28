# Formalismo matemático

## 1. El problema

Sea un partido con $N$ jugadores. La posición del jugador $i$ en el instante
$t$ es $p_i(t)\in\mathbb{R}^2$, en coordenadas de cancha $[0,L]\times[0,A]$ con
$L=105$ y $A=68$ metros. El estado completo es el tensor

$$X\in\mathbb{R}^{T\times N\times 2},\qquad X[t,i,:]=p_i(t)$$

La cámara induce una **máscara de observación** $M\in\{0,1\}^{T\times N}$, donde
$M[t,i]=1$ si el jugador $i$ es visible en $t$. Lo observable es $X\odot M$
junto con $M$ y la trayectoria del balón $b(t)$, que por construcción de la
cámara siempre es visible.

**El problema de imputación consiste en estimar la distribución condicional**

$$p\big(X_{\text{miss}}\;\big|\;X_{\text{obs}},\,M,\,b\big)$$

Es una **distribución**, no un punto. Un defensor oculto veinte segundos puede
estar en varios lugares plausibles, y un modelo honesto expresa esa
incertidumbre. Los pasos 2 y 3 producen estimadores puntuales (la moda o la
media condicional); el paso 4 aborda la distribución completa.

### 1.1 Dos regímenes

La información temporal disponible parte el problema en dos:

| Régimen | Observaciones | Naturaleza | Uso |
|---|---|---|---|
| **Causal (online)** | $t'\le t$ | Extrapolación | Producto en vivo |
| **Bidireccional (offline)** | $t'\in[t-H,\,t+H]$ | Interpolación | Análisis post-partido |

El bidireccional es sustancialmente más fácil: si sabes dónde reapareció el
jugador, el problema se reduce a unir dos puntos con una trayectoria plausible.
El Graph Imputer de DeepMind es bidireccional con $H$ correspondiente a
ventanas de 9.6 s.

**Nunca reportes números bidireccionales para un producto en tiempo real.**
Son problemas distintos y mezclarlos invalida la comparación. Este repositorio
implementa el régimen causal; el bidireccional se añadirá como variante
explícitamente etiquetada.

### 1.2 Población puntuable

No todo par (frame, jugador) oculto es evaluable. Definimos el **gap de
oclusión** $\Delta_i(t)$ como el tiempo transcurrido desde la última
observación del jugador $i$ dentro del periodo actual. La población puntuable es

$$\mathcal{S}=\Big\{(t,i)\;:\;\text{on\_pitch}_i(t)=1,\;M[t,i]=0,\;\Delta_i(t)<\infty\Big\}$$

Se excluyen tres casos:

- **Fuera de cancha** (suplentes, expulsados, descanso): no existen.
- **Cold start** ($\Delta_i(t)=\infty$, nunca visto en el periodo): ningún
  método causal puede posicionarlos, porque no hay ninguna observación previa
  de la cual partir. Puntuarlos penalizaría a todos por igual sin discriminar.
- **Balón muerto** (opcional, activado por defecto): en balón parado la cámara
  hace primeros planos y repeticiones que el modelo de viewport no representa.

El reinicio del gap **por periodo** es deliberado: tras el descanso los equipos
cambian de lado, así que una observación del primer tiempo no informa sobre la
posición en el segundo.

---

## 2. Estimadores sin entrenamiento (paso 2)

Sea $V_t$ el conjunto de jugadores visibles del equipo en cuestión en el
instante $t$, y $\bar p_{V}(t)$ su centroide.

### B1 — última vista con decaimiento

$$\hat p_j(t)=w\,p_j^{\text{last}}+(1-w)\,\bar p_V(t),\qquad w=e^{-\Delta_j(t)/\tau},\ \ \tau=8\ \text{s}$$

### B2 — ancla de formación

Mientras $i$ es visible se almacena su desplazamiento de rol
$\text{off}_i=p_i-\bar p_V$. Al ocultarse, $\hat p_j(t)=\bar p_V(t)+\text{off}_j$.

Es el salto más grande de toda la escalera, y la razón es sencilla: la mayor
parte de la posición de un jugador oculto se explica por *"el equipo se
desplazó y el jugador conservó su rol"*.

### B4 — voto de centroide anclado a roles

El problema de B2 es que $\bar p_V(t)$ **está sesgado**. Cuando la cámara
apunta a la izquierda, los jugadores visibles de un equipo son precisamente los
que están a la izquierda, así que su promedio queda a la izquierda del
centroide verdadero. Cada offset almacenado hereda ese sesgo.

B4 lo atenúa por votación: cada jugador visible propone dónde estaría el
centroide **completo** restándose su propio desplazamiento de rol.

$$\hat c(t)=\frac{1}{|V_t|}\sum_{i\in V_t}\Big(p_i(t)-\text{off}_i(t^-)\Big)$$

$$\text{off}_i(t)\;\leftarrow\;\text{EMA}\Big[p_i(t)-\hat c(t)\Big],\qquad i\in V_t$$

donde $\text{off}_i(t^-)$ es el valor **antes** de la actualización en $t$: se
vota con los offsets previos y luego se actualizan contra el centroide votado.
Es un punto fijo autoconsistente — los offsets se estiman contra $\hat c$ y
$\hat c$ se calcula a partir de esos mismos offsets. El jugador oculto se imputa
en $\hat c(t)+\text{off}_j$.

La cancelación del sesgo es exacta solo si los offsets son estables y los
votantes son condicionalmente representativos; en la práctica es atenuación,
no cancelación. Con menos de tres votantes con offset almacenado, el método
cae a B2, y de ahí a B1.

### 2.1 Partición del portero

**Todos los estimadores anteriores asumen que la posición de un jugador está
gobernada por el centroide de su equipo. Para el portero eso es falso.**

La dinámica del portero obedece a un atractor estacionario —su propia
portería— y se desacopla del centroide justo cuando el equipo ataca, que es
exactamente cuando el portero está oculto. Un modelo homogéneo lo arrastra
campo arriba y genera un error grande y sistemático.

La corrección **no** es forzar $M[t,i]=1$ para porteros. Eso falsearía la
física del broadcast (en un partido real el portero del lado lejano desaparece
de cuadro durante minutos) y rompería la comparabilidad con la literatura. La
corrección es **anclarlo distinto**:

$$\hat p_{\text{GK}}(t)=w_g\,p_{\text{GK}}^{\text{last}}+(1-w_g)\,g,\qquad w_g=e^{-\Delta(t)/\tau_g},\ \ \tau_g=3\ \text{s}$$

donde $g$ es el centro de la portería que defiende su equipo en ese periodo.
La portería defendida se estima empíricamente (la más cercana a la posición
media del portero en ese periodo), de modo que el cambio de lado en el
descanso se resuelve solo, sin depender de metadatos del proveedor.

Configurable vía `LadderConfig(gk_anchor="goal" | "team")`. Las métricas se
reportan **siempre con y sin porteros por separado**.

---

## 3. Estimador aprendido (paso 3)

### 3.1 Parametrización residual

En lugar de predecir la posición absoluta, el modelo predice la **corrección
sobre B4**:

$$\hat p_j(t)=\underbrace{\hat c(t)+\text{off}_j}_{\text{B4, forma cerrada}}\;+\;\underbrace{f_\theta\big(X_{\text{obs}},M,b\big)_j(t)}_{\text{residuo aprendido}}$$

Tres razones:

1. Con 7 partidos de datos, aprender el residuo converge mucho más rápido que
   aprender la función completa.
2. El modelo no puede ser catastróficamente peor que el baseline: si
   $f_\theta\to 0$, recuperas B4.
3. La comparación es directa e interpretable: el residuo *es* la mejora.

### 3.2 Entradas

Por jugador y frame, se alimentan coordenadas **absolutas y relativas**:

$$\Big[\;\underbrace{x_i/L,\;y_i/A}_{\text{absoluta}},\;\underbrace{x_i-x_b,\;y_i-y_b}_{\text{rel. balón}},\;\underbrace{x_i-\hat c_x,\;y_i-\hat c_y}_{\text{rel. centroide}},\;v_x,\,v_y,\;m_i,\;\Delta_i,\;\text{equipo},\;\text{rol}\;\Big]$$

Centrar en el balón mejora la generalización, pero **sustituir** las absolutas
pierde información que sí importa: distancia a la portería propia, línea de
fuera de juego, cercanía a la banda. Se alimentan ambas.

### 3.3 Equivarianza

Los jugadores de un equipo son un **conjunto**, no una lista ordenada. El
modelo debe ser equivariante a permutaciones dentro de cada equipo: reordenar
los índices de entrada debe reordenar las salidas igual.

Esto se garantiza con atención sobre el conjunto (set transformer) o con una
GNN sobre el grafo completo de jugadores. **Un MLP sobre el vector concatenado
de $2N$ coordenadas rompe la equivarianza** y hace que el modelo memorice el
orden del roster en vez de aprender la estructura del juego.

### 3.4 Función de pérdida

$$\mathcal{L}(\theta)=\mathcal{L}_{\text{rec}}+\lambda_s\,\mathcal{L}_{\text{smooth}}+\lambda_v\,\mathcal{L}_{\text{vel}}+\lambda_a\,\mathcal{L}_{\text{acc}}$$

**Reconstrucción**, solo sobre lo oculto:

$$\mathcal{L}_{\text{rec}}=\frac{1}{|\mathcal{S}|}\sum_{(t,i)\in\mathcal{S}}\big\|\hat p_i(t)-p_i(t)\big\|_2^2$$

**Suavidad** (penalización de la segunda derivada discreta):

$$\mathcal{L}_{\text{smooth}}=\frac{1}{|\mathcal{S}|}\sum_{(t,i)\in\mathcal{S}}\big\|\hat p_i(t+1)-2\hat p_i(t)+\hat p_i(t-1)\big\|_2^2$$

**Factibilidad cinemática** (bisagras sobre límites físicos):

$$\mathcal{L}_{\text{vel}}=\sum_{t,i}\Big[\max\big(0,\ \|\hat v_i(t)\|-v_{\max}\big)\Big]^2,\qquad v_{\max}\approx 11\ \text{m/s}$$

$$\mathcal{L}_{\text{acc}}=\sum_{t,i}\Big[\max\big(0,\ \|\hat a_i(t)\|-a_{\max}\big)\Big]^2,\qquad a_{\max}\approx 7\ \text{m/s}^2$$

**Por qué hacen falta las dos familias de términos.** Las bisagras tienen
gradiente **nulo** en la región factible: no penalizan nada mientras el modelo
se mantenga por debajo de $v_{\max}$. Eso deja pasar oscilación de alta
frecuencia — un jugador que tiembla a 3 m/s frame a frame es físicamente
absurdo pero no viola ninguna cota. Los transformers producen exactamente ese
artefacto. $\mathcal{L}_{\text{smooth}}$ lo castiga en todo el dominio;
$\mathcal{L}_{\text{vel}}$ y $\mathcal{L}_{\text{acc}}$ ponen el techo duro.
Se necesitan ambas.

Estas restricciones no son "PINN" en sentido estricto: no hay una ecuación
diferencial que gobierne el sistema. Son penalizaciones de factibilidad. Es
mejor llamarlas así y no inflar el vocabulario.

---

## 4. Notación

| Símbolo | Significado |
|---|---|
| $N$ | jugadores registrados en el partido (con sustituciones, $\approx 28$–32) |
| $T$ | frames |
| $p_i(t)$ | posición del jugador $i$, metros |
| $b(t)$ | posición del balón |
| $M[t,i]$ | máscara de observación de cámara |
| $V_t$ | conjunto de visibles del equipo en $t$ |
| $\bar p_V(t)$ | centroide de los visibles (sesgado) |
| $\hat c(t)$ | centroide votado (sesgo atenuado) |
| $\text{off}_i$ | desplazamiento de rol del jugador $i$ |
| $\Delta_i(t)$ | gap de oclusión, segundos |
| $\mathcal{S}$ | población puntuable |
| $W$ | ancho del viewport, metros |
| $\alpha$ | coeficiente del EMA de paneo |
