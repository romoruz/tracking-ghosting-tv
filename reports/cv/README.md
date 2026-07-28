# Modelos de la validación cruzada

Siete modelos, uno por tanda de leave-one-match-out sobre Sportec. Cada
`fold_<MATCH>.pt` fue entrenado **sin ver** el partido `<MATCH>`, que actúa
como su conjunto de test.

## Partición de cada tanda

| Checkpoint | test | validación | entrenamiento |
|---|---|---|---|
| `fold_J03WMX.pt` | J03WMX | J03WN1 | los otros 5 |
| `fold_J03WN1.pt` | J03WN1 | J03WOH | los otros 5 |
| `fold_J03WOH.pt` | J03WOH | J03WOY | los otros 5 |
| `fold_J03WOY.pt` | J03WOY | J03WPY | los otros 5 |
| `fold_J03WPY.pt` | J03WPY | J03WQQ | los otros 5 |
| `fold_J03WQQ.pt` | J03WQQ | J03WR9 | los otros 5 |
| `fold_J03WR9.pt` | J03WR9 | J03WMX | los otros 5 |

El partido de validación nunca es el de test: sesgaría la selección de modelo
hacia el conjunto de evaluación.

## Configuración (idéntica en las siete)

```
ventana 50 frames (10 s a 5 fps)   stride 25   lote 64
dim 128   bloques 4   cabezas 4    863,618 parámetros
modo CAUSAL (online)               W = 44 m   primeros 45 min
AdamW lr 3e-4 wd 1e-4              ReduceLROnPlateau (factor 0.5, paciencia 3)
early stopping paciencia 12 sobre la mediana global
semilla 1000 + índice de tanda
```

## Qué contiene cada archivo

```python
ck = torch.load('fold_J03WR9.pt', map_location='cpu', weights_only=False)
ck['state_dict']     # pesos
ck['args']           # configuración COMPLETA del entrenamiento
ck['epoch']          # época alcanzada
ck['val_err_m']      # error de validación del mejor checkpoint
ck['b4_val_err_m']   # el piso de B4 en ese mismo conjunto
ck['history']        # curva época a época
```

Los scripts de evaluación leen `args` del checkpoint, así que no hay que
recordar con qué configuración se entrenó cada uno.

## Advertencia

La tanda **J03WQQ falló al entrenar**: early stopping en la época 16 con
validación 7.51 m, frente a ~4 m de las demás. Es una trayectoria de
optimización desafortunada, no un problema de datos. Se conserva a propósito —
el resultado agrupado se sostiene con ella dentro, y eso lo hace más creíble.
Ver `docs/06_resultados.md` §6.
