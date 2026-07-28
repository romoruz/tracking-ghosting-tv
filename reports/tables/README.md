# Tablas de resultados

Salidas de los scripts, en el orden en que se generan.

| Archivo | Script | Contenido |
|---|---|---|
| `viewport_stats.csv` | `01_viewport_report.py` | Estadísticas de oclusión por ancho de viewport |
| `ladder.csv` | `02_run_baselines.py` | Escalera B0–B5 con IC por bin |
| `gk_ablation.csv` | `03_gk_ablation.py` | Tres configuraciones de portero |
| `gk_ablation_deltas.csv` | `03_gk_ablation.py` | Deltas pareados A→B (resultado negativo) |
| `model_vs_b4.csv` | `05_evaluate_model.py` | Modelo contra B4, partición única |
| `train_history.json` | `04_train.py` | Curva de entrenamiento época a época |
| **`cross_validation.csv`** | `07_cross_validate.py` | **Resultado principal: 7 tandas + agrupado** |
| **`external_test_metrica.csv`** | `08_external_test.py` | **Test congelado: 14 evaluaciones + conjunto** |

Los dos en negrita son los que sostienen las afirmaciones del proyecto.

## Esquema de las dos tablas principales

`cross_validation.csv`: `fold, test_id, bin, b4_m, modelo_m, delta_m, ci_lo,
ci_hi, excluye_cero`. Las filas con `test_id == "AGRUPADO"` son el pooling
sobre los 7 partidos.

`external_test_metrica.csv`: `partido, modelo, bin, delta_m, ci_lo, ci_hi,
excluye_cero`. `modelo == "CONJUNTO"` es el promedio de las 7 predicciones;
`partido == "AGRUPADO"` es el pooling de los 2 partidos de Metrica.

> **No agrupes las 14 evaluaciones individuales.** Los mismos frames de Metrica
> aparecerían 7 veces y el intervalo saldría artificialmente estrecho. Por eso
> existe la fila `CONJUNTO`.
