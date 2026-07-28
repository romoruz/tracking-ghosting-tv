# 08 · Cómo correr todo

## 0. Instalación

```bash
tar xzf ghosting-v0.8.0.tar.gz
cd ghosting
./setup.sh              # entorno base, pasos 0-2 (todo CPU)
./setup.sh --torch      # además PyTorch CPU, para el paso 3
source venv/bin/activate
./run.sh test           # 72 tests, ~40 s
```

Probado en Arch Linux, Python 3.10–3.14. Arch aplica PEP 668, por eso el script
obliga a usar un venv. No instales nada con `--break-system-packages`.

Si tienes GPU con CUDA, sustituye el build CPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 1. Verificación sin red

```bash
./run.sh demo
```

Genera partidos sintéticos y corre los pasos 0–2 en ~1 min. Si esto funciona,
el entorno está bien.

> Los datos sintéticos **no son un resultado**. Los autómatas mantienen sus
> offsets de rol con varianza casi nula, así que B4 obtiene un error
> artificialmente bajo (~3 m frente a los 9.7–11.6 m publicados). Lo que sí se
> conserva es el **orden de mérito** entre métodos.

## 2. Descarga de datos reales

```bash
python scripts/00_download.py --provider sportec    # 7 partidos de Bundesliga
python scripts/00_download.py --provider metrica    # 2 partidos, test congelado
```

Pasa por HuggingFace vía kloppy; la primera vez tarda varios minutos por
partido. Los `.npz` quedan en `data/processed/`.

> `--force` es **obligatorio** si cambias el cargador: `is_gk` y las
> coordenadas se guardan dentro del `.npz`.

## 3. Pipeline completo, paso a paso

### Paso 1 — validar el simulador de cámara

```bash
python scripts/01_viewport_report.py
```

Con `W = 44 m` deben verse **entre 14 y 16 jugadores de 22**. El script compara
automáticamente contra los valores publicados y marca desviaciones. Si esto
falla, las coordenadas están mal y nada de lo que venga después sirve.

### Paso 2 — escalera de baselines

```bash
python scripts/02_run_baselines.py --minutes 45 --boot 1000
```

El orden de mérito debe ser `B4 < B3V < B2 < B3 < B3E < B1, B5`. Si tu B4 no
gana, la implementación está mal.

Banderas relevantes:
- `--minutes 45` — replica el protocolo de Choi (evalúa el primer tiempo)
- `--include-dead-ball` — necesario para comparar Sportec con las cifras de Choi
- `--include-gk`, `--gk-anchor {team,goal}` — ver `03_gk_ablation.py`

### Paso 2b — ablación del portero

```bash
python scripts/03_gk_ablation.py --minutes 45 --boot 1000
```

Tres configuraciones y el delta pareado entre ellas. **Lee primero la columna
`peso>9.6s`**: es estructural, no de rendimiento. Solo dentro de la
configuración cuyo peso coincide con el publicado son comparables los errores.

### Paso 3 — entrenar el modelo

**En CPU no es viable** (~19 min/época con throttling). Usa GPU.

```bash
# Medir SIEMPRE antes de una corrida larga
python scripts/04_train.py --epochs 100 --dry-run 60 --device cuda

# Entrenar
python scripts/04_train.py --epochs 100 --batch 64 --device cuda \
    --monitor global --patience 12
```

Banderas útiles:
- `--monitor {global,<=2s,2-9.6s,>9.6s}` — qué métrica dirige checkpoint y early stopping
- `--resume` — continúa desde el último checkpoint
- `--quick` — preset rápido para una primera señal
- `--long` — ventana de 30 s (implementado, nunca ejecutado con éxito)
- `--accumulate K` — acumulación de gradientes (no hace falta con 32 GB)

### Paso 4 — validación cruzada (el resultado que vale)

```bash
python scripts/07_cross_validate.py --epochs 100 --device cuda \
    --batch 64 --monitor global --patience 12
```

Siete tandas, ~2 h en T4. Rota qué partido hace de test; el de validación nunca
es el de test. Produce el delta agrupado y la tabla de consistencia.

### Paso 5 — test externo congelado

```bash
python scripts/08_external_test.py --device cuda --boot 1000
```

Evalúa los 7 modelos sobre Metrica (otro proveedor, otra liga). Reporta las 14
evaluaciones por separado **y** un conjunto (promedio de predicciones) agrupado
sobre los dos partidos.

### Paso 6 — la figura para la presentación

```bash
python scripts/09_figure_panels.py --match J03WR9 --n-frames 4
python scripts/09_figure_panels.py --match metrica_1 --n-frames 4
```

Tres paneles: verdad / lo que ve la cámara / con fantasmas. Corre en CPU en
minutos. Elige automáticamente los frames donde más se nota el efecto.

---

## 4. Kaggle (GPU gratuita)

30 h/semana de T4, sesiones de hasta 9 h. Requiere **verificación telefónica**
en `kaggle.com/settings`, si no los aceleradores salen en gris.

### Preparar el paquete

```bash
python scripts/06_export_bundle.py --split
```

Genera dos zips: `ghosting_data.zip` (~15 MB, subes una vez) y
`ghosting_code.zip` (~200 KB, en cada iteración). Súbelos como **dos datasets
privados**: `ghosting-data` y `ghosting-code`.

> Los datos ya procesados a 5 fps y 45 min ocupan 5–10× menos que los `.npz`
> completos. **No descargues Sportec dentro de la sesión de GPU**: gastarías
> 15–30 min de cómputo alquilado en red y parseo XML que ya hiciste en casa.

### Configuración de la sesión

| Ajuste | Valor |
|---|---|
| Accelerator | GPU T4 ×2 |
| Persistence | **Files only** (si no, pierdes el checkpoint al cortarse) |
| Internet | Off (no hace falta) |
| Environment | Pin to original |

### Celda de arranque

```python
import shutil, os, sys, glob

datos  = glob.glob('/kaggle/input/**/data/processed', recursive=True)
codigo = glob.glob('/kaggle/input/**/scripts/04_train.py', recursive=True)
assert datos and codigo, f'falta algo: {os.listdir("/kaggle/input")}'

# Preservar reports/ al reconstruir: contiene checkpoints que cuestan horas.
if os.path.exists('/kaggle/working/gh/reports'):
    shutil.move('/kaggle/working/gh/reports', '/kaggle/working/_reports_tmp')
shutil.rmtree('/kaggle/working/gh', ignore_errors=True)

shutil.copytree(os.path.dirname(os.path.dirname(codigo[0])), '/kaggle/working/gh')
shutil.copytree(datos[0], '/kaggle/working/gh/data/processed', dirs_exist_ok=True)

if os.path.exists('/kaggle/working/_reports_tmp'):
    shutil.move('/kaggle/working/_reports_tmp', '/kaggle/working/gh/reports')

os.chdir('/kaggle/working/gh')
sys.path.insert(0, '/kaggle/working/gh/src')
print('partidos:', sorted(os.listdir('data/processed')))
```

### Guardar resultados antes de que muera la sesión

```python
import shutil
shutil.make_archive('/kaggle/working/checkpoints_cv', 'zip', 'reports/cv')
shutil.make_archive('/kaggle/working/tablas', 'zip', 'reports/tables')
# Panel derecho > Output > descargar AHORA
```

### Corridas largas

Usa **Save Version → Save & Run All (Commit)**. Corre en los servidores de
Kaggle, sobrevive a que cierres el navegador o suspendas el portátil. Una
sesión interactiva **no** sobrevive a la suspensión.

Si prefieres interactivo, en Arch:

```bash
systemd-inhibit --what=idle:sleep:handle-lid-switch \
  --why="entrenamiento en Kaggle" sleep 4h
```

Y enchufa el cargador.

---

## 5. Estructura de salidas

```
reports/
├── cv/                          checkpoints de la validación cruzada
│   └── fold_<MATCH>.pt          (7 archivos, ~3.5 MB cada uno)
├── tables/
│   ├── viewport_stats.csv       estadísticas de oclusión por ancho
│   ├── ladder.csv               escalera de baselines
│   ├── gk_ablation*.csv         ablación del portero
│   ├── cross_validation.csv     resultado principal
│   ├── external_test_metrica.csv test congelado
│   └── train_history.json       curvas de entrenamiento
└── figures/
    ├── occlusion_*.png          distribución de gaps
    ├── ladder_*.png             comparativa de métodos
    ├── ghosts_*.png             cancha con fantasmas
    └── paneles_*.png            la figura de tres paneles
```

## 6. Qué contiene un checkpoint

```python
ck = torch.load('reports/cv/fold_J03WR9.pt', map_location='cpu',
                weights_only=False)
ck['state_dict']      # pesos del modelo
ck['args']            # TODA la configuración del entrenamiento
ck['epoch']           # época alcanzada
ck['val_err_m']       # error de validación
ck['b4_val_err_m']    # el piso de B4 en ese mismo conjunto
ck['history']         # curva completa, época a época
```

`args` incluye `window`, `dim`, `blocks`, `fps`, `width`, `minutes`. Los
scripts de evaluación lo leen de ahí, así que **no hay que recordar con qué
configuración se entrenó cada modelo**.
