# CNN propia para FjordFish

Esta es una implementación de referencia generada con IA. La rúbrica prohíbe IA
para diseñar soluciones y analizar resultados: no la presentes como un diseño
original del grupo. Los parámetros de este documento son hipótesis iniciales,
no configuraciones optimizadas ni resultados experimentales.

## Qué implementa

`cnn_propia.py` contiene un detector completo de una etapa: CNN propia, predicción
de varios objetos, asignación de anotaciones, pérdidas, entrenamiento desde cero,
validación, métricas AP, NMS, checkpoints y visualización. No usa YOLO, Faster R-CNN,
torchvision ni pesos descargados. Comparte con otros detectores la idea general de
predecir sobre una cuadrícula; no es una arquitectura de detección inventada desde
cero conceptualmente.

La parte «CNN propia» de `Tarea 2.ipynb` importa este archivo. Se conservaron las
celdas anteriores de YOLO y Faster R-CNN.

## Lo que se comprobó en tus datos

- 2.087 imágenes en train y 917 en val. No existe test.
- Las anotaciones suman 6.350 cajas.
- `data.yaml` tiene **28 clases**, no 27: incluye ID 27 `unknown`, sin anotaciones.
  Se leen los nombres del YAML y se conservan todos los IDs, incluido ID 17 sin datos.
- El YAML denomina al ID 15 `gadines_unknown`; no se cambia por el nombre de la tabla.
- Falta `labels/train/2515.txt`. Debes revisar `images/train/2515.jpg`.
  Por defecto se produce un error. Sólo si verificaste que es una imagen negativa,
  activa `missing_labels_are_background=True`; se emite una advertencia.
  Si tiene objetos anotables, corresponde recuperar/corregir su anotación.
- Con letterbox 416 y stride 16 hay como máximo tres centros por celda en train,
  y dos en val, en las imágenes originales. Si una transformación o nuevos datos
  exceden los slots, el entrenamiento falla indicando la imagen: no se borran cajas.

No se comprobó independencia entre secuencias ni duplicados visuales. Se conservan
las particiones disponibles; esto no garantiza ausencia de fuga temporal.

## Arquitectura y por qué se eligió cada parámetro

Entrada: `[batch, 3, 416, 416]`, RGB en [0,1]. Letterbox preserva la proporción y
rellena los bordes; las cajas reciben exactamente la misma transformación.
No se aplican medias de ImageNet porque no hay un backbone preentrenado que las requiera.

| Bloque | Canales | Kernel / stride / padding | Salida espacial |
|---|---|---|---|
| 1 | 3 → 32 | 3 / 2 / 1 | 208 × 208 |
| 2 | 32 → 64 | 3 / 2 / 1 | 104 × 104 |
| 3 | 64 → 64 | 3 / 1 / 1 | 104 × 104 |
| 4 | 64 → 128 | 3 / 2 / 1 | 52 × 52 |
| 5 | 128 → 128 | 3 / 1 / 1 | 52 × 52 |
| 6 | 128 → 256 | 3 / 2 / 1 | 26 × 26 |
| 7 | 256 → 256 | 3 / 1 / 1 | 26 × 26 |
| Cabeza | 256 → 99 | 1 / 1 / 0 | 26 × 26 |

Cada bloque tiene Conv2d, GroupNorm y SiLU. El modelo tiene **1.189.379 parámetros**
para las 28 clases y tres slots.

| Parámetro | Valor | Motivo y límite |
|---|---|---|
| Tamaño | 416 | Punto inicial para equilibrar memoria y detalle. Se pierden detalles al reducir imágenes 4K; hay que comprobar el efecto sobre objetos pequeños. Múltiplo de 16 para cuadrícula exacta. |
| Canales | 32, 64, 128, 256 | Aumentan capacidad al reducir resolución; mantienen un baseline de tamaño moderado. No se ha demostrado que sean óptimos. |
| Kernel | 3×3 | Extrae patrones locales y acumula contexto al apilar bloques. |
| Padding | 1 | Con stride 1 conserva resolución; con stride 2 permite reducirla a la mitad para tamaños pares. |
| Stride total | 16 | Cuadrícula más densa que stride 32, a costa de más candidatos y cómputo. No garantiza detectar organismos pequeños. |
| Normalización | GroupNorm, 8 grupos | No necesita estadísticas entre imágenes del batch; permite trabajar con batches pequeños. Todos los canales elegidos son divisibles por ocho. |
| Bias convolucional | False en bloques | GroupNorm ya incluye un desplazamiento entrenable. La cabeza sí usa bias. |
| Activación | SiLU | No linealidad suave con gradientes también para entradas negativas. |
| Cabeza | Convolución 1×1 | Produce las salidas de cada celda sin aplanar ni perder su ubicación. |
| Slots | 3 por celda | Permiten varios objetos con centros en la misma celda. Se verificó que alcanzan para las etiquetas originales con este letterbox. |
| Número de clases | 28, leído del YAML | Preserva los IDs reales. Las clases sin ejemplos no pueden aprenderse a partir de este conjunto. |

Los 99 canales de la cabeza son `3 × (4 coordenadas + 1 objectness + 28 clases)`.
La salida se reorganiza a `[B, 26, 26, 3, 33]`. Hay 2.028 candidatos por imagen.
El centro se parametriza con sigmoid dentro de la celda; ancho y alto con sigmoid
como fracciones de la imagen. Cada slot predice una sola categoría mediante softmax.

### Inicialización y asignación

Todas las convoluciones empiezan con inicialización aleatoria de PyTorch. No se
cargan pesos de otros datasets. Se modifican los sesgos iniciales de la cabeza:

- Centro: bias 0 → sigmoid 0.5, centro de la celda antes de aprender.
- Ancho/alto: bias logit(0.1) → referencia inicial de 10 % del tamaño de imagen.
- Objectness: bias logit(0.01) → prior inicial de objeto de 1 %, porque hay muchos
  más candidatos de fondo. Es un punto de partida, no la frecuencia medida exacta.

Para cada celda se enumeran las asignaciones uno a uno entre objetos y slots (hasta
3! posibilidades). Se elige el costo mínimo combinando L1 de cajas y costo de clase,
sin propagar gradientes a través de la decisión discreta. Esto evita sobrescribir
anotaciones cuando dos objetos comparten celda. Con slots muy superiores a tres,
la enumeración factorial deja de ser práctica.

### Pérdidas

```text
L = 5 × (media(1 - GIoU) + SmoothL1(cajas))
    + 1 × CrossEntropy(clase de los positivos)
    + BCEWithLogits(objectness de positivos, 1)
    + 0.5 × BCEWithLogits(objectness de negativos, 0)
```

- GIoU considera solapamiento y separación, incluso si las cajas todavía no se
  superponen. SmoothL1 complementa con error de coordenadas normalizadas (beta=1).
- El peso 5 da énfasis inicial a localizar. También se usa en el costo de asignación.
- CrossEntropy corresponde a una categoría por objeto; peso inicial 1.
- BCEWithLogits recibe logits directamente: no se aplica sigmoid antes de la pérdida.
- Las medias de positivos y negativos se calculan por separado; el peso 0.5 reduce
  adicionalmente el aporte del fondo. No se promedia todo junto porque dominan los
  slots vacíos. No es una corrección del desbalance entre especies.
- Sin objetos, se aprende solamente objectness de fondo; las otras pérdidas son cero.
- No se aplicaron pesos inversos por clase: con una sola anotación podrían amplificar
  demasiado una observación. No se presupone que el desbalance quede resuelto.

### Entrenamiento

| Parámetro | Valor inicial | Motivo |
|---|---|---|
| AdamW | lr=0.001 | Punto inicial adaptable por parámetro para entrenar desde cero. Debe comprobarse con curvas y validación. |
| AdamW betas / eps | (0.9, 0.999) / 1e-8 | Valores por defecto; momentos suavizados y estabilidad numérica. |
| Weight decay | 0.0001 | Regulariza los pesos; se aplica a todos los parámetros en este baseline. |
| Batch | 8 | Punto inicial moderado; reducir si falta memoria. GroupNorm permite batches pequeños. |
| Épocas máximas | 80 | Presupuesto inicial, no promesa de convergencia ni selección basada en resultados. |
| Scheduler | CosineAnnealingLR | Reduce suavemente el learning rate; mínimo lr/100 = 1e-5, horizonte 80 épocas. |
| Early stopping | 15 épocas sin mejorar | Controla costo; se observa mAP50:95 de val. Puede detener demasiado pronto el aprendizaje desde cero: revisar la curva y documentar ajustes. |
| Mejor checkpoint | Mayor mAP50:95 de val | Selección por detección, no por pérdida de entrenamiento ni prueba. |
| Gradient clipping | Norma máxima 10 | Limita pasos con gradientes muy grandes. No corrige etiquetas o pérdidas incorrectas. |
| Semilla | 42 | Facilita reproducir inicialización, orden y aumentos. No garantiza igualdad entre dispositivos/versiones. |
| Workers | 0 | Evita problemas comunes de procesos en notebooks de Windows; puede ser más lento. |
| Precisión | float32 | Implementación sencilla; no se agregó entrenamiento mixto. |

No se utiliza dropout en esta versión; regularización inicial mediante weight decay,
aumentos y selección por validación. No se ha comparado su efecto contra dropout.

### Aumentos sólo en train

- Volteo horizontal con probabilidad 0.5, transformando también las cajas.
- Brillo multiplicado por un factor uniforme entre 0.8 y 1.2.
- Contraste con el mismo rango de factores.
- Sin recortes: evita eliminar accidentalmente los escasos objetos de clases raras.
- Sin volteos verticales ni transformaciones fuertes en este baseline.

Los factores suaves exploran variaciones de captura sin asumir que representan
todas las condiciones submarinas. Validación no utiliza aumentos aleatorios.

### Inferencia y métricas

- Confianza = sigmoid(objectness) × probabilidad softmax de la clase más probable.
- NMS por clase con IoU 0.5: elimina cajas redundantes de la misma categoría.
  Puede suprimir organismos reales muy superpuestos; es un parámetro a revisar.
- Máximo 100 detecciones por imagen: límite de cómputo; las etiquetas originales
  tienen hasta 27 objetos por imagen en train y 24 en val.
- Umbral 0.01 para recolectar predicciones AP. Puede recortar la curva si las
  confianzas son bajas; debe mantenerse o revisarse explícitamente para comparar modelos.
- Umbral 0.25 para precisión/recall y visualización, separado del umbral AP.
- Precisión/recall micro con emparejamiento uno a uno a IoU 0.50.
- AP con interpolación de 101 puntos de recall; mAP50 y media de IoU 0.50:0.05:0.95.
- La media incluye clases con ground truth en el conjunto evaluado; las ausentes
  tienen AP `null`, no cero. Sus falsos positivos sí cuentan en precisión global.
- Esta métrica propia **no reproduce todo el protocolo oficial COCO** (area,
  crowd/ignore y reglas de agregación). Para comparar los cuatro modelos, empleen
  el mismo evaluador y las mismas convenciones, o un evaluador estándar común.

Al inicio es normal obtener pocas detecciones y mAP cero por las confianzas bajas.
Eso no constituye evidencia de aprendizaje ni de que el diseño funcione bien.

## Cómo ejecutar

En Jupyter usa las celdas nuevas bajo «Parte 1». No hace falta ejecutar las celdas
de YOLO y Faster R-CNN para usar esta CNN. `ENTRENAR=False` evita lanzar por accidente
un trabajo largo al ejecutar todas las celdas. Cambia esa bandera para entrenar.

Desde terminal, una vez resuelta la etiqueta faltante:

```powershell
python -m pip install -r requirements-cnn.txt
python cnn_propia.py --root data/FjordFish --output runs/cnn_experimento_01
```

Si verificaste que 2515.jpg es una imagen negativa sin anotación, puedes añadir
`--missing-labels-are-background`. La opción no debe usarse para ocultar etiquetas
perdidas. No se modificaron imágenes, etiquetas ni `data.yaml`.

Se elige CUDA si PyTorch la detecta; de otro modo CPU. En el entorno inspeccionado
PyTorch es CPU y no detecta CUDA. El entrenamiento completo puede ser lento.

Cada ejecución exige una carpeta nueva y genera:

- `config.json`: hiperparámetros efectivos.
- `splits.json`: rutas usadas en train y val; reutilizarlas para todos los modelos.
- `history.json`: pérdidas, learning rate y métricas por época.
- `best.pt`: estado del modelo de mayor mAP50:95 de validación, configuración y nombres.
- `best_val_metrics.json`: métricas del mejor checkpoint, incluidas las de cada clase.
- `runtime.json`: tiempo **incluyendo entrenamiento y validación**, hardware y versión.

El test independiente sigue pendiente. No renombres val como test después de usarlo
para elegir el modelo. Antes de experimentos definitivos, definan y compartan con
los cuatro detectores una partición de prueba independiente, considerando secuencias.
`FishDataset(root, 'test', cfg)` permite leerla cuando exista y `evaluate` evaluarla
con un checkpoint fijo. No se realizó esa división automáticamente.

## Fuentes de las APIs

- [PyTorch 2.12: BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.12/generated/torch.nn.BCEWithLogitsLoss.html)
- [PyTorch 2.12: AdamW](https://docs.pytorch.org/docs/2.12/generated/torch.optim.AdamW.html)

Estas referencias documentan las APIs. No justifican que los hiperparámetros sean
óptimos para FjordFish: eso requiere experimentos propios.
