# Tarea 2: Reconocimiento de imagenes y cosas así, probablemente se me ocurra un nombre mejor
Integrantes:
- Claudio Alvial.
- Cristóbal Navarro.
- Cristopher Vasquez.

## Entrenamiento

Instala las dependencias desde una celda con `%pip install -r requirements-entrenamiento.txt` y reinicia el kernel. Ejecuta la celda de Faster R-CNN o la de YOLO en `Tarea 2.ipynb` para entrenar ese modelo y luego generar sus detecciones.

Cada celda permite editar `EPOCAS` (10), `BATCH_SIZE` (2), `LEARNING_RATE`, `PACIENCIA` (3) y `TAMANO_IMAGEN` (640). Una epoca recorre todas las imagenes de entrenamiento. Se usa GPU si esta disponible; en caso de falta de memoria, reduce el batch a 1.

Faster R-CNN guarda `best.pt`, `last.pt` e `historial.csv` en `resultados/faster_entrenamiento`. Su criterio de seleccion es la perdida de validacion; no calcula mAP. YOLO guarda pesos, metricas de deteccion y graficos en `resultados_yolo/entrenamiento*`. Las predicciones finales se realizan sobre validacion con el mejor modelo de cada entrenamiento. Esta validacion sirve para seleccionar modelos; no constituye una evaluacion independiente en test.

Los pesos iniciales son preentrenados y se ajustan a las 28 clases de FjordFish. La configuracion YOLO con rutas locales se genera automaticamente. Ejecutar de nuevo las celdas inicia un entrenamiento nuevo, no reanuda el anterior. Faster R-CNN reemplaza los archivos de su carpeta de salida; cambia el argumento `salida` si quieres conservar corridas anteriores.

Verificacion: `python -m unittest discover -s tests -v`.
