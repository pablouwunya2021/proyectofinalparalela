# Proyecto 3 — Mejorando algoritmos con CUDA

Aceleración en GPU de un **detector de fraude transaccional con LSTM** (PyTorch + CUDA).

Curso: **Computación Paralela y Distribuida** — Universidad del Valle de Guatemala.

## Integrantes
- Pablo Cabrera — 231156
- Eliazar Canastuj
- Diego Ramírez

## Descripción

Partimos del **Laboratorio 3 — Secuencias: RNN y LSTM** (detector de fraude del
"Banco del Altiplano"), desarrollado previamente en Deep Learning con PyTorch, y nos
acogemos a la *Nota adicional* del enunciado para acelerarlo con la GPU usando CUDA.

El objetivo es **trasladar la carga matemática de la red neuronal al GPU** manteniendo
la **preparación de datos en la CPU**, y medir la mejora real de rendimiento.

## Contenido del repositorio

| Archivo | Descripción |
|---------|-------------|
| `Proyecto3_CUDA_LSTM.ipynb` | Notebook principal: versión secuencial (CPU) + versión optimizada (CUDA), medición de tiempos y speedup. Listo para Google Colab con GPU. |
| `Informe_Proyecto3_CUDA_LSTM.docx` | Informe de rendimiento y análisis (tablas, speedup, evidencia y discusión técnica). |
| `S04-S05_Lab03_RNN_LSTM_ESTUDIANTE.ipynb` | Laboratorio 3 original (solución secuencial base). |
| `RESULTADOS.md` | Resumen de los tiempos y speedup obtenidos. |
| `resultados/` | Datos crudos del benchmark y gráficos generados. |
| `requirements.txt` | Dependencias de Python. |

## Cómo ejecutar

1. Subir `Proyecto3_CUDA_LSTM.ipynb` a [Google Colab](https://colab.research.google.com/).
2. *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4)*.
3. *Entorno de ejecución → Ejecutar todo*.

Localmente (sin GPU solo corre la parte de CPU):

```bash
pip install -r requirements.txt
```

## Resultados (Tesla T4, Google Colab)

| Configuración | CPU | GPU | Speedup |
|---|---|---|---|
| Pequeña (N=2000, hidden=32) | 1.03 s | 0.50 s | **2.05×** |
| Escalada (N=6000, hidden=256) | 28.91 s | 0.64 s | **45×** |

El speedup crece con el tamaño del modelo (de 1.6× con hidden=16 hasta 34.7× con
hidden=256), confirmando que CUDA conviene cuanto mayor es la carga matemática.
