# Resultados del benchmark

Mediciones realizadas en **Google Colab** con GPU **NVIDIA Tesla T4** (CUDA 13.0,
PyTorch 2.14.1, cuDNN 9.28, CPU de 2 núcleos).

El speedup se calcula como `t_CPU / t_GPU`.

## Tabla comparativa de tiempos

| Configuración | CPU total (s) | GPU total (s) | GPU cómputo (s) | GPU transfer. (s) | Speedup |
|---|--:|--:|--:|--:|--:|
| Pequeña (N=2000, T=30, hidden=32) | 1.029 | 0.501 | 0.488 | 0.013 | **2.05×** |
| Escalada (N=6000, T=50, hidden=256) | 28.909 | 0.637 | 0.630 | 0.006 | **45.4×** |

Memoria GPU pico: 29.1 MB (pequeña) / 204.9 MB (escalada).

## Barrido: speedup según el tamaño del modelo

| hidden_size | CPU (s) | GPU (s) | Speedup |
|--:|--:|--:|--:|
| 16  | 0.105 | 0.065 | 1.63× |
| 32  | 0.165 | 0.057 | 2.90× |
| 64  | 0.394 | 0.060 | 6.53× |
| 128 | 1.240 | 0.094 | 13.23× |
| 256 | 3.961 | 0.114 | 34.72× |

## Conclusión

- El pipeline queda separado: **datos en CPU**, **entrenamiento de la LSTM en GPU (CUDA)**.
- A tamaño pequeño la ventaja del GPU es modesta (2.05×) porque el overhead de
  transferencia y lanzamiento de kernels pesa en proporción.
- Al escalar el problema el speedup crece de forma clara (hasta 45×), porque la
  matemática de la red domina sobre la transferencia.
- La recurrencia temporal de la LSTM es el principal límite al paralelismo, pero la
  paralelización dentro de cada paso (sobre `batch × hidden`) basta para que el GPU
  gane holgadamente en modelos grandes.

Ver gráficos en `resultados/`.
