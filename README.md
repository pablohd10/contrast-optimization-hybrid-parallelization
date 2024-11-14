# Contrast Optimization using hybrid parallelization

Este proyecto desarrolla una aplicación para mejorar el contraste en imágenes mediante la ecualización del histograma. La implementación está diseñada para ejecutarse en un entorno de cómputo de altas prestaciones, utilizando paralelización con MPI y OpenMP.

## Descripción del Proyecto

La mejora de contraste es esencial en el procesamiento de imágenes científicas, como en radiografías o imágenes satelitales, y también en fotografías de bajo o alto contraste. Este proyecto busca optimizar el procesamiento de contraste mediante técnicas de paralelización, enfocándose en dividir la carga de procesamiento para obtener una mayor eficiencia y tiempos de ejecución más bajos.

### Objetivo

Desarrollar un programa paralelo en C que:
- Aplique ecualización de histograma a imágenes en escala de grises (formato PGM) y en color (formato PPM).
- Mejore el rendimiento mediante el uso de múltiples núcleos y nodos en un clúster de computación.

## Requisitos

- **Lenguaje de programación**: C
- **Librerías**: MPI, OpenMP
- **Formato de entrada**: `in.pgm` (escala de grises) y `in.ppm` (color)
- **Formato de salida**: `out_hsl.ppm` y `out_yuv.ppm`

## Ejecución del Programa

Para ejecutar la aplicación, usa el siguiente comando:
```bash
srun -p gpus -N 1 -n 1 ./contrast
```

## Descargar MPI
descargar versiones:
 - openmpi
 - mpich 
(o algo asi)
