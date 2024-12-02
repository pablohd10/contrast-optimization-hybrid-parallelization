# Contrast Optimization using hybrid parallelization

Este proyecto desarrolla una aplicación para mejorar el contraste en imágenes mediante la ecualización del histograma. La implementación está diseñada para ejecutarse en un entorno de cómputo de altas prestaciones, utilizando paralelización con MPI y OpenMP. Se incluye la versión secuencial, paralela no distribuida (OpenMP), paralela distribuida (MPI), y paralela híbrida (OpenMP + MPI).

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

## Compilación del Programa

```bash
$ cd build
$ cmake .. -DCMAKE_CXX_COMPILER=mpicxx.mpich
$ make
```

## Ejecución del Programa

Para ejecutar la aplicación en un entorno distribuido que utilice slurm como planificador, use el siguiente comando:
```bash
srun -p gpus -N <número de nodos> -n <número de tareas> ./contrast
```

## Descargar MPI
descargar versiones:
 - openmpi _(mejor integración con pmix)_
 - mpich (no se puede usar para usos comerciales)
  -> _(misma interfaz pero mejor runtime, le gusta más al profe)_
(o algo asi)
