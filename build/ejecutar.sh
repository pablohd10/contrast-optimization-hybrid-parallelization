#!/bin/bash

# Nombre del programa a ejecutar
PROGRAM="./contrast"

# Partición a utilizar
PARTITION="gpus"

# Máximo de nodos y procesos
MAX_NODES=4
MAX_PROCESSES=16

# Iterar sobre la cantidad de nodos
for ((nodes=1; nodes<=MAX_NODES; nodes++)); do
    # Iterar sobre la cantidad de procesos
    for ((processes=1; processes<=MAX_PROCESSES; processes++)); do
        echo "Ejecutando con $nodes nodos y $processes procesos..."
      
        # Ejecutar el programa con srun y guardar la salida
        srun -N "$nodes" -n "$processes" -p "$PARTITION" "$PROGRAM"
    done
done

echo "Todas las combinaciones se han ejecutado."
