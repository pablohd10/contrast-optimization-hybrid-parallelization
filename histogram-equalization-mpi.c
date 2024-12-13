#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include "hist-equ.h"


// Paralelizar el cálculo y la ecualización del histograma utilizando MPI implica distribuir 
// la carga de trabajo entre múltiples procesos, calcular histogramas locales, 
// combinar estos histogramas para formar un histograma global, y luego aplicar la LUT resultante 
// a cada porción de la imagen.

// Función para calcular el histograma parcial y reducirlo a uno global
void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }
}

// Histogram equalization
// Note that the parameters will contain the the values por each process
void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in_local, int img_size_local, int nbr_bin){
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Arrays para histograma global y LUT
    int *hist_global = NULL;
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);

    // Reducimos los histogramas locales para obtener el histograma global en el MASTER
    if (!mpi_rank) {
        hist_global = (int *)malloc(nbr_bin * sizeof(int));
    }
    MPI_Reduce(hist_in_local, hist_global, nbr_bin, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Reducimos el tamaño local de la imagen para obtener el tamaño global en el MASTER
    int img_size_global;
    MPI_Reduce(&img_size_local, &img_size_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // MASTER calcula la LUT a partir del histograma global
    if (!mpi_rank) {
        int cdf = 0, min = 0, d;
        int i = 0;
        
        // Encontrar el primer valor no cero para calcular la LUT
        while(i < nbr_bin && min == 0){
            min = hist_global[i++];
        }

        // Si min sigue siendo 0, significa que el histograma está vacío o la imagen es uniforme
        if (min == 0) {
            // Puedes decidir cómo manejar este caso especial
            fprintf(stderr, "Warning: No non-zero values in global histogram.\n");
        }

        d = img_size_global - min;
        cdf = 0;
        for(i = 0; i < nbr_bin; i++){
            cdf += hist_global[i];
            int val = (int)(((float)(cdf - min) * 255) / d + 0.5);
            lut[i] = (val < 0) ? 0 : val;
        }

        free(hist_global);
    }

    // Difundir la LUT a todos los procesos
    MPI_Bcast(lut, nbr_bin, MPI_INT, 0, MPI_COMM_WORLD);

    // Cada proceso aplica la LUT a su porción local de la imagen
    for(int i = 0; i < img_size_local; i++){
        int val = lut[img_in[i]];
        img_out[i] = (unsigned char)((val > 255) ? 255 : val);
    }

    free(lut);
}
