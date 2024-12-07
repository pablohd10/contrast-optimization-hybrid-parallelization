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
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Cada proceso MPI inicializa su propio histograma local (hist_local) a cero.
    int *hist_local = (int*) malloc(nbr_bin * sizeof(int));
    if (hist_local == NULL) {
        fprintf(stderr, "Error al asignar memoria para hist_local\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for(i = 0; i < nbr_bin; i++){
        hist_local[i] = 0;
    }

    // Cada proceso recorre su porción de la imagen (img_in) y actualiza hist_local contando las ocurrencias de cada nivel de intensidad.
    for(i = 0; i < img_size; i++){
        hist_local[img_in[i]]++;
    }

    // Se utiliza MPI_Reduce para sumar todos los histogramas locales en un histograma global (hist_out) que solo está disponible en el proceso raíz (rank 0).
    MPI_Reduce(hist_local, hist_out, nbr_bin, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    free(hist_local);
}

// Función para realizar la ecualización del histograma
void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
    int i, cdf, min, d;
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *lut = NULL;

    if(rank == 0){
        // Solo el proceso raíz (rank 0) calcula la LUT utilizando el histograma global (hist_in).
        lut = (int *)malloc(sizeof(int)*nbr_bin);
        if (lut == NULL) {
            fprintf(stderr, "Error al asignar memoria para LUT\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // Se calcula la CDF (Función de Distribución Acumulativa) y se construye la LUT basada en esta.
        cdf = 0;
        min = 0;
        i = 0;
        while(min == 0 && i < nbr_bin){
            min = hist_in[i++];
        }
        d = img_size - min;

        for(i = 0; i < nbr_bin; i++){
            cdf += hist_in[i];
            lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
            if(lut[i] < 0){
                lut[i] = 0;
            }
        }
    }

    // La LUT calculada en el proceso raíz se difunde a todos los demás procesos utilizando MPI_Bcast.
    if(rank != 0){
        lut = (int *)malloc(sizeof(int)*nbr_bin);
        if (lut == NULL) {
            fprintf(stderr, "Error al asignar memoria para LUT en proceso %d\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Bcast(lut, nbr_bin, MPI_INT, 0, MPI_COMM_WORLD);

    // Cada proceso aplica la LUT a su porción local de la imagen (img_in) para generar la imagen ecualizada local (img_out).
    for(i = 0; i < img_size; i++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
    }

    free(lut);
}