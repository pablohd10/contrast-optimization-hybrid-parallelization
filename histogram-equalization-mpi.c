#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include "hist-equ.h"


void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in_local, int img_size_local, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i;

    int hist_in[nbr_bin];
    int img_size, mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Reduce(hist_in_local, hist_in, nbr_bin, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&img_size_local, &img_size, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        int cdf, min, d;
        /* Construct the LUT by calculating the CDF */
        cdf = 0;
        min = 0;
        i = 0;
        while(min == 0){
            min = hist_in[i++];
        }
        d = img_size - min;
        for(i = 0; i < nbr_bin; i ++){
            cdf += hist_in[i];
            //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
            lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
            if(lut[i] < 0){
                lut[i] = 0;
            }
        }
    }
    MPI_Bcast(lut, nbr_bin, MPI_INT, 0, MPI_COMM_WORLD);
    
    /* Get the result image */
    for(i = 0; i < img_size_local; i ++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
        
    }

    free(lut);
}