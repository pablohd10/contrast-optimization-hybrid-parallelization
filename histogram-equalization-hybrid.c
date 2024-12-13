#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include "hist-equ.h"


void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin) {

    // Initialize each element of the histogram to 0. We decide not to parallelize this loop as it consists of only 256 iterations.
    int i;
    for (i = 0; i < nbr_bin; i++){
        hist_out[i] = 0;
    }

    // Count the number of occurrences in the image for each pixel value
    #pragma omp parallel for reduction(+: hist_out[:nbr_bin]) schedule(static)// Reduction addition operation on hist_out. Iterations are evenly distibuted across threads
    for (int i = 0; i < img_size; i++) {
        hist_out[img_in[i]]++; 
    }
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in_local, int img_size_local, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i;

    int hist_in[nbr_bin];
    int img_size, mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    // Reduce the local histograms to get the global histogram
    MPI_Reduce(hist_in_local, hist_in, nbr_bin, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // Reduce the local image sizes to get the global image size
    MPI_Reduce(&img_size_local, &img_size, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Only rank 0 constructs the LUT
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
    // Broadcast the LUT to all processes
    MPI_Bcast(lut, nbr_bin, MPI_INT, 0, MPI_COMM_WORLD);
    
    /* Get the result image */
    #pragma omp parallel for schedule(static) // Iterations are evenly distibuted across threads (independent and identical operations)
        for(int i = 0; i < img_size_local; i ++){
            if(lut[img_in[i]] > 255){
                img_out[i] = 255;
            }
            else{
                img_out[i] = (unsigned char)lut[img_in[i]];
            }
        }
}
