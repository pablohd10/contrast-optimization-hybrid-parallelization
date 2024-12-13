#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "hist-equ.h"


void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin) {

    // Initialize each element of the histogram to 0. We decide not to parallelize this loop as it consists of only 256 iterations.
    int i;
    for (i = 0; i < nbr_bin; i++){
        hist_out[i] = 0;
    }

    // Count the number of occurrences in the image for each pixel value
    #pragma omp parallel for reduction(+: hist_out[:nbr_bin]) schedule(static)// Reduction addition operation on hist_out. Iterations are evenly distibuted across threads (static scheduling)
    for (int i = 0; i < img_size; i++) {
        hist_out[img_in[i]]++; 
    }
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist_in[i++];
    }
    d = img_size - min;

    // We decide not to parallelize this loop as it consists of only 256 iterations.
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
        
        
    }
    
    /* Get the result image */
    #pragma omp parallel for schedule(static) // Iterations are evenly distibuted across threads (static scheduling)
        for(int i = 0; i < img_size; i ++){
            if(lut[img_in[i]] > 255){
                img_out[i] = 255;
            }
            else{
                img_out[i] = (unsigned char)lut[img_in[i]];
            }
        }
}
