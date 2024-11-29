#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "hist-equ.h"


void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    omp_set_num_threads(omp_get_num_procs() - 1);

    int i;
    for (i = 0; i < nbr_bin; i++){
        hist_out[i] = 0;
    }

    // Count the number of occurrences in the image for each pixel value --> go through the image and increment the corresponding histogram index.
    #pragma omp parallel shared(hist_out, img_size, img_in, nbr_bin)
    {
        int local_hist_out[nbr_bin];
        for ( i = 0; i < nbr_bin; i ++){
            local_hist_out[i] = 0;
        }

        int num_threads = omp_get_num_threads();
        int chunk = (img_size + num_threads - 1) / num_threads;
        #pragma omp for schedule (static, chunk)
            for (i = 0; i < img_size; i++) {
                hist_out[img_in[i]]++;
            }

        #pragma omp critical
        {
            for (i = 0; i < nbr_bin; i++) {
                hist_out[i] += local_hist_out[i];
            }
        }
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
    // POSSIBLE SECTION TO PARALLELIZE using OpenMP (although there are only 256 iterations, maybe it's not worth it. think about it)
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
        
        
    }
    // POSSIBLE SECTION TO PARALLELIZE using OpenMP. This loop goes through each pixel of the image and assigns the new pixel value to the corresponding position of img_out. (O(w*h) time complexity)
    
    /* Get the result image */
    for(i = 0; i < img_size; i ++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
        
    }
}



