#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include "hist-equ.h"


void run_cpu_color_test(PPM_IMG img_in);
void run_cpu_gray_test(PGM_IMG img_in);
void save_results_to_file(double time_taken, int mode);

int main(){
    PGM_IMG img_ibuf_g;
    PPM_IMG img_ibuf_c;

    MPI_Init(NULL, NULL);

    // Get start time using MPI_Wtime
    double start = MPI_Wtime();

    // NUMBER OF THREADS
    omp_set_num_threads(omp_get_num_procs() - 1); // 1 core for the OS (in the GPUs partition, we will have 12-1 = 11 threads)

    printf("Running contrast enhancement for gray-scale images.\n");
    img_ibuf_g = read_pgm("in.pgm"); // Returns the local image
    run_cpu_gray_test(img_ibuf_g); // All processes process their local gray image
    free_pgm(img_ibuf_g);
    
    // ppm
    printf("Running contrast enhancement for color images.\n");
    img_ibuf_c = read_ppm("in.ppm"); // Returns the local image
    run_cpu_color_test(img_ibuf_c); // All processes process their local color image
    free_ppm(img_ibuf_c);

    double end = MPI_Wtime(); // Get end time
    double time_taken = (end - start) * 1000; // Calculate duration
    printf("Time taken: %f milliseconds\n", time_taken); // Print the time to the console
    save_results_to_file(time_taken, 3);

    MPI_Finalize();

    return 0;
}

void save_results_to_file(double time_taken_local, int mode) {
    int max_threads = omp_get_max_threads();

    double time_taken;
    // Reduce the time taken to process the images to get the maximum time taken
    MPI_Reduce(&time_taken_local, &time_taken, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if (mpi_rank != 0) return; // Only rank 0 writes to the file

    // Get Slurm job information
    const char* partition = getenv("SLURM_JOB_PARTITION");
    const char* nodes = getenv("SLURM_NNODES");
    const char* tasks = getenv("SLURM_NTASKS");

    // Check if running on Slurm in the 'gpus' partition
    if (partition != NULL && strcmp(partition, "gpus") == 0) {
        // Open the file for appending. If it does not exist, it is created
        FILE *outfile;
        if (mode == 0) {
            outfile = fopen("./hybrid-output/time_results_gray.txt", "a"); // mode "a" -> stream is positioned at the end of the file
        } else if (mode == 1) {
            outfile = fopen("./hybrid-output/time_results_hsl.txt", "a"); // mode "a" -> stream is positioned at the end of the file
        } else if (mode == 2) {
            outfile = fopen("./hybrid-output/time_results_yuv.txt", "a"); // mode "a" -> stream is positioned at the end of the file
        } else {
            outfile = fopen("./hybrid-output/time_results_total.txt", "a"); // mode "a" -> stream is positioned at the end of the file
        }
        
        // Check if the file is open successfully
        if (outfile != NULL) {
            if (ftell(outfile) == 0) {  // If file is empty (end of file is at position 0), write headers
                fprintf(outfile, "N (Nodes)\tn (Processes)\tThreads\t\tTime (milliseconds)\n");
            }

            // write results
            fprintf(outfile, "\t%s\t\t\t%s\t\t\t\t%i\t\t%f\n", nodes ? nodes : "N/A", tasks ? tasks : "N/A", max_threads, time_taken);
            
            // close file
            fclose(outfile);  // Close the file
        } else {
            fprintf(stderr, "Error opening file for writing.\n");
        }
    }
}

void run_cpu_color_test(PPM_IMG img_in){
    PPM_IMG img_obuf_hsl, img_obuf_yuv;
    double tstart, tend;
    
    printf("Starting CPU processing...\n");
    
    tstart = MPI_Wtime();
    img_obuf_hsl = contrast_enhancement_c_hsl(img_in);
    tend = MPI_Wtime();
    printf("HSL processing time: %f (ms)\n", (tend - tstart) * 1000 /* TIMER */ );
    save_results_to_file((tend - tstart) * 1000, 1);
    
    write_ppm(img_obuf_hsl, "./hybrid-output/out_hsl.ppm");

    tstart = MPI_Wtime();
    img_obuf_yuv = contrast_enhancement_c_yuv(img_in);
    tend = MPI_Wtime();
    printf("YUV processing time: %f (ms)\n", (tend - tstart) * 1000 /* TIMER */);
    save_results_to_file((tend - tstart) * 1000, 2);
    
    write_ppm(img_obuf_yuv, "./hybrid-output/out_yuv.ppm");
    
    free_ppm(img_obuf_hsl);
    free_ppm(img_obuf_yuv);
}

void run_cpu_gray_test(PGM_IMG img_in){
    PGM_IMG img_obuf;
    double tstart, tend;    
    
    printf("Starting CPU processing...\n");
    
    tstart = MPI_Wtime();
    img_obuf = contrast_enhancement_g(img_in);
    tend = MPI_Wtime();
    printf("Processing time: %f (ms)\n", (tend - tstart) * 1000 /* TIMER */ );
    save_results_to_file((tend - tstart) * 1000, 0);
    
    write_pgm(img_obuf, "./hybrid-output/out.pgm");
    free_pgm(img_obuf);
}

PPM_IMG read_ppm(const char * path){
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    char *ibuf;
    int i, total_w, total_h;
    // Rank 0 reads the image
    if (mpi_rank == 0) {
        FILE * in_file;
        char sbuf[256];
        int v_max;

        in_file = fopen(path, "r");
        if (in_file != NULL){
            /*Skip the magic number*/
            fscanf(in_file, "%s", sbuf);

            //result = malloc(sizeof(PPM_IMG));
            fscanf(in_file, "%d",&total_w);
            fscanf(in_file, "%d",&total_h);
            fscanf(in_file, "%d\n",&v_max);
            printf("Image size: %d x %d\n", total_w, total_h);
            
            ibuf = (char *)malloc(3 * total_w * total_h * sizeof(char));
            fread(ibuf,sizeof(unsigned char), 3 * total_w * total_h, in_file);
            fclose(in_file);
        }
        else {
            printf("Input file not found!\n");
            total_h = 0;
        }
    }
    // Broadcast the image size to all processes
    MPI_Bcast(&total_h, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_w, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (total_h == 0) exit(1);
    
    // Local image
    PPM_IMG result;
    result.w = total_w;
    result.h = (total_h + mpi_rank) / mpi_size; // distribute pixel rows to processes
    
    // Receive buffer
    char * receivebuf;
    receivebuf = (char *)malloc(3 * result.w * result.h * sizeof(char));

    // Calculate the number of pixels to send to each process and the displacements
    int sendcnts[mpi_size], displs[mpi_size];
    displs[0] = 0;
    for (i = 0; i < mpi_size; i++) {
        sendcnts[i] = 3 * ((total_h + i) / mpi_size) * total_w;
        if (i < mpi_size - 1) displs[i + 1] = displs[i] + sendcnts[i];
    }   

    // Scatter the pixels of the image to all processes
    MPI_Scatterv(ibuf, sendcnts, displs, MPI_UNSIGNED_CHAR, receivebuf, 3 * result.w * result.h, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Allocate memory for the image
    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    // Copy the pixels to the local image in a parallel loop
    #pragma omp parallel for schedule(static)
    for (i = 0; i < result.w * result.h; i++) {
        result.img_r[i] = receivebuf[3 * i + 0];
        result.img_g[i] = receivebuf[3 * i + 1];
        result.img_b[i] = receivebuf[3 * i + 2];
    }
    
    free(receivebuf);
    if (mpi_rank == 0) free(ibuf);
    return result;
}

void write_ppm(PPM_IMG img, const char * path){
    int mpi_rank, mpi_size, total_h, i;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    // Reduce the height of the image to get the total height
    MPI_Allreduce(&img.h, &total_h, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    // Obtain the pixels of the local images in a buffer in a parallel loop
    char * sendbuf;
    sendbuf = (char *)malloc(3 * img.w * img.h * sizeof(char));    
    #pragma omp parallel for schedule(static)
    for(i = 0; i < img.w*img.h; i ++){
        sendbuf[3*i + 0] = img.img_r[i];
        sendbuf[3*i + 1] = img.img_g[i];
        sendbuf[3*i + 2] = img.img_b[i];
    }

     // Allocate memory for the result image in rank 0
    char * obuf;
    if (mpi_rank == 0) obuf = (char *)malloc(3 * img.w * total_h * sizeof(char));

    // Calculate the number of pixels each process sends to rank 0 and the displacements
    int recvcnts[mpi_size], displs[mpi_size];
    displs[0] = 0;
    for (i = 0; i < mpi_size; i++) {
        recvcnts[i] = 3 * ((total_h + i) / mpi_size) * img.w;
        if (i < mpi_size - 1) displs[i + 1] = displs[i] + recvcnts[i];
    }
    // Gather the pixels of the local images to rank 0
    MPI_Gatherv(sendbuf, 3 * img.w * img.h, MPI_UNSIGNED_CHAR, obuf, recvcnts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    free(sendbuf);
    // Rank 0 writes the result image to the file
    if (mpi_rank == 0) {
        FILE * out_file;
        out_file = fopen(path, "wb");
        fprintf(out_file, "P6\n");
        fprintf(out_file, "%d %d\n255\n",img.w, total_h);
        fwrite(obuf,sizeof(unsigned char), 3*img.w*total_h, out_file);
        fclose(out_file);
        free(obuf);
    }
}

void free_ppm(PPM_IMG img)
{
    free(img.img_r);
    free(img.img_g);
    free(img.img_b);
}

PGM_IMG read_pgm(const char * path){
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    // Rank 0 reads the image
    char *ibuf;
    int i, total_w, total_h;
    if (mpi_rank == 0) {
        FILE * in_file;
        char sbuf[256];
        int v_max;

        in_file = fopen(path, "r");
        if (in_file != NULL){
            fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
            fscanf(in_file, "%d",&total_w);
            fscanf(in_file, "%d",&total_h);
            fscanf(in_file, "%d\n",&v_max);
            printf("Image size: %d x %d\n", total_w, total_h);

            ibuf = (char *)malloc(total_w * total_h * sizeof(char));
            fread(ibuf,sizeof(unsigned char), total_w * total_h, in_file);
            fclose(in_file);
        }
        else {
            printf("Input file not found!\n");
            total_h = 0;
        }
    }
    // Broadcast the image size to all processes
    MPI_Bcast(&total_h, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_w, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (total_h == 0) exit(1);
    
    // Local image
    PGM_IMG result;
    result.w = total_w;
    result.h = (total_h + mpi_rank) / mpi_size; // distribute pixel rows to processes
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    // Calculate the number of pixels each process sends to each process and the displacements
    int sendcnts[mpi_size], displs[mpi_size];
    displs[0] = 0;
    for (i = 0; i < mpi_size; i++) {
        sendcnts[i] = ((total_h + i) / mpi_size) * total_w;
        if (i < mpi_size - 1) displs[i + 1] = displs[i] + sendcnts[i];
    }
    // Scatter the pixels of the image to all processes
    MPI_Scatterv(ibuf, sendcnts, displs, MPI_UNSIGNED_CHAR, result.img, result.w * result.h, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    
    if (mpi_rank == 0) free(ibuf);
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    int mpi_rank, mpi_size, total_h, i;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Reduce the height of the image to get the total height
    MPI_Allreduce(&img.h, &total_h, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Allocate memory for the pixels of the result image in a buffer
    char * obuf;
    if (mpi_rank == 0) obuf = (char *)malloc(img.w * total_h * sizeof(char));

    // Calculate the number of pixels each process sends to rank 0 and the displacements
    int recvcnts[mpi_size], displs[mpi_size];
    displs[0] = 0;
    for (i = 0; i < mpi_size; i++) {
        recvcnts[i] = ((total_h + i) / mpi_size) * img.w;
        if (i < mpi_size - 1) displs[i + 1] = displs[i] + recvcnts[i];
    }
    // Gather the pixels of the local images to rank 0 within the buffer
    MPI_Gatherv(img.img, img.w * img.h, MPI_UNSIGNED_CHAR, obuf, recvcnts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Rank 0 writes the result image to the file
    if (mpi_rank == 0) {
        FILE * out_file;
        out_file = fopen(path, "wb");
        fprintf(out_file, "P5\n");
        fprintf(out_file, "%d %d\n255\n",img.w, total_h);
        fwrite(obuf,sizeof(unsigned char), img.w*total_h, out_file);
        fclose(out_file);
        free(obuf);
    }
}

void free_pgm(PGM_IMG img){
    free(img.img);
}
