#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include "hist-equ.h"

void run_cpu_color_test(PPM_IMG img_in);
void run_cpu_gray_test(PGM_IMG img_in);
void save_results_to_file(double time_taken);


int main(){
    PGM_IMG img_ibuf_g;
    PPM_IMG img_ibuf_c;

    MPI_Init(NULL, NULL);

    double start = MPI_Wtime(); // Get start time

    printf("Running contrast enhancement for gray-scale images.\n");
    img_ibuf_g = read_pgm("in.pgm");
    run_cpu_gray_test(img_ibuf_g);
    free_pgm(img_ibuf_g);
    
    printf("Running contrast enhancement for color images.\n");
    img_ibuf_c = read_ppm("in.ppm");
    run_cpu_color_test(img_ibuf_c);
    free_ppm(img_ibuf_c);
    
    double end = MPI_Wtime(); // Get end time
    double time_taken = end - start; // Calculate duration
    printf("Time taken: %f seconds\n", time_taken); // Print the time to the console
    save_results_to_file(time_taken);

    MPI_Finalize();

    return 0;
}

void save_results_to_file(double time_taken_local) {
    double time_taken;
    MPI_Reduce(&time_taken_local, &time_taken, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if (mpi_rank != 0) return;

    // Get Slurm job information
    const char* partition = getenv("SLURM_JOB_PARTITION");
    const char* nodes = getenv("SLURM_NNODES");
    const char* tasks = getenv("SLURM_NTASKS");

    // Check if running on Slurm in the 'gpus' partition
    if (partition != NULL && strcmp(partition, "gpus") == 0) {
        // Open the file for appending. If it does not exist, it is created
        FILE *outfile = fopen("./mpi-output/time_results.txt", "a"); // mode "a" -> stream is positioned at the end of the file
        
        // Check if the file is open successfully
        if (outfile != NULL) {
            if (ftell(outfile) == 0) {  // If file is empty (end of file is at position 0), write headers
                fprintf(outfile, "N (Nodes)\tn (Processes)\tTime (seconds)\n");
            }

            // write results
            fprintf(outfile, "\t%s\t\t\t%s\t\t\t%f\n", nodes ? nodes : "N/A", tasks ? tasks : "N/A", time_taken);
            
            // close file
            fclose(outfile);  // Close the file
        } else {
            fprintf(stderr, "Error opening file for writing.\n");
        }
    }
}

void run_cpu_color_test(PPM_IMG img_in)
{
    PPM_IMG img_obuf_hsl, img_obuf_yuv;
    double tstart, tend;
    
    printf("Starting CPU processing...\n");

    tstart = MPI_Wtime();
    img_obuf_hsl = contrast_enhancement_c_hsl(img_in);
    tend = MPI_Wtime();
    printf("HSL processing time: %f (ms)\n", (tend - tstart) * 1000 /* TIMER */ );
    
    write_ppm(img_obuf_hsl, "./mpi-output/out_hsl.ppm");

    tstart = MPI_Wtime();
    img_obuf_yuv = contrast_enhancement_c_yuv(img_in);
    tend = MPI_Wtime();
    printf("YUV processing time: %f (ms)\n", (tend - tstart) * 1000 /* TIMER */);
    
    write_ppm(img_obuf_yuv, "./mpi-output/out_yuv.ppm");
    
    free_ppm(img_obuf_hsl);
    free_ppm(img_obuf_yuv);
}




void run_cpu_gray_test(PGM_IMG img_in)
{
    PGM_IMG img_obuf;
    double tstart, tend;
    
    
    printf("Starting CPU processing...\n");
    
    tstart = MPI_Wtime();
    img_obuf = contrast_enhancement_g(img_in);
    tend = MPI_Wtime();
    printf("Processing time: %f (ms)\n", (tend - tstart) * 1000 /* TIMER */ );
    
    write_pgm(img_obuf, "./mpi-output/out.pgm");
    free_pgm(img_obuf);
}



PPM_IMG read_ppm(const char * path){
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    char *ibuf;
    int i, total_w, total_h;
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
    MPI_Bcast(&total_h, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_w, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (total_h == 0) exit(1);
    
    PPM_IMG result;
    result.w = total_w;
    result.h = (total_h + mpi_rank) / mpi_size; // distribute pixel rows to processes

    if (mpi_rank == 0) {
        ibuf += 3 * result.w * result.h;
        int result_h;
        for (i = 1; i < mpi_size; i++) {
            result_h = (total_h + i) / mpi_size;
            MPI_Send(ibuf, 3 * result.w * result_h, MPI_CHAR, i, 1, MPI_COMM_WORLD);
            ibuf += 3 * result.w * result_h;
        }
        ibuf -= 3 * result.w * total_h;
    }
    else {
        ibuf = (char *)malloc(3 * result.w * result.h * sizeof(char));
        MPI_Recv(ibuf, 3 * result.w * result.h, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    for(i = 0; i < result.w*result.h; i ++){
        result.img_r[i] = ibuf[3*i + 0];
        result.img_g[i] = ibuf[3*i + 1];
        result.img_b[i] = ibuf[3*i + 2];
    }
    
    free(ibuf);
    return result;
}

void write_ppm(PPM_IMG img, const char * path){
    int mpi_rank, mpi_size, total_h, i;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Reduce(&img.h, &total_h, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    char * obuf;
    if (mpi_rank == 0) {
        obuf = (char *)malloc(3 * img.w * total_h * sizeof(char));
    }
    else {
        obuf = (char *)malloc(3 * img.w * img.h * sizeof(char));
    }
    
    for(i = 0; i < img.w*img.h; i ++){
        obuf[3*i + 0] = img.img_r[i];
        obuf[3*i + 1] = img.img_g[i];
        obuf[3*i + 2] = img.img_b[i];
    }

    if (mpi_rank != 0) {
        MPI_Send(obuf, 3 * img.w * img.h, MPI_CHAR, 0, 100, MPI_COMM_WORLD);
    }
    else {
        obuf += 3 * img.w * img.h;

        int img_h;
        for (i = 1; i < mpi_size; i++) {
            img_h = (total_h + i) / mpi_size;
            MPI_Recv(obuf, 3 * img.w * img_h, MPI_CHAR, i, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            obuf += 3 * img.w * img_h;
        }
        obuf -= 3 * img.w * total_h;

        FILE * out_file;
        out_file = fopen(path, "wb");
        fprintf(out_file, "P6\n");
        fprintf(out_file, "%d %d\n255\n",img.w, total_h);
        fwrite(obuf,sizeof(unsigned char), 3*img.w*total_h, out_file);
        fclose(out_file);
    }

    free(obuf);
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
    
    char *ibuf;
    int i, total_w, total_h;
    if (!mpi_rank) {
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

            ibuf = (unsigned char *)malloc(total_w * total_h * sizeof(unsigned char));
            fread(ibuf,sizeof(unsigned char), total_w * total_h, in_file);
            fclose(in_file);
        }
        else {
            printf("Input file not found!\n");
            total_h = 0;
        }
    }
	
	// Broadcast de las dimensiones de la imagen
    MPI_Bcast(&total_h, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_w, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
	// Si no se ha leído la imagen -> salimos code error 1
	if (!total_h) exit(1);
    
	// Calculo de variables locales para cada proceso
	int *pixels_per_process = NULL;
    int *pixels_per_process = NULL;

    if (!mpi_rank) {
        pixels_per_process = (int*)malloc(mpi_size * sizeof(int));
        processes_offsets = (int*)malloc(mpi_size * sizeof(int));
        
        int offset = 0;
        for (int i = 0; i < mpi_size; i++) {
            int local_h = (total_h + i) / mpi_size;		// asignacion variable pero equitativa
            pixels_per_process[i] = local_h * total_w; 	// total de pixeles para ese proceso
            processes_offsets[i] = offset;
            offset += pixels_per_process[i];
        }
    }
	
	// Creación de la variable imagen resultante -> asignación de algunas de las variables
    PGM_IMG result;
    result.w = total_w;
    result.h = (total_h + mpi_rank) / mpi_size; // distribute pixel rows to processes
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

	// Scatter_V
	// Se mandan los arrays con los distintos offsets y pixeles que le corresponden a cada proceso
	MPI_Scatterv(ibuf, pixels_per_process, processes_offsets, MPI_UNSIGNED_CHAR,
                 result.img, result.w * result.h, MPI_UNSIGNED_CHAR,
                 0, MPI_COMM_WORLD);

	// Free memory
    if (!mpi_rank){
		free(ibuf);
		free(pixels_per_process);
		free(processes_offsets);
	}
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    int mpi_rank, mpi_size, total_h, i;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Reduce(&img.h, &total_h, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mpi_rank != 0) {
        MPI_Send(img.img, img.w * img.h, MPI_CHAR, 0, 99, MPI_COMM_WORLD);
    }
    else {
        char * obuf = (char *)malloc(img.w * total_h * sizeof(char));

        for (i = 0; i < img.w * img.h; i++) {
            obuf[i] = img.img[i];
        }
        obuf += img.w * img.h;

        int img_h;
        for (i = 1; i < mpi_size; i++) {
            img_h = (total_h + i) / mpi_size;
            MPI_Recv(obuf, img.w * img_h, MPI_CHAR, i, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            obuf += img.w * img_h;
        }
        obuf -= img.w * total_h;

        FILE * out_file;
        out_file = fopen(path, "wb");
        fprintf(out_file, "P5\n");
        fprintf(out_file, "%d %d\n255\n",img.w, total_h);
        fwrite(obuf,sizeof(unsigned char), img.w*total_h, out_file);
        fclose(out_file);
    }
}

void free_pgm(PGM_IMG img)
{
    free(img.img);
}