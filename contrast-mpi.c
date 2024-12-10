#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include "hist-equ.h"

#define MASTER 0

void run_cpu_color_test(PPM_IMG img_in, int local_height_c, int total_height_c, int total_width_c, int *chunk_heights_c, int *displacements_c);
void run_cpu_gray_test(PGM_IMG img_in, int local_height_c, int total_height_g, int total_width_c, int *chunk_heights_c, int *displacements_c);

// Auxiliary functions
void save_results_to_file(double time_taken);
void calculate_chunk_height(int num_procesos, int total_height, int *chunk_heights, int *row_displacements);

int main(){

    MPI_Init(NULL, NULL);

    PGM_IMG img_ibuf_g, local_img_g;
    PPM_IMG img_ibuf_c, local_img_c;

    int total_width_g, total_width_c, total_height_g, total_height_c;

    double start;

    // Obtain information about the process it is being executed and the total number of processes
    int num_procesos, rank, len_node_name;
    char node_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Comm_size(MPI_COMM_WORLD, &num_procesos);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(node_name, &len_node_name);

    // Varables for chunking
    int *chunk_heights_g, *displacements_g, *chunk_heights_c, *displacements_c;

    // Only the MASTER process reads the images. MAYBE WE CAN DO THAT 1 PROCESS READS PGM AND ANOTHER THE PPM IN PARALLEL
    if (rank == MASTER){

        // Get start time using MPI_Wtime
        start = MPI_Wtime();

        printf("MASTER process (%d) reading pgm image.\n", rank);
        img_ibuf_g = read_pgm("in.pgm");
        total_width_g = img_ibuf_g.w;
        total_height_g = img_ibuf_g.h;
        
        printf("MASTER process (%d) reading ppm image.\n", rank);
        img_ibuf_c = read_ppm("in.ppm");
        total_width_c = img_ibuf_c.w;
        total_height_c = img_ibuf_c.h;

        // Calculate the height of each chunk and what the position at which it starts
        chunk_heights_g = (int *) malloc(num_procesos * sizeof(int));
        displacements_g = (int *) malloc(num_procesos * sizeof(int));
        chunk_heights_c = (int *) malloc(num_procesos * sizeof(int));
        displacements_c = (int *) malloc(num_procesos * sizeof(int));
        calculate_chunk_height(num_procesos, total_height_g, chunk_heights_g, displacements_g);
        calculate_chunk_height(num_procesos, total_height_c, chunk_heights_c, displacements_c);
    }

    // Broadcast image width and height
    MPI_Bcast(&total_width_g, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&total_width_c, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&total_height_g, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&total_height_c, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    int local_height_g, local_height_c;
    // Scatter chunk heights
    MPI_Scatter(chunk_heights_g, 1, MPI_INT, &local_height_g, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Scatter(chunk_heights_c, 1, MPI_INT, &local_height_c, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Allocate space for local PGM_IMG
    local_img_g.w = total_width_g;
    local_img_g.h = local_height_g;
    local_img_g.img = (unsigned char *)malloc(local_height_g * total_width_g * sizeof(unsigned char));
    // Allocate space for local PPM_IMG
    local_img_c.w = total_width_c;
    local_img_c.h = local_height_c;
    local_img_c.img_r = (unsigned char *)malloc(local_height_c * total_width_c * sizeof(unsigned char));
    local_img_c.img_g = (unsigned char *)malloc(local_height_c * total_width_c * sizeof(unsigned char));
    local_img_c.img_b = (unsigned char *)malloc(local_height_c * total_width_c * sizeof(unsigned char));

    // Scatter gray image rows
    MPI_Scatterv(img_ibuf_g.img, chunk_heights_g, displacements_g, MPI_UNSIGNED_CHAR,
                 local_img_g.img, local_height_g * total_width_g, MPI_UNSIGNED_CHAR,
                 MASTER, MPI_COMM_WORLD);
    // Scatter rgb image rows
    MPI_Scatterv(img_ibuf_c.img_r, chunk_heights_c, displacements_c, MPI_UNSIGNED_CHAR,
                 local_img_c.img_r, local_height_c * total_width_c, MPI_UNSIGNED_CHAR,
                 MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(img_ibuf_c.img_g, chunk_heights_c, displacements_c, MPI_UNSIGNED_CHAR,
                 local_img_c.img_g, local_height_c * total_width_c, MPI_UNSIGNED_CHAR,
                 MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(img_ibuf_c.img_b, chunk_heights_c, displacements_c, MPI_UNSIGNED_CHAR,
                 local_img_c.img_b, local_height_c * total_width_c, MPI_UNSIGNED_CHAR,
                 MASTER, MPI_COMM_WORLD);
                 
    // Barrier to ensure that all processes have received their corresponding chunk
    MPI_Barrier(MPI_COMM_WORLD);

    printf("Running contrast enhancement for gray-scale images.\n");
    run_cpu_gray_test(local_img_g, local_height_g, total_height_g, total_width_g, chunk_heights_g, displacements_g);
    free_pgm(local_img_g);
    
    printf("Running contrast enhancement for color images.\n");
    run_cpu_color_test(local_img_c, local_height_c, total_height_c, total_width_c, chunk_heights_c, displacements_c);
    free_ppm(local_img_c);
    
    if (rank == MASTER){
        // Get end time
        double end = MPI_Wtime();
        // Calculate duration
        double time_taken = end - start;
        printf("Time taken: %f seconds\n", time_taken);
        save_results_to_file(time_taken);

        free_pgm(img_ibuf_g);
        free_ppm(img_ibuf_c);
    }

    MPI_Finalize();

    return 0;
}

void calculate_chunk_height(int num_procesos, int total_height, int *chunk_heights, int *row_displacements) {
    int base_height = total_height / num_procesos;
    int remainder = total_height % num_procesos;

    int offset = 0;

    for (int i = 0; i < num_procesos; i++) {
        // Distribute the extra rows (remainder) among the first processes
        chunk_heights[i] = base_height + (i < remainder ? 1 : 0);

        // Calculate the starting row for each process
        row_displacements[i] = offset;

        // Update offset to point to the next process's starting row
        offset += chunk_heights[i];
    }
}

void save_results_to_file(double time_taken) {
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

void run_cpu_color_test(PPM_IMG img_in, int local_height_c, int total_height_c, int total_width_c, int *chunk_heights_c, int *displacements_c) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    PPM_IMG img_obuf_hsl, img_obuf_yuv, img_obuf_final_hsl, img_obuf_final_yuv;
    double tstart, tend;
    
    printf("Starting CPU processing...\n");

     // Allocate memory for the full image (MASTER process)
    if (rank == MASTER) {
        img_obuf_final_hsl.img_r = (unsigned char *)malloc(total_height_c * total_width_c * sizeof(unsigned char));
        img_obuf_final_hsl.img_g = (unsigned char *)malloc(total_height_c * total_width_c * sizeof(unsigned char));
        img_obuf_final_hsl.img_b = (unsigned char *)malloc(total_height_c * total_width_c * sizeof(unsigned char));

        img_obuf_final_yuv.img_r = (unsigned char *)malloc(total_height_c * total_width_c * sizeof(unsigned char));
        img_obuf_final_yuv.img_g = (unsigned char *)malloc(total_height_c * total_width_c * sizeof(unsigned char));
        img_obuf_final_yuv.img_b = (unsigned char *)malloc(total_height_c * total_width_c * sizeof(unsigned char));
    }

    // Start processing HSL
    tstart = MPI_Wtime();
    img_obuf_hsl = contrast_enhancement_c_hsl(img_in);
    tend = MPI_Wtime();
    printf("HSL processing time: %f (ms)\n", (tend - tstart) * 1000 /* TIMER */);
    
    // Barrier to ensure that all processes have computed their local operations
    MPI_Barrier(MPI_COMM_WORLD);
    // Gather processed HSL image back to MASTER
    MPI_Gatherv(img_obuf_hsl.img_r, local_height_c * total_width_c, MPI_UNSIGNED_CHAR,
                img_obuf_final_hsl.img_r, chunk_heights_c, displacements_c, MPI_UNSIGNED_CHAR,
                MASTER, MPI_COMM_WORLD);
    MPI_Gatherv(img_obuf_hsl.img_g, local_height_c * total_width_c, MPI_UNSIGNED_CHAR,
                img_obuf_final_hsl.img_g, chunk_heights_c, displacements_c, MPI_UNSIGNED_CHAR,
                MASTER, MPI_COMM_WORLD);
    MPI_Gatherv(img_obuf_hsl.img_b, local_height_c * total_width_c, MPI_UNSIGNED_CHAR,
                img_obuf_final_hsl.img_b, chunk_heights_c, displacements_c, MPI_UNSIGNED_CHAR,
                MASTER, MPI_COMM_WORLD);


    // After gathering, MASTER saves the image
    if (rank == MASTER) {
        write_ppm(img_obuf_hsl, "./mpi-output/out_hsl.ppm");
    }

    // Process YUV
    tstart = MPI_Wtime();
    img_obuf_yuv = contrast_enhancement_c_yuv(img_in);
    tend = MPI_Wtime();
    printf("YUV processing time: %f (ms)\n", (tend - tstart) * 1000 /* TIMER */);
    
    // Barrier to ensure that all processes have computed their local operations
    MPI_Barrier(MPI_COMM_WORLD);
    // Gather processed YUV image back to MASTER
    MPI_Gatherv(img_obuf_yuv.img_r, local_height_c * total_width_c, MPI_UNSIGNED_CHAR,
                img_obuf_final_yuv.img_r, chunk_heights_c, displacements_c, MPI_UNSIGNED_CHAR,
                MASTER, MPI_COMM_WORLD);
    MPI_Gatherv(img_obuf_yuv.img_g, local_height_c * total_width_c, MPI_UNSIGNED_CHAR,
                img_obuf_final_yuv.img_r, chunk_heights_c, displacements_c, MPI_UNSIGNED_CHAR,
                MASTER, MPI_COMM_WORLD);
    MPI_Gatherv(img_obuf_yuv.img_b, local_height_c * total_width_c, MPI_UNSIGNED_CHAR,
                img_obuf_final_yuv.img_r, chunk_heights_c, displacements_c, MPI_UNSIGNED_CHAR,
                MASTER, MPI_COMM_WORLD);

    // After gathering, MASTER saves the image
    if (rank == MASTER) {
        write_ppm(img_obuf_final_yuv, "./mpi-output/out_yuv.ppm");
        free_ppm(img_obuf_final_hsl);
        free_ppm(img_obuf_final_yuv);
    }
    free_ppm(img_obuf_hsl);
    free_ppm(img_obuf_yuv);
}

void run_cpu_gray_test(PGM_IMG img_in, int local_height_g, int total_height_g, int total_width_g, int *chunk_heights_g, int *displacements_g) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    PGM_IMG img_obuf, img_obuf_final;
    double tstart, tend;
    
    printf("Starting CPU processing...\n");
     // Allocate memory for the full image (MASTER process)
    if (rank == MASTER) {
        img_obuf_final.img = (unsigned char *)malloc(total_height_g * total_width_g * sizeof(unsigned char));
    }

    // Start processing grayscale image
    tstart = MPI_Wtime();
    img_obuf = contrast_enhancement_g(img_in);
    tend = MPI_Wtime();
    printf("Processing time: %f (ms)\n", (tend - tstart) * 1000 /* TIMER */);

    // Barrier to ensure that all processes have computed their local operations
    MPI_Barrier(MPI_COMM_WORLD);
    // Gather processed grayscale image back to MASTER
    MPI_Gatherv(img_obuf.img, local_height_g * total_width_g, MPI_UNSIGNED_CHAR,
                img_obuf_final.img, chunk_heights_g, displacements_g, MPI_UNSIGNED_CHAR,
                MASTER, MPI_COMM_WORLD);

    // After gathering, MASTER saves the image
    if (rank == MASTER) {
        write_pgm(img_obuf, "./mpi-output/out.pgm");
        free_pgm(img_obuf_final);
    }

    free_pgm(img_obuf);
}


PPM_IMG read_ppm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    char *ibuf;
    PPM_IMG result;
    int v_max, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    /*Skip the magic number*/
    fscanf(in_file, "%s", sbuf);


    //result = malloc(sizeof(PPM_IMG));
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    

    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    ibuf         = (char *)malloc(3 * result.w * result.h * sizeof(char));

    
    fread(ibuf,sizeof(unsigned char), 3 * result.w*result.h, in_file);

    for(i = 0; i < result.w*result.h; i ++){
        result.img_r[i] = ibuf[3*i + 0];
        result.img_g[i] = ibuf[3*i + 1];
        result.img_b[i] = ibuf[3*i + 2];
    }
    
    fclose(in_file);
    free(ibuf);
    
    return result;
}

void write_ppm(PPM_IMG img, const char * path){
    FILE * out_file;
    int i;
    
    char * obuf = (char *)malloc(3 * img.w * img.h * sizeof(char));

    for(i = 0; i < img.w*img.h; i ++){
        obuf[3*i + 0] = img.img_r[i];
        obuf[3*i + 1] = img.img_g[i];
        obuf[3*i + 2] = img.img_b[i];
    }
    out_file = fopen(path, "wb");
    fprintf(out_file, "P6\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(obuf,sizeof(unsigned char), 3*img.w*img.h, out_file);
    fclose(out_file);
    free(obuf);
}

void free_ppm(PPM_IMG img)
{
    free(img.img_r);
    free(img.img_g);
    free(img.img_b);
}

PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    

    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

        
    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    free(img.img);
}
