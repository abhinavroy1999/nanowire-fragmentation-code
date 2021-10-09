//-------------------------------------------------------------------------------------------------------------------------
/*
Nanowire simulation parallel code implemented using CUDA API
Author: Abhinav Roy
Date: 22/01/2021
*/
//-------------------------------------------------------------------------------------------------------------------------

// Including all the required header files
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>
//-------------------------------------------------------------------------------------------------------------------------
// Including all the supplementary code
// #include "ran2.c"		//Pseudo random number generator code
#include "real2complex.cu"
#include "complex2real.cu"
#include "complex2real_y.cu"
#include "solve_cahn_hilliard.cu"
#include "compute_variable_mobility.cu"
#include "compute_g.cu"
#include "compute_q.cu"
#include "compute_y.cu"
#include "compute_r.cu"
//-------------------------------------------------------------------------------------------------------------------------
// Defining the value of PI upto 15 decimal places

#define PI 3.141592653589793

//-------------------------------------------------------------------------------------------------------------------------

 #define RESTART

//-------------------------------------------------------------------------------------------------------------------------
#include "cuda_profiler_api.h"
#include <unistd.h>
//#include <complex.h>
/*
// Header files for the cuRAND library
#include <curand.h>
#include<curand_kernel.h>
#include <helper_functions.h>
#include <helper_cuda.h>
*/

#include "gsl/gsl_math.h"
#include "gsl/gsl_rng.h"


//-------------------------------------------------------------------------------------------------------------------------
int main(void)
{

	//-------------------------------------------------------------------------------------------------------------------------
	// Declaration of the variables
	FILE *fp, *fr;
	char file_name[100];
	int i1, i2, i3;
	double dt, dx, dy, dz, M, kappa, A;
	double delkx, delky, delkz, kx, ky, kz;
	int Nx, Ny, Nz, halfNx, halfNy, halfNz;
	int temp, time_step, start_time, end_time;
	double time_elapsed = 0.0;
	time_t begin_t, end_t;
	int block_size_x, block_size_y;
	cudaError_t err;
	// Saving the start time of the simulation run
	begin_t = time(NULL);
	//-------------------------------------------------------------------------------------------------------------------------

	/*
	// cuRAND pseudo random number generator

	const unsigned int DEFAULT_SEED = 777;
	int rand_n = 1;
	float *d_Rand;
	checkCudaErrors(cudaMalloc((void **)&d_Rand, rand_n * sizeof(float)));
	// parsing the seed
    	int seed = DEFAULT_SEED;

    	cudaStream_t stream;
	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	curandGenerator_t prngGPU;
   	checkCudaErrors(curandCreateGenerator(&prngGPU, CURAND_RNG_PSEUDO_MTGP32));
   	checkCudaErrors(curandSetStream(prngGPU, stream));
   	checkCudaErrors(curandSetPseudoRandomGeneratorSeed(prngGPU, seed));
	*/

	//-------------------------------------------------------------------------------------------------------------------------

	// Variables initialization for the simulation
	if ((fr = fopen("./time_data.dat","r")) == NULL)
	{
		printf("Unable to open the time data input file. Exiting!\n");
	}
	else
	{
		fr = fopen("./time_data.dat","r");
	}
	(void) fscanf(fr,"%d%d%d", &start_time, &end_time, &time_step);
	(void) fclose(fr);

	// simulation system parametrs
	Nx = 768;
	Ny = 768;
	Nz = 96;
	dx = dy = dz = 0.5;

	// time step value
	dt = 1.0;

	// half of the simulation system size
	halfNx = (int) Nx/2;
	halfNy = (int) Ny/2;
	halfNz = (int) Nz/2;


	// for defining the Fourier modes
	delkx = 2*PI/(Nx*dx);
	delky = 2*PI/(Ny*dy);
	delkz = 2*PI/(Nz*dz);

	// Simulation parameters
	M = 1.0;
	A = 1.0;
	kappa = 1.0;

	// Stability factor
	double alpha = 0.5;

	// Defining the number of threads in the x and y dimensions of a block (maximum 32*32 = 1024 threads per block allowed)
	block_size_x = 32;
	block_size_y = 32;

	//-------------------------------------------------------------------------------------------------------------------------
	dim3 dimBlock(block_size_x, block_size_y , 1);
	dim3 dimGrid (256, 216);
	//-------------------------------------------------------------------------------------------------------------------------
	// I guess this is to make the numbers x and y the next whole numbers (From Arijit)
	if ( Nx%block_size_x != 0)
	{
		dimGrid.x += 1;
	}
	if (Ny%block_size_y!=0)
	{
		dimGrid.y += 1;
	}
	//-------------------------------------------------------------------------------------------------------------------------
	//checking whether there are sufficient no of blocks
	printf("\nThe no. of blocks created in the x-direction=%d\n",dimGrid.x);
	printf("The no. of blocks created in the y-direction=%d\n",dimGrid.y);
	printf("The no. of blocks created in the z-direction=%d\n",dimGrid.z);
	//-------------------------------------------------------------------------------------------------------------------------
	// Declaring all the host variables
	double *c_host, *k2_vect_host, *kx_vect_host, *ky_vect_host, *kz_vect_host;
	// Declaring arrays for the device
	cufftComplex *c_device_f, *g_device_f, *r_device_f, *y1_device_f, *y2_device_f, *y3_device_f, *qx_device_f, *qy_device_f, *qz_device_f;
	double *c_device, *g_device, *phi_device, *y1_device, *y2_device, *y3_device, *k2_vect_device, *kx_vect_device, *ky_vect_device, *kz_vect_device;
	//-------------------------------------------------------------------------------------------------------------------------

	// Allocating memory to the arrays in the host
	c_host = (double *) malloc(Nx*Ny*Nz*sizeof(double));
	k2_vect_host = (double *) malloc(Nx*Ny*Nz*sizeof(double));
	kx_vect_host = (double *) malloc(Nx*Ny*Nz*sizeof(double));
	ky_vect_host = (double *) malloc(Nx*Ny*Nz*sizeof(double));
	kz_vect_host = (double *) malloc(Nx*Ny*Nz*sizeof(double));

	// Allocating memory to the arrays in the device
	cudaMalloc((void**)&c_device, sizeof(double)*Nx*Ny*Nz);
	cudaMalloc((void**)&g_device, sizeof(double)*Nx*Ny*Nz);
	cudaMalloc((void**)&k2_vect_device, sizeof(double)*Nx*Ny*Nz);
	cudaMalloc((void**)&kx_vect_device, sizeof(double)*Nx*Ny*Nz);
	cudaMalloc((void**)&ky_vect_device, sizeof(double)*Nx*Ny*Nz);
	cudaMalloc((void**)&kz_vect_device, sizeof(double)*Nx*Ny*Nz);
	cudaMalloc((void**)&y1_device, sizeof(double)*Nx*Ny*Nz);
	cudaMalloc((void**)&y2_device, sizeof(double)*Nx*Ny*Nz);
	cudaMalloc((void**)&y3_device, sizeof(double)*Nx*Ny*Nz);
	cudaMalloc((void**)&phi_device, sizeof(double)*Nx*Ny*Nz);
	// cufftComplex type variables
	cudaMalloc((void**)&c_device_f, sizeof(cufftComplex)*Nx*Ny*Nz);
	cudaMalloc((void**)&g_device_f, sizeof(cufftComplex)*Nx*Ny*Nz);
	cudaMalloc((void**)&r_device_f, sizeof(cufftComplex)*Nx*Ny*Nz);
	cudaMalloc((void**)&y1_device_f, sizeof(cufftComplex)*Nx*Ny*Nz);
	cudaMalloc((void**)&y2_device_f, sizeof(cufftComplex)*Nx*Ny*Nz);
	cudaMalloc((void**)&y3_device_f, sizeof(cufftComplex)*Nx*Ny*Nz);
	cudaMalloc((void**)&qx_device_f, sizeof(cufftComplex)*Nx*Ny*Nz);
	cudaMalloc((void**)&qy_device_f, sizeof(cufftComplex)*Nx*Ny*Nz);
	cudaMalloc((void**)&qz_device_f, sizeof(cufftComplex)*Nx*Ny*Nz);

	//-------------------------------------------------------------------------------------------------------------------------
	// Defining the Fourier modes

	for (i1 = 0 ; i1 < Nx; ++i1)
	{
		if(i1 <= halfNx)
		{
			kx = i1*delkx;
		}
		else
		{
			kx = (i1-Nx)*delkx;
		}
		for(i2 = 0; i2 < Ny; ++i2)
		{
			if(i2 <= halfNy)
			{
				ky = i2*delky;
			}
			else
			{
				ky = (i2-Ny)*delky;
			}
			for (i3 = 0; i3 < Nz; ++i3)
			{
				if(i3 <= halfNz)
				{
					kz = i3*delkz;
				}
				else
				{
					kz = (i3-Nz)*delkz;
				}
				k2_vect_host[i3 + Nz*(i2 + Ny*i1)] = kx*kx + ky*ky + kz*kz;
				kx_vect_host[i3 + Nz*(i2 + Ny*i1)] = kx;
				ky_vect_host[i3 + Nz*(i2 + Ny*i1)] = ky;
				kz_vect_host[i3 + Nz*(i2 + Ny*i1)] = kz;
			}

		}
	}

	// Transfer the k2, kx, ky vector to device
	cudaMemcpy(k2_vect_device, k2_vect_host, Nx*Ny*Nz*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(kx_vect_device, kx_vect_host, Nx*Ny*Nz*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(ky_vect_device, ky_vect_host, Nx*Ny*Nz*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(kz_vect_device, kz_vect_host, Nx*Ny*Nz*sizeof(double), cudaMemcpyHostToDevice);

	err = cudaPeekAtLastError();
	if (err != cudaSuccess)
	{
		printf("Error 1: %s\n", cudaGetErrorString(err));
	}
	//-------------------------------------------------------------------------------------------------------------------------
							/*RESTART OF SIMULATION*/
	//-------------------------------------------------------------------------------------------------------------------------
	#ifdef RESTART
	{
		printf("\nCode execution for the restarted simulation has commenced:\n");
		sprintf(file_name, "./output/time%d.dat", start_time);
		if ((fp = fopen(file_name,"rb")) == NULL)
		{
			printf("Unable to open data file. Exiting.\n");
			exit(0);
		}
		else
		{
			fp = fopen(file_name,"rb");
		}
		(void) fread(&c_host[0], sizeof(double),(size_t) Nx*Ny*Nz, fp);
		(void) fclose (fp);
		fflush(fp);

	}

	//-------------------------------------------------------------------------------------------------------------------------
							/*NEW SIMULATION*/
	//-------------------------------------------------------------------------------------------------------------------------
	#else
	{
		printf("\nCode execution for a new simulation has started:\n");

		//Remove all the existing output]
		(void) system("rm -rf ./output/*");
		//-------------------------------------------------------------------------------------------------------------------------
		unsigned long int seed_val = 5454;
		// long SEED = 654321;
		//-------------------------------------------------------------------------------------------------------------------------
		double R1, R2;
		double c_zero, c_noise;

		// Radius of the nanowires
		R1 = 12.0;
		R2 = 14.0;

		// The composition/density variable

		c_zero = 1.0;
		c_noise = 1.0e-03;

		//Setting the initial composition profile.
		for (i1 = 0; i1 < Nx; ++i1)
		{
			for (i2 = 0; i2 < Ny; ++i2)
			{
				for (i3 = 0; i3 < Nz; ++i3)
				{
					c_host[i3 + Nz*(i2 + Ny*i1)] = 0.0;
				}

			}
		}
		//-------------------------------------------------------------------------------------------------------------------------

		// ORIENTATION OF 30 DEGREES BETWEEN THE RODS
		/*
		int *center1, *center2;
		center1 = (int *) malloc((size_t) Nz*sizeof(int));
		center2 = (int *) malloc((size_t) Nz*sizeof(int));
		int C1, C2;
		C1 = halfNx;
	        C2 = halfNx + R1 + R2;

		for(i1=0; i1 < Nx; i1++)
	        {
	                for(i2=0; i2 < Ny; i2++)
	                {
	                        for (i3=0; i3<Nz; i3++)
	                        {
	                                if ((i1 - C1)*(i1 - C1) + (i2 - halfNy)*(i2 - halfNy) < R1*R1)
	                                {
	                                        c_host[i3 + Nz*(i2 + Ny*i1)] = c_zero;

	                                }
	                        }
	                }
	        }

	        for (i2=0; i2 < Ny; ++i2)
	        {
	                for (i3=0; i3 < Nz; ++i3)
	                {
	                        if (fabs((i3 - halfNz) - (sqrt(3))*(i2 - halfNy)) <= 2/sqrt(3))
	                        {
	                                center1[i3] = i3;
	                                center2[i3] = i2;
	                        }
	                }
	        }

	        for (i1 = 0; i1 < Nx; ++i1)
	        {
	                for(i2=0; i2 < Ny; i2++)
	                {
	                        for (i3=0; i3<Nz; i3++)
	                        {
						if(i3 > 30 && i3 < Nz - 30 && i2 > 30 && i2 < Ny - 30)
						{

	                                                        //if ((i3 - center1[i3])*(i3 - center1[i3]) + (i2 - center2[i3])*(i2 - center2[i3]) + (i1 - C2)*(i1 - C2)< R2*R2)
								if (fabs((i2 - center2[i3])*(i2 - center2[i3])*3/4 + (i1 - C2)*(i1 - C2))< R2*R2)
	                                                      {
	                                                                c_host[i3 + Nz*(i2 + Ny*i1)] = c_zero;

	                                                      }

						}

	                        }
			}
	        }
		// Free all the dynamically allocated variables for storing the central axis of different wire orientations
		free(center1);
		free(center2);
		*/
		//-------------------------------------------------------------------------------------------------------------------------

		// ORIENTATION OF 60 DEGREES BETWEEN THE RODS
		/*
		int *center1, *center2;
		center1 = (int *) malloc((size_t) Nz*sizeof(int));
		center2 = (int *) malloc((size_t) Nz*sizeof(int));
		int  C1,C2;
		C1 = halfNx;
		C2 = halfNx + R1 + R2;

		//Setting the initial density profile.
		for (i1=0; i1<Nx; i1++)
		{
		        for (i2=0; i2<Ny; i2++)
		        {
		                for (i3=0; i3<Nz; i3++)
		                {
		                        c_host[i3 + Nz*(i2 + Ny*i1)] = 0.0;
		                }
		        }
		}

		for(i1=0; i1 < Nx; i1++)
		{
		        for(i2=0; i2 < Ny; i2++)
		        {
		                for (i3=0; i3<Nz; i3++)
		                {
		                        if ((i1 - C1)*(i1 - C1) + (i2 - halfNy)*(i2 - halfNy) < R1*R1)
		                        {
		                                c_host[i3 + Nz*(i2 + Ny*i1)] = c_zero;

		                        }
		                }
		        }
		}

		for (i2=0; i2 < Ny; ++i2)
		{
		        for (i3=0; i3 < Nz; ++i3)
		        {
		                if (fabs((i3 - halfNz) - (1/sqrt(3))*(i2 - halfNy)) <= (13*sqrt(3))/2)
		                {
		                        center1[i3] = i3;
		                        center2[i3] = i2;
		                }
		        }
		}

		for (i1 = 0; i1 < Nx; ++i1)
		{
		        for(i2=0; i2 < Ny; i2++)
		        {
		                for (i3=0; i3<Nz; i3++)
		                {
						if(i3 > 30 && i3 < Nz - 30 && i2 > 30 && i2 < Ny - 30)
						{

		                                                //if (round(pow(sin(M_PI/3),2)*(i3 - center1[i3])*(i3 - center1[i3]))+ round(pow(sin(M_PI/3),2)*(i2 - center2[i3])*(i2- center2[i3])) + (i1 - C2)*(i1 - C2) <= R2*R2)

								if (fabs((i2 - center2[i3])*(i2 - center2[i3])*1/4 + (i1 - C2)*(i1 - C2))< R2*R2)
								{
		                                                        c_host[i3 + Nz*(i2 + Ny*i1)] = c_zero;


		                                              }




						}

		                }
			}
		}
		// Free all the dynamically allocated variables for storing the central axis of different wire orientations
		free(center1);
		free(center2);
		*/
		//-------------------------------------------------------------------------------------------------------------------------

		// MULTIPLE JUNCTIONS
		/*
		for(i1=0; i1 < Nx; i1++)
	        {
	                for(i2=0; i2 < Ny; i2++)
	                {
	                        for (i3=0; i3<Nz; i3++)
	                        {
	                                if ((i1 - halfNx)*(i1 - halfNx) + (i2 - halfNy - halfNy/2)*(i2 - halfNy - halfNy/2) < R1*R1)
	                                {
	                                        c_host[i3 + Nz*(i2 + Ny*i1)] = c_zero;

	                                }
	                        }
	                }
	        }

		for(i1=0; i1 < Nx; i1++)
	        {
	                for(i2=0; i2 < Ny; i2++)
	                {
	                        for (i3=0; i3<Nz; i3++)
	                        {
	                                if ((i1 - halfNx)*(i1 - halfNx) + (i2 - halfNy)*(i2 - halfNy) < R1*R1)
	                                {
	                                        c_host[i3 + Nz*(i2 + Ny*i1)] = c_zero;

	                                }
	                        }
	                }
	        }

		for(i1=0; i1 < Nx; i1++)
	        {
	                for(i2=0; i2 < Ny; i2++)
	                {
	                        for (i3=0; i3<Nz; i3++)
	                        {
	                                if ((i1 - halfNx)*(i1 - halfNx) + (i2 - halfNy + halfNy/2)*(i2 - halfNy + halfNy/2) < R1*R1)
	                                {
	                                        c_host[i3 + Nz*(i2 + Ny*i1)] = c_zero;

	                                }
	                        }
	                }
	        }

		for(i1=0; i1 < Nx; i1++)
	        {
	                for(i2=0; i2 < Ny; i2++)
	                {
	                        for (i3=0; i3<Nz; i3++)
	                        {
	                                if ((i1 - (halfNx + R1 + R2))*(i1 - (halfNx + R1 + R2)) + (i3 - halfNz - halfNz/2)*(i3 - halfNz - halfNz/2) < R2*R2)
	                                {
	                                        c_host[i3 + Nz*(i2 + Ny*i1)] = c_zero;

	                                }
	                        }
	                }
	        }

		for(i1=0; i1 < Nx; i1++)
	        {
	                for(i2=0; i2 < Ny; i2++)
	                {
	                        for (i3=0; i3<Nz; i3++)
	                        {
	                                if ((i1 - (halfNx + R1 + R2))*(i1 - (halfNx + R1 + R2)) + (i3 - halfNz)*(i3 - halfNz) < R2*R2)
	                                {
	                                        c_host[i3 + Nz*(i2 + Ny*i1)] = c_zero;

	                                }
	                        }
	                }
	        }

		for(i1=0; i1 < Nx; i1++)
	        {
	                for(i2=0; i2 < Ny; i2++)
	                {
	                        for (i3=0; i3<Nz; i3++)
	                        {
	                                if ((i1 - (halfNx + R1 + R2))*(i1 - (halfNx + R1 + R2)) + (i3 - halfNz + halfNz/2)*(i3 - halfNz + halfNz/2) < R2*R2)
	                                {
	                                        c_host[i3 + Nz*(i2 + Ny*i1)] = c_zero;

	                                }
	                        }
	                }
	        }
		*/

		//-------------------------------------------------------------------------------------------------------------------------
		// ORIENTATION OF 45 DEGREES BETWEEN THE RODS
		/*
		int *center1, *center2;
		center1 = (int *) malloc((size_t) Nz*sizeof(int));
		center2 = (int *) malloc((size_t) Nz*sizeof(int));
		int C1, C2;
		C1 = halfNx;
		C2 = halfNx + R1 + R2;

		for(i1=0; i1 < Nx; i1++)
		{
			for(i2=0; i2 < Ny; i2++)
			{
				for (i3=0; i3<Nz; i3++)
				{
					if ((i1 - C1)*(i1 - C1) + (i2 - halfNy)*(i2 - halfNy) < R1*R1)
					{
						c_host[i3 + Nz*(i2 + Ny*i1)] = c_zero;

					}
				}
			}
		}

		for (i2=0; i2 < Ny; ++i2)
		{
			for (i3=0; i3 < Nz; ++i3)
			{
				if (i2 == i3)
				{
					center1[i3] = i3;
					center2[i3] = i2;
				}
			}
		}

		for (i1 = 0; i1 < Nx; ++i1)
		{
			for(i2=0; i2 < Ny; i2++)
			{
				for (i3=0; i3<Nz; i3++)
				{
						if(i3 > 30 && i3 < Nz - 30 && i2 > 30 && i2 < Ny - 30)
						{


								//if ((i3 - center1[i3])*(i3 - center1[i3]) + (i2 - center2[i3])*(i2 - center2[i3]) + (i1 - C2)*(i1 - C2)< R2*R2)
								if (fabs((i2 - center2[i3])*(i2 - center2[i3])/2 + (i1 - C2)*(i1 - C2))< R2*R2)
								{
									c_host[i3 + Nz*(i2 + Ny*i1)] = c_zero;
								}

						}

				}
			}
		}
		// Free all the dynamically allocated variables for storing the central axis of different wire orientations
		free(center1);
		free(center2);
		*/

		//-------------------------------------------------------------------------------------------------------------------------

		// ORIENTATION OF 90 DEGREES BETWEEN THE RODS
		/**/
	        for(i1=0; i1 < Nx; i1++)
	        {
	                for(i2=0; i2 < Ny; i2++)
	                {
	                        for (i3=0; i3<Nz; i3++)
	                        {
	                                if ((i1 - halfNx)*(i1 - halfNx) + (i2 - halfNy)*(i2 - halfNy) < R1*R1)
	                                {
	                                        c_host[i3 + Nz*(i2 + Ny*i1)] = c_zero;

	                                }
	                        }
	                }
	        }

	        for(i1=0; i1 < Nx; i1++)
	        {
	                for(i2=0; i2 < Ny; i2++)
	                {
	                        for (i3=0; i3<Nz; i3++)
	                        {
	                                if ((i1 - (halfNx + R1 + R2))*(i1 - (halfNx + R1 + R2)) + (i3 - halfNz)*(i3 - halfNz) < R2*R2)
	                                {
	                                        c_host[i3 + Nz*(i2 + Ny*i1)] = c_zero;

	                                }
	                        }
	                }
	        }

		//-------------------------------------------------------------------------------------------------------------------------
		// Introducing noise in the density variable

		// GSL Tausworthe random number generator.
		gsl_rng * ran_num;
		const gsl_rng_type * Taus;
		Taus = gsl_rng_taus2;
		ran_num = gsl_rng_alloc(Taus);
		gsl_rng_set(ran_num,seed_val);

		for (i1 = 0; i1 < Nx; ++i1)
		{
			for (i2 = 0; i2 < Ny; ++i2)
			{
				for (i3 = 0; i3 < Nz; ++i3)
				{
					// c_host[i3 + Nz*(i2 + Ny*i1)] += c_noise*(2.0*ran2(&SEED) - 1.0);
					c_host[i3 + Nx*(i2+Ny*i1)] += c_noise*(0.5 - gsl_rng_uniform_pos(ran_num));
				}

			}
		}
		//-------------------------------------------------------------------------------------------------------------------------
		// Writing the initial composition profile

		sprintf(file_name,"./output/time%d.dat", start_time);
		fp = fopen(file_name,"wb");
		fwrite(&c_host[0], sizeof(double),(size_t) Nx*Ny*Nz, fp);
		(void) fclose(fp);
		fflush(fp);

	}
	#endif
	//-------------------------------------------------------------------------------------------------------------------------
	// Define fft plan for CUFFT
	cufftHandle plan;
	cufftPlan3d(&plan, Nx, Ny, Nz, CUFFT_C2C);

	// Copy the initial configuration from the host to the device

	cudaMemcpy(c_device, c_host, sizeof(double)*Nx*Ny*Nz, cudaMemcpyHostToDevice);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess)
	{
		printf("Error 3: %s\n", cudaGetErrorString(err));
	}
	//-------------------------------------------------------------------------------------------------------------------------
	// 						Temporal evolution loop
	//-------------------------------------------------------------------------------------------------------------------------
	for (temp = start_time + 1; temp < end_time + 1; ++temp)
	{
		//-----------------------------------------------------------------------------------------------
		// Calculate the value of g
		compute_g <<<dimGrid, dimBlock>>> (c_device, g_device, A, Nx, Ny, Nz);
		cudaDeviceSynchronize();
		err = cudaPeekAtLastError();
		if (err != cudaSuccess)
		{
			printf("Error 4: %s\n", cudaGetErrorString(err));
		}
		//-----------------------------------------------------------------------------------------------
		compute_variable_mobility <<<dimGrid, dimBlock>>> (phi_device, c_device, Nx, Ny, Nz);
		cudaDeviceSynchronize();
		err = cudaPeekAtLastError();
		if (err != cudaSuccess)
		{
			printf("Error 5: %s\n", cudaGetErrorString(err));
		}
		//-----------------------------------------------------------------------------------------------
		// Move composition and g from real to complex on the device
		real2complex <<<dimGrid, dimBlock>>>(c_device_f, c_device, g_device_f, g_device, Nx, Ny, Nz);
		cudaDeviceSynchronize();
		err = cudaPeekAtLastError();
		if (err != cudaSuccess)
		{
			printf("Error 6: %s\n", cudaGetErrorString(err));
		}
		//-----------------------------------------------------------------------------------------------
		// Taking the variables comp and g from real to fourier space.
		cufftExecC2C(plan, c_device_f, c_device_f, CUFFT_FORWARD);
		cufftExecC2C(plan, g_device_f, g_device_f, CUFFT_FORWARD);
		compute_r <<<dimGrid, dimBlock>>> (r_device_f, c_device_f, g_device_f, kappa, k2_vect_device, Nx, Ny, Nz);
		cudaDeviceSynchronize();
		err = cudaPeekAtLastError();
		if (err != cudaSuccess)
		{
			printf("Error 7: %s\n", cudaGetErrorString(err));
		}
		//-----------------------------------------------------------------------------------------------
		compute_y <<<dimGrid, dimBlock>>> (y1_device_f, y2_device_f, y3_device_f, r_device_f, kx_vect_device, ky_vect_device, kz_vect_device, Nx, Ny, Nz);
		cudaDeviceSynchronize();
		err = cudaPeekAtLastError();
		if (err != cudaSuccess)
		{
			printf("Error 8: %s\n", cudaGetErrorString(err));
		}
		//-----------------------------------------------------------------------------------------------
		cufftExecC2C(plan, y1_device_f, y1_device_f, CUFFT_INVERSE);
		cufftExecC2C(plan, y2_device_f, y2_device_f, CUFFT_INVERSE);
		cufftExecC2C(plan, y3_device_f, y3_device_f, CUFFT_INVERSE);
		//-----------------------------------------------------------------------------------------------
		complex2real_y <<<dimGrid, dimBlock>>> (y1_device_f, y1_device, y2_device_f, y2_device, y3_device_f, y3_device, Nx, Ny, Nz);
		cudaDeviceSynchronize();
		err = cudaPeekAtLastError();
		if (err != cudaSuccess)
		{
			printf("Error 9: %s\n", cudaGetErrorString(err));
		}
		//-----------------------------------------------------------------------------------------------
		compute_q <<<dimGrid, dimBlock>>> (qx_device_f, qy_device_f, qz_device_f, phi_device, y1_device, y2_device, y3_device, M, Nx, Ny, Nz);
		cudaDeviceSynchronize();
		err = cudaPeekAtLastError();
		if (err != cudaSuccess)
		{
			printf("Error 10: %s\n", cudaGetErrorString(err));
		}
		//-----------------------------------------------------------------------------------------------
		cufftExecC2C(plan, qx_device_f, qx_device_f, CUFFT_FORWARD);
		cufftExecC2C(plan, qy_device_f, qy_device_f, CUFFT_FORWARD);
		cufftExecC2C(plan, qz_device_f, qz_device_f, CUFFT_FORWARD);
		//-----------------------------------------------------------------------------------------------
		// Solve the variable mobility cahn hilliard equation
		solve_cahn_hilliard <<<dimGrid, dimBlock>>>(c_device_f, qx_device_f, qy_device_f, qz_device_f, k2_vect_device, kx_vect_device, ky_vect_device, kz_vect_device, Nx, Ny, Nz, kappa, alpha, dt);
		cudaDeviceSynchronize();
		err = cudaPeekAtLastError();
		if (err != cudaSuccess)
		{
			printf("Error 11: %s\n", cudaGetErrorString(err));
		}
		//-----------------------------------------------------------------------------------------------
		// Bring composition back to real space
		cufftExecC2C(plan, c_device_f, c_device_f, CUFFT_INVERSE);
		//-----------------------------------------------------------------------------------------------
		// Complex to real of both c and g
		complex2real <<<dimGrid, dimBlock>>> (c_device_f, c_device, Nx, Ny, Nz);
		cudaDeviceSynchronize();
		err = cudaPeekAtLastError();
		if (err != cudaSuccess)
		{
			printf("Error 12: %s\n", cudaGetErrorString(err));
		}
		//-----------------------------------------------------------------------------------------------
		if(temp%time_step == 0)
		{
			// Copy the composition value from the device to the host for writing
			cudaMemcpy(c_host, c_device, sizeof(double)*Nx*Ny*Nz, cudaMemcpyDeviceToHost);

			// Change the directory path to "./output/time%d.dat" for linux system
			sprintf(file_name, "./output/time%d.dat", temp);

			//-----------------------------------------------------------------------------------------------
			fp = fopen(file_name,"wb");
			fwrite(&c_host[0], sizeof(double),(size_t) Nx*Ny*Nz, fp);
			(void) fclose (fp);
			fflush(fp);

		}

	}
	//-------------------------------------------------------------------------------------------------------------------------
	// 						Temporal evolution loop ends
	//-------------------------------------------------------------------------------------------------------------------------

	cufftDestroy(plan);

	// Free all the dynamically allocated variables on the host
	free(c_host);
	free(k2_vect_host);
	free(kx_vect_host);
	free(ky_vect_host);
	free(kz_vect_host);


	// Free all the dynamically allocated variables on the device

	cudaFree(c_device);
	cudaFree(g_device);
	cudaFree(phi_device);
	cudaFree(y1_device);
	cudaFree(y2_device);
	cudaFree(y3_device);
	cudaFree(k2_vect_device);
	cudaFree(kx_vect_device);
	cudaFree(ky_vect_device);
	cudaFree(kz_vect_device);
	cudaFree(c_device_f);
	cudaFree(g_device_f);
	cudaFree(r_device_f);
	cudaFree(y1_device_f);
	cudaFree(y2_device_f);
	cudaFree(y3_device_f);
	cudaFree(qx_device_f);
	cudaFree(qy_device_f);
	cudaFree(qz_device_f);

	printf("\nCode execution has completed.\n");

	// Calculation of the total simulation time required
	end_t = time(NULL);
	time_elapsed = (double) (end_t - begin_t);
	printf("\nThe total simulation time elapsed = %f\n", time_elapsed);

	return 0;
}
//-------------------------------------------------------------------------------------------------------------------------
//							END OF CODE
//-------------------------------------------------------------------------------------------------------------------------
