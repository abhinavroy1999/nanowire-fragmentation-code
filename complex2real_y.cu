// Function to take the variables y1 and y2 from complex space to real space.

__global__ void complex2real_y (cufftComplex *y1_device_f, double *y1_device, cufftComplex *y2_device_f, double *y2_device, cufftComplex *y3_device_f, double *y3_device, int Nx, int Ny, int Nz)
{

                int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
                int index = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

                while(index < Nx*Ny*Nz)
                {
                        y1_device[index] = y1_device_f[index].x/(Nx*Ny*Nz);
                        y2_device[index] = y2_device_f[index].x/(Nx*Ny*Nz);
                        y3_device[index] = y3_device_f[index].x/(Nx*Ny*Nz);
                        index+=blockDim.x*gridDim.x*blockDim.y*gridDim.y;

                }
}
