// Function to take the composition variable from complex space to real space after normalizing

__global__ void complex2real (cufftComplex *c_device_f, double *c_device, int Nx, int Ny, int Nz)
{

        int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
        int index = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

        while(index < Nx*Ny*Nz)
        {
                c_device[index] = c_device_f[index].x/(Nx*Ny*Nz);
                index+=blockDim.x*gridDim.x*blockDim.y*gridDim.y;

        }

}
