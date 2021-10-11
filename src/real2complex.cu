// Function to port from real to complex
__global__ void real2complex(cufftComplex *c_device_f, double *c_device, cufftComplex *g_device_f, double *g_device, int Nx, int Ny, int Nz)
{
        int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
        int index = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

        while(index < Nx*Ny*Nz)
        {
                c_device_f[index].x = c_device[index];
                c_device_f[index].y = 0.0f;
                g_device_f[index].x = g_device[index];
                g_device_f[index].y = 0.0f;

                // To ensure unique thread execution
                index+=blockDim.x*gridDim.x*blockDim.y*gridDim.y;
        }
}
