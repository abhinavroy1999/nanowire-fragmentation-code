// Function to compute the value of g function (derivative of bulk free energy density function)

__global__ void compute_g(double *c_device, double *g_device,double A, int Nx, int Ny, int Nz)
{

        int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
        int index = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
        while(index < Nx*Ny*Nz)
        {
                g_device[index] =  2.0*A*c_device[index]*(1.0 - c_device[index])*(1.0 - 2.0*c_device[index]);
                index+=blockDim.x*gridDim.x*blockDim.y*gridDim.y;

        }
}
