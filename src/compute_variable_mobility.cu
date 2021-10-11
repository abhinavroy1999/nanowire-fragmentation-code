// Function to compute the variable mobility function
__global__ void compute_variable_mobility (double *phi_device, double *c_device, int Nx, int Ny, int Nz)
{
        int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
        int index = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
        while(index < Nx*Ny*Nz)
        {
                phi_device[index] = sqrt(fabs(c_device[index] - c_device[index]*c_device[index]));
                index+=blockDim.x*gridDim.x*blockDim.y*gridDim.y;

        }
}
