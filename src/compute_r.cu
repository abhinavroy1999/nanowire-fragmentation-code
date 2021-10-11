// Function to compute the value of r function which is a function of g and composition function

__global__ void compute_r (cufftComplex *r_device_f, cufftComplex *c_device_f, cufftComplex *g_device_f, double kappa, double *k2_vect_device, int Nx, int Ny, int Nz)
{

        int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
        int index = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

        while(index < Nx*Ny*Nz)
        {

                r_device_f[index].x = g_device_f[index].x + kappa*k2_vect_device[index]*c_device_f[index].x;
                r_device_f[index].y = g_device_f[index].y + kappa*k2_vect_device[index]*c_device_f[index].y;

                index+=blockDim.x*gridDim.x*blockDim.y*gridDim.y;
        }
}
