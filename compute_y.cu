// Function to compute the y1 and y2 functions

__global__ void compute_y (cufftComplex *y1_device_f, cufftComplex *y2_device_f, cufftComplex *y3_device_f, cufftComplex *r_device_f, double *kx_vect_device, double *ky_vect_device, double *kz_vect_device, int Nx, int Ny, int Nz)
{

        int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
        int index = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
        while(index < Nx*Ny*Nz)
        {
                y1_device_f[index].x = - kx_vect_device[index]*r_device_f[index].y;
                y1_device_f[index].y =   kx_vect_device[index]*r_device_f[index].x;
                y2_device_f[index].x = - ky_vect_device[index]*r_device_f[index].y;
                y2_device_f[index].y =   ky_vect_device[index]*r_device_f[index].x;
                y3_device_f[index].x = - kz_vect_device[index]*r_device_f[index].y;
                y3_device_f[index].y =   kz_vect_device[index]*r_device_f[index].x;
                index+=blockDim.x*gridDim.x*blockDim.y*gridDim.y;

        }
}
