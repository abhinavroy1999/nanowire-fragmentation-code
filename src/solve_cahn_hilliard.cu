// Function to solve the Cahn-Hilliard equation and save in the complex variable

__global__ void solve_cahn_hilliard(cufftComplex *c_device_f, cufftComplex *qx_device_f, cufftComplex *qy_device_f, cufftComplex *qz_device_f, double *k2_vect_device, double *kx_vect_device, double *ky_vect_device, double *kz_vect_device, int Nx, int Ny, int Nz, double kappa, double alpha, double dt)
{
        int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
        int index = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
        double k4, denominator;
        while(index < Nx*Ny*Nz)
        {
                k4 = k2_vect_device[index]*k2_vect_device[index];
                // Isotropic formulation segment
                denominator = (1.0 + alpha*dt*kappa*k4);
                c_device_f[index].x = c_device_f[index].x - (dt*(kx_vect_device[index]*qx_device_f[index].y + ky_vect_device[index]*qy_device_f[index].y + kz_vect_device[index]*qz_device_f[index].y))/denominator;
                c_device_f[index].y = c_device_f[index].y + (dt*(kx_vect_device[index]*qx_device_f[index].x + ky_vect_device[index]*qy_device_f[index].x + kz_vect_device[index]*qz_device_f[index].x))/denominator;

                index+=blockDim.x*gridDim.x*blockDim.y*gridDim.y;
        }
}
