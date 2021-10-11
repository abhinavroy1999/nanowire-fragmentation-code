// Function to compute the value of qx and qy function (which is a function of the mobility function and y1 and y2 function)

__global__ void compute_q (cufftComplex *qx_device_f, cufftComplex *qy_device_f, cufftComplex *qz_device_f, double *phi_device, double *y1_device, double *y2_device,  double *y3_device, double M, int Nx, int Ny, int Nz)
{


        int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
        int index = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
        while(index < Nx*Ny*Nz)
        {
                qx_device_f[index].x =  M*phi_device[index]*y1_device[index];
                qx_device_f[index].y =  0.0f;
                qy_device_f[index].x =  M*phi_device[index]*y2_device[index];
                qy_device_f[index].y =  0.0f;
                qz_device_f[index].x =  M*phi_device[index]*y3_device[index];
                qz_device_f[index].y =  0.0f;
                index+=blockDim.x*gridDim.x*blockDim.y*gridDim.y;

        }
}
