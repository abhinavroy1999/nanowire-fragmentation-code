printf "\nThe nanowire 3D simulation using CUDA\n"
nvcc nanowire3D.cu -gencode arch=compute_75,code=[sm_75,compute_75] -lcufft -lgsl -lm
./a.out
cd visualize/
./visualize.sh
printf "\nThe code execution has commenced\n"
