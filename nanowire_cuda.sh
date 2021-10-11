printf "\nThe nanowire fragmentation 3D simulation using CUDA:\n"
nvcc ./src/nanowire3D.cu -o nanowire3D.o -gencode arch=compute_75,code=[sm_75,compute_75] -lcufft -lgsl -lm -D$1 -D$2
./nanowire3D.o
echo "The simulation type is: $1" >> logfile.txt
echo "The simulation configuration is: $2" >> logfile.txt
cd visualize/
./visualize.sh $1
printf "\nThe code execution has commenced\n"
