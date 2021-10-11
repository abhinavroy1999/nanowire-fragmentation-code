#! /usr/bin/bash
printf "\nThe visualization code has commenced\n"
gcc vtkfile.c -o vtkfile.o -D$1
./vtkfile.o 
printf "\nThe visualization code execution has completed\n"
