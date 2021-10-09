#! /usr/bin/bash
printf "The visualization code has commenced\n"
gcc vtkfile.c -o vtkfile.o
./vtkfile.o
printf "\nThe visualization code execution has completed\n"
