<div>
<a name="logo"/>
<div align="center">
<img src="assets/background.svg" alt="Nanowire Logo" width="512" height="384"></img>
</a>
</div>
This repository contains simulation code for surface diffusion enhanced nanowire junction break-up. The code is written in CUDA C (GPU accelerated parallel code for Nvidia GPUs). Pre-print of the research article where this code was used can be found here: http://arxiv.org/abs/2107.01801.

#### Directory contents ####
The contents of this repository are described below: -

- **inputs**: Constains the following files: -
    - _comp\_data.txt_: Contains composition value of inside the nanowires, along with noise in the composition profile.
    - _parameters\_data.txt_: Contains simulation parameters such as mobility, free energy barrier, gradient energy coefficient, and stability factor for the numerical scheme.
    - _radius\_data.txt_: The two nanowire radii values (can be different in case crossing nanowires, and equal in case of single nanowire configuration).
    - _system\_data.txt_: Contains the system size data along with the grid spacing used for the structured mesh.
    - _time\_data.txt_: Contains the simulation time along with the user time-step (intervals at which to print the output) and the numerical scheme time-step. 
- **visualize**: Contains the following files: -
    - _vtkfile.c_: Code for converting the binary output time-series data into [VTK datasets](https://docs.paraview.org/en/latest/UsersGuide/understandingData.html#vtk-data-model).   
    - _visualize.sh_: Script for compiling and executing the _vtkfile_ code.
- **src**: Contains the source cuda C files. The main code file is _nanowire3D.cu_. All other code files are function files which _nanowire3D.cu_ file calls for different calculations.
- **mayavi_Plot3D**: Contains _mayavi\_visualization.py_ script for converting the binary output time-series data files intp 3D plots directly. The output is stored in the _plots3D/_ directory in the form of PNG image files. The image files are also combined to form an animated GIF files names _animation.gif_.

#### Dependencies ####
The code is written for Linux based operating systems with Nvidia CUDA enabled GPUs. The following are list of dependencies which are used in the system: -

- cuFFT
- gcc
- GSL
- [Paraview](https://www.paraview.org/) (for visualization and data analysis of VTK datasets)
- Python3 (preferably Python3.7)
- Mayavi (for generating 3D plots directly)

**NOTE:** The code can also be run in windows operating system. In this case the bash script files cannot be used, rather the user has to compile the code based on the compilation commands given in the assets/ directory without the -lgsl and -lm flags, and a suitable random number generator.
