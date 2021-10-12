<div>
<a name="logo"/>
<div align="center">
<img src="assets/background.svg" alt="Nanowire Logo" width="512" height="384"></img>
</a>
</div>
This repository contains simulation code for surface diffusion enhanced nanowire junction break-up. The code is written in CUDA C (GPU accelerated parallel code for Nvidia GPUs). Pre-print of the research article where this code was used can be found here: http://arxiv.org/abs/2107.01801. The code is based on the variable mobility Cahn-Hilliard (VMCH) formulation. The details of VMCH algorithm can be found at the following DOI: https://doi.org/10.13140/RG.2.2.29738.13760/4.

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
- **output**: Contains output binary time-series data files generated from a particular simulation run. Subsequent simulation runs overwrite files into this directory and delete files from the previous run. Hence, users are advised to store the generated simulation data to a different folder for future use.

#### Dependencies ####
The code is written for Linux based operating systems with Nvidia CUDA enabled GPUs. The following are list of dependencies which are used in the code: -

- [cuFFT](https://docs.nvidia.com/cuda/cufft/index.html)
- [gcc](https://gcc.gnu.org/)
- [GSL](https://www.gnu.org/software/gsl/)
- [Paraview](https://www.paraview.org/) (for visualization and data analysis of VTK datasets)
- [Python3 (preferably Python3.7)](https://www.python.org/downloads/release/python-379/)
- [Mayavi](https://docs.enthought.com/mayavi/mayavi/) (for generating 3D plots directly)

#### Code execution ####
The compilation and execution of the simulation code is handled by the _nanowire\_cuda.sh_ script (to be executed from the *nanowireCUDA3D/* directory). 

The command format: _./nanowire\_cuda.sh ARG1 ARG2_

The following command line arguments are to be used for different configurations: -

First command line argument:

- _NEW_: For a fresh simulation run.
- _RESTART_: For a restarted simulation run.

Second command line argument:

- _DEG90_: 90 degree configuration between two crossing nanowires.
- _DEG60_: 60 degree configuration between two crossing nanowires.
- _DEG45_: 45 degree configuration between two crossing nanowires.
- _DEG30_: 30 degree configuration between two crossing nanowires.
- _SINGLE_WIRE_: For single nanowire configuration.
- _MULT_JUNC_: For nanowire grid consisting of 9 junctions.

A sample command for a *new* simulation for *90 degree* configuration is as follows: 
```bash
./nanowire_cuda.sh NEW DEG90 
```

#### Post-processing ####
The binary simulation output time-series data in the *output/* directory can be used for all sorts of post-processing operations. The binary data converted into VTK datasets in the *visualize/vtk_data/* directory can be loaded into *Paraview* directly and visualized. Paraview also provides a lot of filters for data analysis purposes.

Another option is to make on the fly plots directly from the binary data generated. This can be achieved using the *mayavi_visualization.py* script in the *mayavi_plots3D/* directory. To generate the plots, type the following command after the simulation run has finished:

```bash
cd mayavi_plots3D/
python3 mayavi_visualization.py
```
Output in the form of PNG snapshots will be stored in *mayavi_plots3D/plots3D/* directory. Also, an animated gif of the entire simulation named *animation.gif* will be stored in the *mayavi_plots3D/* directory.

**NOTE:** 
- For the _RESTART_ simulation type, make sure to keep the binary data in the *output/* directory corresponding to the last time-step from which simulation needs to be restarted. Accordingly, updated the *start_time* in the _inputs/time_data.txt_ file to this last time-step from which to restart the simulation. 
- The code can also be run in windows operating system. In that case the bash script files cannot be used, rather the user has to compile the code manually using the compilation commands provided in the assets/ directory without the -lgsl and -lm flags, and with a suitable random number generator for windows.
