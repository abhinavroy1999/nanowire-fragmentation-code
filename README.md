<div>
<a name="logo"/>
<div align="center">
<img src="assets/background.svg" alt="Nanowire Logo" width="512" height="384"></img>
</a>
</div>
This repository contains simulation code for surface diffusion enhanced nanowire junction break-up. The code is written in CUDA C (GPU accelerated parallel code for Nvidia GPUs). Pre-print of the research article where this code was used can be found here: http://arxiv.org/abs/2107.01801.

The contents of this repository are described below: -
- **inputs**: Constains the following files: -
    - _comp\_data.txt_: Contains composition value of inside the nanowires, along with noise in the composition profile.
    - _parameters\_data.txt_: Contains simulation parameters such as mobility, free energy barrier, gradient energy coefficient, and stability factor for the numerical scheme.
    - _radius\_data.txt_: The two nanowire radii values (can be different in case crossing nanowires, and equal in case of single nanowire configuration).
    - _system\_data.txt_: Contains the system size data along with the grid spacing used for the structured mesh.
    - _time\_data.txt_: Contains the simulation time along with the user time-step (intervals at which to print the output) and the numerical scheme time-step. 
- **visualize**: Contains the following files: -
    - _vtkfile.c_: Code for converting the binary output time-series data into [VTK datasets](https://docs.paraview.org/en/latest/UsersGuide/understandingData.html#vtk-data-model).   
