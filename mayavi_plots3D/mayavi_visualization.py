'''
Script to read the binary output data files (in binary time-series format), 
and print 3D plots using Mayavi Mlab module. The dataset is converted to 
tvtk.StructuredPoints format and thereafter plotted and saved as PNG figure files.

The figure files are also compiled to print an animated GIF in the current directory
using the MoviePy module.

Copyright (C) 2020  Abhinav Roy, Arjun Varma R, M.P. Gururajan

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
import numpy as np
from mayavi import mlab
from tvtk.api import tvtk
import moviepy.editor as mpy
import os

os.system('rm -rf plots3D/*')
mlab.options.offscreen = True
fig = mlab.figure(size=(1024,1024), bgcolor=(0.3,0.3,0.5))

Nx = 96
Ny = 768
Nz = 768

with open("../inputs/time_data.txt") as my_file:
	time_info = my_file.read().splitlines()
	
comp = np.zeros((Nx,Ny,Nz))
start_time = int(time_info[0])
end_time = int(time_info[1])
time_step = int(time_info[2])
file_list = []

for temp in range(start_time, end_time+1, time_step):
	mlab.clf()
	infile = "../output/time%d.dat" % temp
	data = np.fromfile(infile, dtype=np.float64)
	for i in range(Nx):
		for j in range(Ny):
			for k in range(Nz):
				comp[i,j,k] = data[k + Nz*(j + Ny*i)]
	obj = tvtk.StructuredPoints(origin=(0,0,0), spacing=(1,1,1), dimensions=(Nx,Ny,Nz))
	obj.point_data.scalars = comp.T.ravel()
	obj.point_data.scalars.name = 'scalars'
	src = mlab.pipeline.add_dataset(obj)
	surf = mlab.pipeline.iso_surface(src, colormap='PuOr')
	# mlab.scalarbar(object=surf, orientation='vertical', title='composition_field')
	mlab.view(azimuth=0, elevation=90, distance=2000)
	mlab.outline()
	mlab.orientation_axes()
	outfile = "./plots3D/image%d.png" % temp
	file_list.append(outfile)
	mlab.savefig(outfile)
	
total_frames = len(file_list)
duration = 5
fps = total_frames / duration
clip_duration = duration / total_frames
clips = [mpy.ImageClip(m).set_duration(clip_duration) for m in file_list]

concat_clip = mpy.concatenate_videoclips(clips, method='compose')
concat_clip.write_gif("animation.gif", fps=fps)


