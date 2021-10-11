import numpy as np
from mayavi import mlab
from tvtk.api import tvtk
import moviepy.editor as mpy

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


