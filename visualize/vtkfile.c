/*-------------------------------------------------------------------------
Code to convert the data file into VTK file so as to plot the 3D morphology.
-------------------------------------------------------------------------*/
#include<stdio.h>
#include<stdlib.h>

// # define RESTART

int main(void)
{
	#ifndef RESTART
	{
		system("rm -rf data_vtk/*");
	}
	#endif
	FILE *fp, *fw;
	int i1, i2, i3;
	int Nx, Ny, Nz;
	double *c;
	char file_name[50];
	char vtk_file_name[50];

	int start_time, end_time, time_step;
	int temp = 0;
	start_time = 0;
	time_step = 1;
	end_time = 50;	//To be determined from the simulation data.
	Nx = 64;
	Ny = 512;
	Nz = 512;		//To be determined from the simulation data.
	//--------------------------------------------------------------------//
	//Conversion loop begins.

	for (temp = start_time; temp < end_time + 1; temp += time_step)
	{
		c = (double*)malloc((size_t) Nx*Ny*Nz* sizeof(double));

		sprintf(file_name, "../output/time%d.dat", temp);
		sprintf(vtk_file_name, "data_vtk/time_vtk%d.vtk", temp);

		if ((fp = fopen(file_name,"r")) == NULL)
		{
			printf("Unable to open data file. Exiting.");
			exit(0);
		}
		else
		{
			fp = fopen(file_name,"r");
		}
		fread(&c[0], sizeof(double),(size_t) Nx*Ny*Nz, fp);
		(void) fclose(fp);
		fflush(fp);

		if ((fw = fopen(vtk_file_name,"w")) == NULL)
		{
			printf("Unable to open VTK file. Exiting.");
			exit(0);
		}
		else
		{
			fw = fopen(vtk_file_name,"w");
		}

		//Preamble of the VTK file.

		fprintf(fw,"# vtk DataFile Version 3.0\n");
		fprintf(fw,"composition field data\n");
		fprintf(fw,"ASCII\n");
		fprintf(fw,"DATASET STRUCTURED_POINTS\n");
		fprintf(fw,"DIMENSIONS %d %d %d\n",Nz,Ny,Nx);
		fprintf(fw,"ORIGIN 0 0 0\n");
		fprintf(fw,"SPACING 1 1 1\n");
		fprintf(fw,"POINT_DATA %d\n",Nx*Ny*Nz);
		fprintf(fw,"SCALARS density_field double\n");
		fprintf(fw,"LOOKUP_TABLE default\n");

		for (i1 = 0; i1<Nx; ++i1)
		{
			for (i2 = 0; i2<Ny; ++i2)
			{
				for (i3 = 0; i3<Nz; ++i3)
				{
					fprintf(fw,"%le\n", c[i3 + Nz*(i2 + Ny*i1)]);
				}
			}
		}
		(void) fclose(fw);
		fflush(fw);

		free(c);
	}
}
