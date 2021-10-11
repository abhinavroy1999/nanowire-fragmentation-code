/**************************************************************************
Code to convert the data file into VTK file so as to plot the 3D morphology.

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
**************************************************************************/
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int main(int argc, char** argv)
{
	#ifdef NEW
	{
		system("rm -rf ./vtk_data/*");
	}
	#endif
	FILE *fp, *fw;
	int i1, i2, i3;
	int Nx, Ny, Nz;
	double *c;
	char file_name[100];
	char vtk_file_name[100];
	int start_time, end_time, time_step;

	// Variables initialization for the simulation
	if ((fp = fopen("../inputs/time_data.txt","r")) == NULL)
	{
		printf("Unable to open the time data input file. Exiting!\n");
	}
	else
	{
		fp = fopen("../inputs/time_data.txt","r");
	}
	(void) fscanf(fp,"%d%d%d", &start_time, &end_time, &time_step);
	(void) fclose(fp);
	int temp = 0;

	// To be determined from the simulation data.
	Nx = 96;
	Ny = 768;
	Nz = 768;
	//--------------------------------------------------------------------//
	// Conversion loop begins.

	for (temp = start_time; temp < end_time + 1; temp+=time_step)
	{
		c = (double*)malloc((size_t)Nx*Ny*Nz* sizeof(double));
		sprintf(file_name, "../output/time%d.dat", temp);
		sprintf(vtk_file_name, "./vtk_data/time_%d.vtk", temp);
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
		// Loop for writing composition data to VTK file.
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
	return 0;
}
