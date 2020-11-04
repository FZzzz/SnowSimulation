#ifndef _SIM_PARAMS_H_
#define _SIM_PARAMS_H_

#include <cuda_runtime.h>
#include <helper_math.h>

struct SimParams
{
	float3 gravity;
	float  global_damping;
	float  maximum_speed;
	float  minimum_speed;

	// pbf sph coeffcients
	float  epsilon;
	float  pbd_epsilon;
	float  kernel_epsilon;
	float  effective_radius;
	float  particle_radius;
	float  rest_density;
	float  scorr_coeff;
	float scorr_divisor;

	// dem coeffcients
	float  boundary_damping;
	float  static_friction;
	float  kinematic_friction;
	
	// heat conduction coeffcients
	float  k_water;
	float  k_snow;
	float  C_water;
	float  C_snow;
	float  freezing_point;

	float  blending_speed;

	float  sor_coeff;
	float  sph_viscosity;
	float  dem_viscosity;

	float3 world_origin;
	float3 cell_size;

	uint3  grid_size;
	uint   num_cells;

	float poly6;
	float poly6_G;
	float spiky;
	float spiky_G;
	float viscosity_laplacian;
};


#endif
