#ifndef _SIM_PARAMS_H_
#define _SIM_PARAMS_H_

#include <cuda_runtime.h>
#include <helper_math.h>

struct SimParams
{
	float3 gravity;
	float  global_damping;
	float  maximum_speed;

	// pbf sph coeffcients
	float  epsilon;
	float  pbd_epsilon;
	float  effective_radius;
	float  particle_radius;
	float  rest_density;
	float  scorr_coeff;

	// dem coefficients
	float  boundary_damping;
	float  static_friction;
	float  kinematic_friction;

	// wetness coefficients
	float wetness_threshold;
	float wetness_max;
	float k_p; // propagation coefficient
	float k_bridge;


	float  sor_coeff;
	float  viscosity;

	float3 world_origin;
	float3 cell_size;

	uint3  grid_size;
	uint   num_cells;

	float poly6;
	float poly6_G;
	float spiky;
	float spiky_G;
};


#endif
