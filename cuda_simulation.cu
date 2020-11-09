#include <cuda_runtime.h>
#include <cstdlib>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <device_atomic_functions.h>
#include <helper_math.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <cooperative_groups.h>
#include "cuda_simulation.cuh"
#include "cuda_tool.cuh"
#include <chrono>
#include "imgui/imgui.h"

#define NUM_CELLS 262144

namespace cg = cooperative_groups;

/*SPH Kernels*/
inline __device__ float sph_kernel_poly6(const float& distance, const float& effective_radius)
{
	if (distance >= 0 && distance < effective_radius)
	{
		const float h = effective_radius;
		const float d = distance;

		float h2 = h * h;
		//float h9 = pow(h, 9);
		float d2 = d * d;
		float q = h2 - d2;
		float q3 = q * q * q;

		//float result = (315.0f / (64.0f * CUDART_PI * h9)) * q3;
		float result = params.poly6 * q3;

		return result;
	}
	else
	{
		return 0.0f;
	}
}

inline __device__ float3 sph_kernel_poly6_gradient(const float3& diff, const float& distance, const float& effective_radius)
{
	if (distance >= 0 && distance < effective_radius)
	{
		const float h = effective_radius;

		float h2 = h * h;
		//float h9 = pow(h, 9);
		float d2 = distance * distance;
		float  q = h2 - d2;
		float q2 = q * q;

		//float scalar = (-945.0f / (32.0f * CUDART_PI * h9));
		float scalar = params.poly6_G;
		scalar = scalar * q2;
		float3 result = scalar * diff;// make_float3(diff.x, diff.y, diff.z);

		return result;
	}
	else
	{
		return make_float3(0, 0, 0);
	}
}


inline __device__ float sph_kernel_spiky(const float& distance, const float& effective_radius)
{
	if (distance >= 0 && distance < effective_radius)
	{
		const float h = effective_radius;
		const float d = distance;

		//float h6 = pow(h, 6);
		float q = h - d;
		float q3 = q * q * q;

		//float result = ((15.0f / (CUDART_PI * h6)) * q3);
		float result = params.spiky * q3;

		return result;
	}
	else
	{
		return 0.0f;
	}
}

inline __device__ float3 sph_kernel_spiky_gradient(const float3& diff, const float& distance, const float& effective_radius)
{
	if (distance >= 0 && distance < effective_radius)
	{
		const float h = effective_radius;
		//float h6 = pow(h, 6);
		float q = h - distance;
		float q2 = q * q;

		//float scalar = (-45.0f / (CUDART_PI*h6)) * (q2 / distance);
		float scalar = params.spiky_G * (q2 / (distance + params.kernel_epsilon));
		float3 result = scalar * diff;// make_float3(diff.x, diff.y, diff.z);

		return result;
	}
	else
	{
		return make_float3(0, 0, 0);
	}
}

inline __device__ float sph_kernel_viscosity_laplacian(const float &d, const float& h)
{
	if (d >= 0 && d < h)
	{
		float q = h - d;

		float result = params.viscosity_laplacian * q;

		return result;
	}
	else
	{
		return 0.0f;
	}
}


// calculate position in uniform grid
inline __device__ int3 calcGridPos(float3 p)
{
	int3 gridPos;
	gridPos.x = floor((p.x - params.world_origin.x) / params.cell_size.x);
	gridPos.y = floor((p.y - params.world_origin.y) / params.cell_size.y);
	gridPos.z = floor((p.z - params.world_origin.z) / params.cell_size.z);
	return gridPos;
}

// calculate address in grid from position (clamping to edges)
inline __device__ uint calcGridHash(int3 gridPos)
{
	gridPos.x = gridPos.x & (params.grid_size.x - 1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (params.grid_size.y - 1);
	gridPos.z = gridPos.z & (params.grid_size.z - 1);
	return __umul24(__umul24(gridPos.z, params.grid_size.y), params.grid_size.x) + __umul24(gridPos.y, params.grid_size.x) + gridPos.x;
}

inline __device__
float sph_boundary_volume(
	int3 grid_pos,
	uint index,
	float3 pos1,
	float* mass,
	CellData data
)
{
	uint grid_hash = calcGridHash(grid_pos);

	uint start_index = data.cell_start[grid_hash];

	float rho = 0.f;

	if (start_index != 0xffffffff)
	{
		uint end_index = data.cell_end[grid_hash];

		for (uint j = start_index; j < end_index; ++j)
		{
			if (j != index)
			{
				uint original_index = data.grid_index[j];
				float3 pos2 = data.sorted_pos[j];
				float3 vec = pos1 - pos2;
				float dist = length(vec);
				rho += mass[original_index] * sph_kernel_poly6(dist, params.effective_radius);
			}
		}
	}

	return rho;
}

__global__ void calcHashD(
	CellData cell_data,			// output		
	float3* pos,		        // input: positions
	uint    num_particles)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= num_particles) return;

	volatile float3 p = pos[index];

	// get address in grid
	int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
	uint hash = calcGridHash(gridPos);

	// store grid hash and particle index
	cell_data.grid_hash[index] = hash;
	cell_data.grid_index[index] = index;
}

__global__ 
void calcHash_boundary_D(
	CellData cell_data,
	float3* pos,               // input: positions
	uint    num_particles)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= num_particles) return;

	//printf("%u \n", index);

	volatile float3 p = pos[index];

	// get address in grid
	int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
	uint hash = calcGridHash(gridPos);

	// store grid hash and particle index
	cell_data.grid_hash[index] = hash;
	cell_data.grid_index[index] = index;
}



/*
 * Reorder data to find cell start and end (for neighbor searching)
 */
__global__
void reorderDataAndFindCellStartD(
	CellData cell_data,
	float3* oldPos,           // input: sorted position array
	uint    numParticles)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	uint hash;

	// handle case when no. of particles not multiple of block size
	if (index < numParticles)
	{
		hash = cell_data.grid_hash[index];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = cell_data.grid_hash[index - 1];
		}
	}

	cg::sync(cta);

	if (index < numParticles)
	{
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

		if (index == 0 || hash != sharedHash[threadIdx.x])
		{
			cell_data.cell_start[hash] = index;

			if (index > 0)
				cell_data.cell_end[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1)
		{
			cell_data.cell_end[hash] = index + 1;
		}

		// Now use the sorted index to reorder the pos and vel data
		uint sortedIndex = cell_data.grid_index[index];
		float3 pos = oldPos[sortedIndex];

		cell_data.sorted_pos[index] = pos;
	}
}

__global__
void compute_boundary_volume_d(
	CellData data, 
	float* mass, float* volume, 
	uint numParticles)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	uint originalIndex = data.grid_index[index];

	// read particle data from sorted arrays
	float3 pos = data.sorted_pos[index];

	// initial volume
	float rho = mass[originalIndex] * sph_kernel_poly6(0, params.effective_radius);

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	// traverse 27 neighbors
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				rho += sph_boundary_volume(
					neighbor_pos, index,
					pos, mass,
					data
				);
			}
		}
	}

	// Update volume
	volume[originalIndex] = mass[originalIndex] / rho;

	//printf("rho = %f\n", rho);
	//printf("C[%u]: %f\n", originalIndex, C[originalIndex]);
}

inline void compute_grid_size(uint n, uint block_size, uint& num_blocks, uint& num_threads)
{
	num_threads = min(block_size, n);
	num_blocks = (n % num_threads != 0) ? (n / num_threads + 1) : (n / num_threads);
}

void calculate_hash(
	CellData cell_data,
	float3* pos,
	uint    num_particles)
{
	uint num_blocks, num_threads;
	compute_grid_size(num_particles, MAX_THREAD_NUM, num_blocks, num_threads);
	calcHashD << < num_blocks, num_threads >> > (
		cell_data,
		pos,
		num_particles);
	getLastCudaError("Kernel execution failed: calc_hash");
}

void reorder_data(
	CellData cell_data,
	float3* oldPos,
	uint	numParticles,
	uint	numCells)
{
	uint num_threads, num_blocks;
	compute_grid_size(numParticles, MAX_THREAD_NUM, num_blocks, num_threads);

	// set all cells to empty
	checkCudaErrors(cudaMemset(cell_data.cell_start, 0xffffffff, numCells * sizeof(uint)));

	uint smemSize = sizeof(uint) * (num_threads + 1);
	reorderDataAndFindCellStartD << < num_blocks, num_threads, smemSize >> > (
		cell_data,
		oldPos,
		numParticles);
	getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

}

void sort_and_reorder(
	float3* predict_pos,
	CellData cell_data,
	uint num_particles,
	uint num_cells=NUM_CELLS
) 
{
	calculate_hash(
		cell_data,
		predict_pos,
		num_particles
	);
	sort_particles(
		cell_data,
		num_particles
	);
	reorder_data(
		cell_data,
		predict_pos,
		num_particles,
		num_particles
	);
}

void compute_boundary_volume(CellData data, float* mass, float* volume, uint numParticles)
{
	uint num_threads, num_blocks;
	compute_grid_size(numParticles, MAX_THREAD_NUM, num_blocks, num_threads);

	compute_boundary_volume_d << <num_blocks, num_threads >> > (
		data,
		mass, volume,
		numParticles);

	getLastCudaError("Kernel execution failed: copmute_boundary_volume");
}

__global__ void test_offset(float3* positions)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	/*
	if (i == 0)
		printf("particles[0]: %f, %f, %f\n", positions[i].x , positions[i].y, positions[i].z);
	*/
	positions[i].x = positions[i].x + 0.001f;
	positions[i].y = positions[i].y + 0.001f;
	positions[i].z = positions[i].z + 0.001f;
}

__global__ 
void integrate_pbd_d(
	float3* pos, float3* vel, float3* force, float* massInv,
	float3* predict_pos, float3* new_pos,
	float dt,
	uint numParticles)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	
	float3 t_vel = vel[index] + dt * params.gravity;
	t_vel = t_vel * params.global_damping;
	float3 t_pos = pos[index] + dt * t_vel;

	predict_pos[index] = pos[index] + dt * t_vel;
	vel[index] = t_vel; 
	new_pos[index] = predict_pos[index];


}

__global__
void integrate_pbd_cd_d(
	ParticleDeviceData data,
	float dt,
	uint numParticles)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	float3* pos = data.m_d_positions;
	float3* vel = data.m_d_velocity;
	float3* force = data.m_d_force;
	float3* predict_pos = data.m_d_predict_positions;
	float3* new_pos = data.m_d_new_positions;
	float* massInv = data.m_d_massInv;
	float* contrib = data.m_d_contrib;

	float3 t_vel = vel[index] + dt * params.gravity;
	t_vel = t_vel * params.global_damping;
	float3 t_pos = pos[index] + dt * t_vel;

	if (t_pos.x + params.particle_radius >= 1.0f)
	{
		t_pos.x = 1.f;
		t_vel.x =  -abs(t_vel.x);
		t_vel *= params.boundary_damping;
	}

	if (t_pos.x - params.particle_radius <= -1.0f)
	{
		t_pos.x = -1.f;
		t_vel.x = abs(t_vel.x);
		t_vel *= params.boundary_damping;
	}

	if (t_pos.z + params.particle_radius >= 1.0f)
	{
		t_pos.z = 1.f;
		t_vel.z = -abs(t_vel.z);
		t_vel *= params.boundary_damping;
	}

	if (t_pos.z - params.particle_radius <= -1.0f)
	{
		t_pos.z = -1.f;
		t_vel.z = abs(t_vel.z);
		t_vel *= params.boundary_damping;
	}
	
	if (pos[index].y - params.particle_radius <= 0.f )
	{
		pos[index].y = params.particle_radius;
		t_vel.y = abs(t_vel.y);
		t_vel *= params.boundary_damping;
	}
	

	// Velocity limitation
	const float max_limit = params.maximum_speed;
	
	if (length(t_vel) > max_limit)
	{
		//printf("%u: pos: %f, %f, %f vel: %f T: %f contrib: %f density: %f C:%f lambda: %f\n", index, pos[index].x, pos[index].y, pos[index].z, length(t_vel),data.m_d_T[index] , data.m_d_contrib[index], data.m_d_density[index], data.m_d_C[index], data.m_d_lambda[index]);
		t_vel = (max_limit / length(t_vel)) * t_vel;
		
	}

	predict_pos[index] = pos[index] + dt * t_vel;
	vel[index] = t_vel;
	new_pos[index] = predict_pos[index];

	if (contrib[index] < 0.99f)	contrib[index] += params.blending_speed;
	
	if (contrib[index] >= 0.99f) contrib[index] = 1.f;
}

// compute density without volume
inline __device__
float pbf_density_0(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float*	mass,
	CellData cell_data
)  
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_data.cell_start[grid_hash];
	float density = 0.0f;

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			if (j != index)                // check not colliding with self
			{
				uint original_index = cell_data.grid_index[j];
				
				float3 pos2 = cell_data.sorted_pos[j];
				float3 vec = pos - pos2;
				float dist = length(vec);
				float rho = 0.f;

				rho = mass[original_index] * sph_kernel_poly6(dist, params.effective_radius);

				density += rho;
			}
		}
	}

	return density;
}

// compute density 'with' volume 
inline __device__
float pbf_density_1(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float*	b_mass,
	float*	b_volume,	
	CellData cell_data	
) 
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_data.cell_start[grid_hash];
	float density = 0.0f;

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			if (j != index)                // check not colliding with self
			{
				uint original_index = cell_data.grid_index[j];

				float3 pos2 = cell_data.sorted_pos[j];
				float3 vec = pos - pos2;
				float dist = length(vec);
				float rho = 0.f;

				rho = params.rest_density * b_volume[original_index] * sph_kernel_poly6(dist, params.effective_radius);

				density += rho;
			}
		}
	}

	return density;
}


inline __device__
float pbf_density_coupling_vol(
	int3    grid_pos,
	float3  pos1,
	float*	volume,
	CellData cell_data
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_data.cell_start[grid_hash];
	float density = 0.0f;

	// if cell of boundary cell data is not empty
	if (start_index != 0xffffffff)
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{	
			// no need to check collision (boundary cell data is not the same as fluid cell data)
			uint original_index = cell_data.grid_index[j];

			float3 pos2 = cell_data.sorted_pos[j];
			float3 vec = pos1 - pos2;
			float dist = length(vec);

			float rho = params.rest_density * volume[original_index] * sph_kernel_poly6(dist, params.effective_radius);

			density += rho;	
		}
	}

	// return contributions of boundary paritcles
	return density;
}

// boundary - fluid
inline __device__
float pbf_boundary_density_coupling(
	// boundary
	int3		grid_pos,	// searching grid pos
	float3		pos1,		// position of boundary particle
	// fluid
	float*		mass,
	CellData	cell_data
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_data.cell_start[grid_hash];
	float density = 0.0f;

	// if cell of boundary cell data is not empty
	if (start_index != 0xffffffff)
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			// no need to check collision (boundary cell data is not the same as fluid cell data)
			uint original_index = cell_data.grid_index[j];

			float3 pos2 = cell_data.sorted_pos[j];
			float3 vec = pos1 - pos2;
			float dist = length(vec);

			float rho = mass[original_index] * sph_kernel_poly6(dist, params.effective_radius);

			density += rho;
		}
	}

	// return contributions of boundary paritcles
	return density;
}


// compute lambda 'without' volume
inline __device__
float pbf_lambda_0(
	int3    grid_pos,
	uint    index,
	float3  pos,
	CellData cell_data
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_data.cell_start[grid_hash];
	float gradientC_sum = 0.f;

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			if (j != index)                // check not colliding with self
			{
				//uint original_index = cell_data.grid_index[j];
				//float particle_mass = mass[original_index];
				float3 pos2 = cell_data.sorted_pos[j];
				float3 vec = pos - pos2;
				float dist = length(vec);

				float3 gradientC_j;

				gradientC_j = (1.f / params.rest_density) *
					sph_kernel_spiky_gradient(vec, dist, params.effective_radius);

				float dot_val = dot(gradientC_j, gradientC_j);
				gradientC_sum += dot_val;
			}
		}
	}
	return gradientC_sum;
}

// compute lambda with volume
inline __device__
float pbf_lambda_1(
	int3    grid_pos,
	uint    index,
	float3  pos,
	
	float* mass,
	float3* sorted_pos,
	uint* cell_start,
	uint* cell_end,
	uint* gridParticleIndex,
	float* b_volume = nullptr)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_start[grid_hash];
	float gradientC_sum = 0.f;

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			if (j != index)                // check not colliding with self
			{
				uint original_index = gridParticleIndex[j];
				float particle_mass = mass[original_index];
				float3 pos2 = sorted_pos[j];
				float3 vec = pos - pos2;
				float dist = length(vec);

				float3 gradientC_j;
				float vol = b_volume[original_index];

				gradientC_j = (1.f / params.rest_density) *
						(params.rest_density * vol / particle_mass) *
						sph_kernel_spiky_gradient(vec, dist, params.effective_radius);

				float dot_val = dot(gradientC_j, gradientC_j);
				gradientC_sum += dot_val;
			}
		}
	}
	return gradientC_sum;
}

// fluid - boundary
inline __device__
float pbf_lambda_boundary(
	int3		grid_pos,	// searching grid pos
	float3		pos1,		// position of fluid particle

	float		particle_mass,
	CellData	cell_data,	// cell data of boundary particle,
	float*		volume
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_data.cell_start[grid_hash];
	float gradientC_sum = 0.f;

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			uint original_index = cell_data.grid_index[j];
			float vol = volume[original_index];

			float3 pos2 = cell_data.sorted_pos[j];
			float3 vec = pos1 - pos2;
			float dist = length(vec);

			float3 gradientC_j = (1.f / params.rest_density) * 
				(params.rest_density * vol / particle_mass) *  
				sph_kernel_spiky_gradient(vec, dist, params.effective_radius);

			float dot_val = dot(gradientC_j, gradientC_j);
			gradientC_sum += dot_val;
		}
	}
	return gradientC_sum;
}

// Boundary - fluid 
inline __device__
float pbf_boundary_lambda(
	// boundary
	int3		grid_pos,	// searching grid pos
	float3		pos1,		// position of boundary particle
	// fluid
	CellData	cell_data
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_data.cell_start[grid_hash];
	float gradientC_sum = 0.f;

	// search in fluid cell
	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			float3 pos2 = cell_data.sorted_pos[j];
			float3 vec = pos1 - pos2;
			float dist = length(vec);

			float3 gradientC_j = (1.f / params.rest_density) *
				sph_kernel_spiky_gradient(vec, dist, params.effective_radius);

			float dot_val = dot(gradientC_j, gradientC_j);
			gradientC_sum += dot_val;
		}
	}
	return gradientC_sum;
}

inline __device__
float3 pbf_correction(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float	lambda_i,
	float*	lambdas,
	float*  contrib,
	CellData cell_data,
	float	dt)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_data.cell_start[grid_hash];
	float3 correction = make_float3(0, 0, 0);

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			if (j != index)                // check not colliding with self
			{
				uint original_index = cell_data.grid_index[j];

				const float3 pos2 = cell_data.sorted_pos[j];
				const float3 vec = pos - pos2;
				const float dist = length(vec);

				const float3 gradient = sph_kernel_spiky_gradient(vec, dist, params.effective_radius);
				
				float x = sph_kernel_poly6(dist, params.effective_radius) / params.scorr_divisor;
				x = pow(x, 4);
				const float scorr = -params.scorr_coeff * x * dt * dt;

				correction += contrib[original_index] * (lambda_i + lambdas[original_index] + scorr) * gradient;

			}
		}

		//printf("Num neighbors: %u\n", end_index - start_index);
	}
	return correction;
}

// compute correction from boundary particles
inline __device__
float3 pbf_correction_boundary(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float	lambda_i,
	// boundary
	CellData b_cell_data,
	float*	b_lambdas,
	float	dt)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = b_cell_data.cell_start[grid_hash];
	float3 correction = make_float3(0, 0, 0);

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = b_cell_data.cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			uint original_index = b_cell_data.grid_index[j];

			float lambda_j = b_lambdas[original_index];
			float3 pos2 = b_cell_data.sorted_pos[j];
			float3 vec = pos - pos2;
			float dist = length(vec);

			float3 gradient = sph_kernel_spiky_gradient(vec, dist, params.effective_radius);

			float scorr = -0.1f;
			float x = sph_kernel_poly6(dist, params.effective_radius) /
				sph_kernel_poly6(0.3f * params.effective_radius, params.effective_radius);
			x = pow(x, 4);
			scorr = scorr * x * dt * dt;

			float3 res = (lambda_i + lambda_j + scorr) * gradient;

			correction += res;
		}

		//printf("Num neighbors: %u\n", end_index - start_index);
	}
	return correction;
}

// compute correction from other particle set (triggered when close enough)
inline __device__
float3 pbf_correction_coupling(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float	lambda_i,
	// boundary
	CellData other_cell_data,
	float*   other_lambdas,
	float	dt)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = other_cell_data.cell_start[grid_hash];
	float3 correction = make_float3(0, 0, 0);

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = other_cell_data.cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			if (j != index)                // check not colliding with self
			{
				uint original_index = other_cell_data.grid_index[j];

				float lambda_j = other_lambdas[original_index];
				float3 pos2 = other_cell_data.sorted_pos[j];
				float3 vec = pos - pos2;
				float dist = length(vec);

				if (dist <= 2.f * params.particle_radius)
				{
					float3 gradient = sph_kernel_poly6_gradient(vec, dist, params.effective_radius);

					float scorr = -0.1f;
					float x = sph_kernel_poly6(dist, params.effective_radius) /
						sph_kernel_poly6(0.3f * params.effective_radius, params.effective_radius);
					x = pow(x, 4);
					scorr = scorr * x * dt * dt;

					float3 res = (lambda_i + lambda_j + scorr) * gradient;

					correction += res;
				}				
			}
		}

		//printf("Num neighbors: %u\n", end_index - start_index);
	}
	return correction;
}

__global__
void compute_density_d(
	float*	density,					// output: computed density
	float*	mass,						// input: mass
	float*	C,							// input: contraint
	float*	b_volume,
	CellData cell_data,
	CellData b_cell_data,
	uint	numParticles
	)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	
	if (index >= numParticles) return;

	uint originalIndex = cell_data.grid_index[index];

	// read particle data from sorted arrays
	float3 pos = cell_data.sorted_pos[index];
	
	// initial density
	float rho = mass[originalIndex] * sph_kernel_poly6(0, params.effective_radius);

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	// traverse 27 neighbors (fluid - fluid)
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				rho += pbf_density_0(
					neighbor_pos, index, 
					pos, mass, 
					cell_data
					//cell_start, cell_end, gridParticleIndex
				);
			}
		}
	}

	// use gridPos to traverse 27 surrounding grids (fluid - boundary)
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_gridPos = gridPos + make_int3(x, y, z);
				rho += pbf_density_coupling_vol(
					// fluid
					neighbor_gridPos,
					pos, 
					// boundary
					b_volume,
					b_cell_data
				);
			}
		}
	}


	// Update date density and constraint value
	density[originalIndex] = rho;
	C[originalIndex] = (rho / params.rest_density) - 1.f;

	//printf("rho = %f\n", rho);
	//printf("C[%u]: %f\n", originalIndex, C[originalIndex]);

}

__global__
void compute_boundary_density_d(
	// fluid
	float*		mass,						// input: mass of fluid paritcle
	// boundary
	float*		b_mass,
	float*		b_volume,
	float*		b_C,
	float*		b_density,					// output: boundary density
	// cell data
	CellData	cell_data,
	CellData	b_cell_data,
	uint		b_numParticles
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= b_numParticles) return;

	// original index of boundary particle
	uint originalIndex = b_cell_data.grid_index[index];

	// read position from sorted arrays
	float3 pos = b_cell_data.sorted_pos[index];

	// initial density 
	float rho = params.rest_density * b_volume[originalIndex] * sph_kernel_poly6(0, params.effective_radius);

	// get address in grid of boundary particles (basically the same as fluid particle)
	int3 gridPos = calcGridPos(pos);

	// use gridPos to traverse 27 surrounding grids (boundary - boundary)
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_gridPos = gridPos + make_int3(x, y, z);
				rho += pbf_density_1(
					neighbor_gridPos, index,
					pos, 
					b_mass,
					b_volume,					
					b_cell_data
				);
			}
		}
	}

	// use gridPos to traverse 27 surrounding grids (boundary - fluid)
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_gridPos = gridPos + make_int3(x, y, z);
				rho += pbf_boundary_density_coupling(
					// boundary
					neighbor_gridPos,
					pos,
					// fluid
					mass,
					cell_data
				);
			}
		}
	}

	// Update density of fluid particle
	b_density[originalIndex] = rho;
	// **repeated code**
	// Recompute constraint value of fluid particle
	b_C[originalIndex] = (b_density[originalIndex] / params.rest_density) - 1.f;
}

/* fluid - boundary */
__global__
void compute_lambdas_d(
	float*	lambda,						// output: computed density
	float*	C,							// input: contraint
	float*  mass,
	float*	b_volume,
	CellData cell_data,
	CellData b_cell_data,
	uint	numParticles
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	uint originalIndex = cell_data.grid_index[index];

	// read particle data from sorted arrays
	float3 pos = cell_data.sorted_pos[index];

	// initial density
	lambda[originalIndex] = -C[originalIndex];

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	float3 gradientC_i = make_float3(0);
		//-(1.f / params.rest_density) *
		//Poly6_W_Gradient_CUDA(make_float3(0, 0, 0), 0, params.effective_radius);
	float gradientC_sum = dot(gradientC_i, gradientC_i);
	// traverse 27 neighbors
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				float res = pbf_lambda_0(
					neighbor_pos, index,
					pos, 
					cell_data
				);
				gradientC_sum += res;
			}
		}
	}

	// traverse 27 neighbors in "boundary cells"
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				float res = pbf_lambda_boundary(
					neighbor_pos, 
					pos, 
					mass[originalIndex],  // paritcle_mass
					b_cell_data,
					b_volume
				);
				gradientC_sum += res;
			}
		}
	}

	//printf("gradientC_sum: %f\n", gradientC_sum);
	lambda[originalIndex] /= gradientC_sum + params.epsilon;

	//lambda[originalIndex] = lambda_res;
}

__global__
void compute_boundary_lambdas_d(
	float* b_lambda,				// lambda of boundary particles
	float* b_vol,
	float3* b_pos,
	float* b_C,
	float* b_mass,
	CellData b_cell_data,
	// Cell data of fluid particles
	CellData cell_data,
	uint	b_numParticles		// number of boundary particles
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= b_numParticles) return;

	uint originalIndex = b_cell_data.grid_index[index];

	// read particle data from sorted arrays
	float3 pos = b_cell_data.sorted_pos[index];

	// initial density
	b_lambda[originalIndex] = -b_C[originalIndex];

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	float3 gradientC_i = make_float3(0);
	//-(1.f / params.rest_density) *
	//Poly6_W_Gradient_CUDA(make_float3(0, 0, 0), 0, params.effective_radius);
	float gradientC_sum = dot(gradientC_i, gradientC_i);
	
	// traverse 27 neighbors in boundary cells (boundary - boundary)
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				float res = pbf_lambda_1(
					neighbor_pos, index,
					pos, 
					b_mass,
					b_cell_data.sorted_pos,
					b_cell_data.cell_start, b_cell_data.cell_end, b_cell_data.grid_index,
					b_vol
				);
				gradientC_sum += res;
			}
		}
	}

	// traverse 27 neighbors in "fluid cells" (boundary - fluid)
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				float res = pbf_boundary_lambda(
					// boundary
					neighbor_pos,
					pos, 
					// fluid
					cell_data
				);
				gradientC_sum += res;
			}
		}
	}

	//printf("gradientC_sum: %f\n", gradientC_sum);
	b_lambda[originalIndex] /= gradientC_sum + params.epsilon;

	//lambda[originalIndex] = lambda_res;
}

__global__
void apply_correction(
	ParticleDeviceData data,
	/*
	float3* new_pos,
	float3* predict_pos,
	float3* correction,
	*/
	CellData cell_data,
	uint numParticles
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;
	//predict_pos[index] = new_pos[index];
	
	uint original_index = cell_data.grid_index[index];
		
	data.m_d_new_positions[original_index] = cell_data.sorted_pos[index] + params.sor_coeff * data.m_d_correction[original_index];
	/*
	if (isnan(data.m_d_new_positions[original_index].x) || isnan(data.m_d_new_positions[original_index].y) || isnan(data.m_d_new_positions[original_index].z))
		printf("(%u) new_pos NAN at %u: sorted_pos: %f %f %f  predict_pos: %f %f %f\n", 
			data.m_d_trackId[original_index], original_index, cell_data.sorted_pos[index].x, cell_data.sorted_pos[index].y, cell_data.sorted_pos[index].z,
		    data.m_d_new_positions[original_index].x, data.m_d_new_positions[original_index].y, data.m_d_new_positions[original_index].z);
	*/
	data.m_d_predict_positions[original_index] = data.m_d_new_positions[original_index];
	// write back to sorted_pos for next iteration
	//cell_data.sorted_pos[index] = new_pos[original_index]; // disabled (since every step the sorted pos will be filled with different value)
	data.m_d_correction[original_index] = make_float3(0, 0, 0);
}


__global__
void finalize_correction(
	float3* pos,
	float3* new_pos,
	float3* predict_pos,
	float3* velocity,
	uint numParticles,
	float dt
) 
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	//float3 res = new_pos[index];
	//float3 vel = (res - pos[index]) / dt;
	float3 diff = new_pos[index] - pos[index];
	float3 t_pos = new_pos[index];
	float3 t_vel = (length(diff) >= params.minimum_speed) ? (diff) / dt : make_float3(0);
	
	velocity[index] = t_vel;
	//predict_pos[index] = t_pos;
	pos[index] = t_pos;


}

void allocate_array(void** devPtr, size_t size)
{
	checkCudaErrors(cudaMalloc(devPtr, size));
}

void set_sim_params(SimParams* param_in)
{
	checkCudaErrors(cudaMemcpyToSymbol(params, param_in, sizeof(SimParams)));
}

/* Integration for Position based Dynamics */
void integrate_pbd(
	ParticleSet* particles,
	float deltaTime,
	uint numParticles,
	bool cd_on = false
)
{
	uint num_threads, num_blocks;
	compute_grid_size(numParticles, MAX_THREAD_NUM, num_blocks, num_threads);
	if (cd_on)
	{
		integrate_pbd_cd_d << <num_blocks, num_threads >> > (
			particles->m_device_data,
			deltaTime,
			numParticles
			);
	}
	else
	{
		integrate_pbd_d << <num_blocks, num_threads >> > (
			particles->m_device_data.m_d_positions,
			particles->m_device_data.m_d_velocity,
			particles->m_device_data.m_d_force,
			particles->m_device_data.m_d_massInv,
			particles->m_device_data.m_d_predict_positions,
			particles->m_device_data.m_d_new_positions,
			deltaTime,
			numParticles
			);
	}
	getLastCudaError("Kernel execution failed: integrate_pbd_d ");
}

void sort_particles(CellData cell_data, uint numParticles)
{
	uint* grid_hash = cell_data.grid_hash;
	uint* grid_index = cell_data.grid_index;

	thrust::device_ptr<uint> d_hash_ptr = thrust::device_pointer_cast(cell_data.grid_hash);
	thrust::device_ptr<uint> d_index_ptr = thrust::device_pointer_cast(cell_data.grid_index);

	thrust::sort_by_key(
		d_hash_ptr,
		d_hash_ptr + numParticles, 
		d_index_ptr		
	);
}

inline __device__
float3 pbd_distance_correction_contrib(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float	w0,
	float* invMass,
	float* contrib,
	CellData cell_data
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_data.cell_start[grid_hash];
	float3 correction = make_float3(0, 0, 0);

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cell_end[grid_hash];

		// reuse C in searching
		float C = 0;

		for (uint j = start_index; j < end_index; j++)
		{
			float3 correction_j = make_float3(0, 0, 0);
			if (j != index)                // check not colliding with self
			{
				uint original_index_j = cell_data.grid_index[j];

				float3 pos2 = cell_data.sorted_pos[j];
				float3 v = pos - pos2;
				float dist = length(v);

				// correct if distance is close
				if (dist < 2.f * params.particle_radius)
				{
					// Non-penetration correction
					const float w1 = invMass[original_index_j];

					float w_sum = w0 + w1;
					C = dist - 2.f * params.particle_radius;

					float3 n = v / (dist + params.pbd_epsilon);// +0.000001f);

					correction_j = -w0 * (1.f / w_sum) * C * n;
					//correction_j *= contrib[original_index_j];
				}
			}
			correction += correction_j;
		}

		//printf("Num neighbors: %u\n", end_index - start_index);
	}
	return correction;
}

inline __device__
float3 pbd_distance_correction(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float	w0,
	float*	invMass,
	CellData cell_data
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_data.cell_start[grid_hash];
	float3 correction = make_float3(0, 0, 0);

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cell_end[grid_hash];

		// reuse C in searching
		float C = 0;

		for (uint j = start_index; j < end_index; j++)
		{
			float3 correction_j = make_float3(0, 0, 0);
			if (j != index)                // check not colliding with self
			{
				uint original_index_j = cell_data.grid_index[j];

				float3 pos2 = cell_data.sorted_pos[j];
				float3 v = pos - pos2;
				float dist = length(v);

				// correct if distance is close
				if (dist < 2.f * params.particle_radius)
				{
					// Non-penetration correction
					const float w1 = invMass[original_index_j];

					float w_sum = w0 + w1;
					C = dist - 2.f * params.particle_radius;
					
					float3 n = v / (dist + params.pbd_epsilon);// +0.000001f);

					correction_j = -w0 * (1.f / w_sum) * C * n;
				 }
			}
			correction += correction_j;
		}

		//printf("Num neighbors: %u\n", end_index - start_index);
	}
	return correction;
}

inline __device__
float3 pbd_sph_sph_distance_correction(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float	w0,
	float  contrib,
	float* invMass,	
	CellData cell_data
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_data.cell_start[grid_hash];
	float3 correction = make_float3(0, 0, 0);

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cell_end[grid_hash];

		// reuse C in searching
		float C = 0;

		for (uint j = start_index; j < end_index; j++)
		{
			float3 correction_j = make_float3(0, 0, 0);
			if (j != index)                // check not colliding with self
			{
				uint original_index_j = cell_data.grid_index[j];

				if (contrib <= 0.99f)
				{

					float3 pos2 = cell_data.sorted_pos[j];
					float3 v = pos - pos2;
					float dist = length(v);

					// correct if distance is close
					if (dist < 2.f * params.particle_radius)
					{
						// Non-penetration correction
						const float w1 = invMass[original_index_j];

						float w_sum = w0 + w1;
						C = dist - 2.f * params.particle_radius;

						float3 n = v / (dist + params.pbd_epsilon);// +0.000001f);

						correction_j = -w0 * (1.f / w_sum) * C * n;
					}
				}
			}
			correction += correction_j;
		}

		//printf("Num neighbors: %u\n", end_index - start_index);
	}
	return correction;
}

inline __device__
float3 pbd_distance_correction_coupling_contrib(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float	w0,
	float*  other_invMass,    // invMass of boundary particles
	float*  other_contrib,
	CellData other_cell_data	// cell_data of boundary particles
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = other_cell_data.cell_start[grid_hash];
	float3 correction = make_float3(0, 0, 0);

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = other_cell_data.cell_end[grid_hash];

		// reuse C in searching
		float C = 0;

		for (uint j = start_index; j < end_index; j++)
		{
			float3 correction_j = make_float3(0, 0, 0);

			uint original_index_j = other_cell_data.grid_index[j];

			float3 pos2 = other_cell_data.sorted_pos[j];
			float3 v = pos - pos2;
			float dist = length(v);

			// correct if distance is close
			if (dist < 2.f * params.particle_radius)
			{
				// Non-penetration correction
				const float w1 = other_invMass[original_index_j];

				float w_sum = w0 + w1;
				C = dist - 2.f * params.particle_radius;

				// normalize v + 0.000001f to prevent not becoming infinite
				float3 n = v / (dist + params.pbd_epsilon);// +0.000001f);

				correction_j = -w0 * (1.f / w_sum) * C * n;
				//correction_j *= other_contrib[original_index_j];
			}

			correction += correction_j;
		}

		//printf("Num neighbors: %u\n", end_index - start_index);
	}
	return correction;
}


inline __device__
float3 pbd_distance_correction_boundary(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float	w0,
	float*	b_invMass,    // invMass of boundary particles
	CellData b_cell_data	// cell_data of boundary particles
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = b_cell_data.cell_start[grid_hash];
	float3 correction = make_float3(0, 0, 0);

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = b_cell_data.cell_end[grid_hash];

		// reuse C in searching
		float C = 0;

		for (uint j = start_index; j < end_index; j++)
		{
			float3 correction_j = make_float3(0, 0, 0);
			
			uint original_index_j = b_cell_data.grid_index[j];

			float3 pos2 = b_cell_data.sorted_pos[j];
			float3 v = pos - pos2;
			float dist = length(v);

			// correct if distance is close
			if (dist < 2.f * params.particle_radius)
			{
				// Non-penetration correction
				const float w1 = b_invMass[original_index_j];

				float w_sum = w0 + w1;
				C = dist - 2.f * params.particle_radius;

				// normalize v + 0.000001f to prevent not becoming infinite
				float3 n = v / (dist + params.pbd_epsilon);// +0.000001f);

				correction_j = -w0 * (1.f / w_sum) * C * n;
			}

			correction += correction_j;
		}

		//printf("Num neighbors: %u\n", end_index - start_index);
	}
	return correction;
}

__global__
void compute_dem_distance_correction(
	ParticleDeviceData dem_data,
	ParticleDeviceData b_data,
	CellData	cell_data,		// input: cell data of dem particles
	CellData	b_cell_data,
	uint		numParticles	// input: number of DEM particles
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	
	if (index >= numParticles) return;

	uint original_index = cell_data.grid_index[index];

	// read particle data from sorted arrays
	float3 pos = cell_data.sorted_pos[index];
	float w0 = dem_data.m_d_massInv[original_index];
	// get address in grid
	int3 gridPos = calcGridPos(pos);

	float3 corr = make_float3(0, 0, 0);


	//dem-dem
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				corr += pbd_distance_correction_contrib(
					neighbor_pos, index,
					pos, w0,
					dem_data.m_d_massInv,
					dem_data.m_d_contrib,
					cell_data
				);
			}
		}
	}

	// dem-boundary
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				corr += pbd_distance_correction_coupling_contrib(
					neighbor_pos, index,
					pos, w0,
					b_data.m_d_massInv,
					b_data.m_d_contrib,
					b_cell_data
				);
			}
		}
	}

	corr *= dem_data.m_d_contrib[original_index];
	dem_data.m_d_correction[original_index] += corr;
}

inline __device__
float3 pbd_friction_correction(
	int3    grid_pos,
	uint    index,
	float3  predict_pos0,
	float3	original_pos0,
	float	w0,
	float3* predict_pos,	
	float3* original_pos,
	float*	invMass,	
	CellData cell_data
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_data.cell_start[grid_hash];
	float3 result = make_float3(0,0,0);// correction_i;

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			float3 correction_j = make_float3(0, 0, 0);
			if (j != index)                // check not colliding with self
			{
				uint original_index_j = cell_data.grid_index[j];

				float3 original_pos1 = original_pos[original_index_j];
				//float3 predict_pos1 = predict_pos[original_index_j];

				float3 predict_pos1 = cell_data.sorted_pos[j];
				float3 v = predict_pos0 - predict_pos1;
				float dist = length(v);

				// correct if distance is close
				if (dist <= 2.f * params.particle_radius)
				{
					// Non-penetration correction
					const float w1 = invMass[original_index_j];

					float w_sum = w0 + w1;

					// normalize v + 0.000001f for vanish problem
					float3 n = v / (dist + params.pbd_epsilon);

					float penetration = 2.f * params.particle_radius - dist;
					float3 dx = (predict_pos0 - original_pos0) + (predict_pos1 - original_pos1);
					float3 dx_t = dx - (dot(dx, n) * n);

					//printf("dx: %f, %f, %f\n", dx_t.x, dx_t.y, dx_t.z);
					//printf("penetration: %f\n", penetration);


					float threshold = params.static_friction * penetration;
					float len = length(dx_t);

					// use kinematic friction model
					if (length(dx_t) > threshold)
					{
						float coeff = min(params.kinematic_friction * penetration / (len+0.000001f), 1.f);
						dx_t = coeff * dx_t;
					}/*
					else
					{
						printf("static\n");
					}
					*/

					dx_t = -(w0 / w_sum) * dx_t;
					correction_j += dx_t;
					//printf("dx: %f, %f, %f\n", dx_t.x, dx_t.y, dx_t.z);
				}
			}
			result += correction_j;
		}

	}
	return result;
}

inline __device__
float3 pbd_friction_correction_boundary(
	int3    grid_pos,
	uint    index,
	float3  predict_pos0,
	float3	original_pos0,
	float	w0,
	float*  b_invMass,
	CellData b_cell_data
	)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = b_cell_data.cell_start[grid_hash];
	float3 result = make_float3(0, 0, 0);// correction_i;

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = b_cell_data.cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			float3 correction_j = make_float3(0, 0, 0);
			
			uint original_index_j = b_cell_data.grid_index[j];

			float3 pos1 = b_cell_data.sorted_pos[j];
			float3 v = predict_pos0 - pos1;
			float dist = length(v);

			// correct if distance is close
			if (dist <= 2.f * params.particle_radius)
			{
				// Non-penetration correction
				const float w1 = b_invMass[original_index_j];

				float w_sum = w0 + w1;

				// normalize v + 0.000001f for vanish problem
				float3 n = v / (dist + params.pbd_epsilon);// +0.000001f);

				float penetration = 2.f * params.particle_radius - dist;
				float3 dx = (predict_pos0 - original_pos0);
				float3 dx_t = dx - (dot(dx, n) * n);

				//printf("dx: %f, %f, %f\n", dx.x, dx.y, dx.z);


				float threshold = params.static_friction * penetration;
				float len = length(dx_t);

				// if exceed threshold use kinematic friction model
				if (length(dx_t) > threshold)
				{
					float coeff = min(params.kinematic_friction * penetration / (len+0.000001f), 1.f);
					dx_t = coeff * dx_t;
				}

				dx_t = -(w0 / w_sum) * dx_t;
				correction_j += dx_t;
			}
			result += correction_j;
		}
	}
	return result;
}

__global__
void compute_friction_correction(
	float3* correction,
	float3* new_pos,	// output: corrected pos
	float3* original_pos, // input: position at the start of this time step
	float* invMass,		// input: mass
	float* b_invMass,
	CellData	cell_data,		// input: cell data of dem particles
	CellData	b_cell_data,
	uint		numParticles	// input: number of DEM particles
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	uint original_index = cell_data.grid_index[index];

	// read particle data from sorted arrays
	float3 pos = cell_data.sorted_pos[index];
	float3 original_pos0 = original_pos[original_index];

	float w0 = invMass[original_index];
	
	//float3 correction_i = correction[original_index];
	// get address in grid
	int3 gridPos = calcGridPos(pos);

	float3 corr = make_float3(0, 0, 0);

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				corr += pbd_friction_correction(
					neighbor_pos, index,
					pos, original_pos0,w0, 
					new_pos, original_pos, invMass,
					cell_data
				);
			}
		}
	}
	
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				corr += pbd_friction_correction_boundary(
					neighbor_pos, index,
					pos, original_pos0, w0,
					b_invMass,
					b_cell_data
				);
			}
		}
	}
	correction[original_index] = corr;
}

__global__
void compute_sph_dem_distance_correction(
	ParticleDeviceData sph_data,
	ParticleDeviceData dem_data,
	CellData	sph_cell_data,		// input: cell data of dem particles
	CellData	dem_cell_data,
	uint		numParticles	// input: number of sph particles
	)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	uint original_index = sph_cell_data.grid_index[index];

	// read particle data from sorted arrays
	float3 pos = sph_cell_data.sorted_pos[index];
	float w0 = sph_data.m_d_massInv[original_index];
	// get address in grid
	int3 gridPos = calcGridPos(pos);

	float3 corr = make_float3(0, 0, 0);

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				corr += pbd_distance_correction_coupling_contrib(
					neighbor_pos, index,
					pos, w0,
					dem_data.m_d_massInv,
					dem_data.m_d_contrib,
					dem_cell_data
					);
			}
		}
	}

	corr *= (1.f - sph_data.m_d_contrib[original_index]);

	sph_data.m_d_correction[original_index] += corr;
}

__global__
void compute_dem_sph_distance_correction(
	ParticleDeviceData dem_data,
	ParticleDeviceData sph_data,
	CellData	dem_cell_data,		// input: cell data of dem particles
	CellData	sph_cell_data,
	uint		numParticles	// input: number of sph particles
	)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	uint original_index = dem_cell_data.grid_index[index];

	// read particle data from sorted arrays
	float3 pos = dem_cell_data.sorted_pos[index];
	float w0 = dem_data.m_d_massInv[original_index];
	// get address in grid
	int3 gridPos = calcGridPos(pos);

	float3 corr = make_float3(0, 0, 0);

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				corr += pbd_distance_correction_coupling_contrib(
					neighbor_pos, index,
					pos, w0,
					sph_data.m_d_massInv,
					sph_data.m_d_contrib,
					sph_cell_data
					);
			}
		}
	}

	//corr_{new_dem} = (1-a) corr_{sph} + "a corr_{dem}"
	corr *= dem_data.m_d_contrib[original_index];

	dem_data.m_d_correction[original_index] += corr;
}

__global__
void compute_sph_sph_distance_correction(
	ParticleDeviceData sph_data,
	/*
	float3* correction,		// output: corrected pos
	float* invMass,		// input: mass
	*/
	CellData	sph_cell_data,		// input: cell data of dem particles
	uint		numParticles	// input: number of sph particles
	)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	float3* correction = sph_data.m_d_correction;
	float* invMass = sph_data.m_d_massInv;
	float* contrib = sph_data.m_d_contrib;



	uint original_index = sph_cell_data.grid_index[index];

	// read particle data from sorted arrays
	float3 pos = sph_cell_data.sorted_pos[index];
	float w0 = invMass[original_index];
	// get address in grid
	int3 gridPos = calcGridPos(pos);

	float3 corr = make_float3(0, 0, 0);

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				corr += pbd_sph_sph_distance_correction(
					neighbor_pos, index,
					pos, w0, contrib[original_index],
					invMass,
					sph_cell_data
					);
			}
		}
	}
	corr *= (1.f - contrib[original_index]);
	correction[original_index] += corr;
}

__global__
void compute_snow_sph_density_d(
	float* density,					// output: computed density
	float* mass,						// input: mass
	float* C,							// input: contraint
	float* dem_mass,
	float* b_volume,
	CellData sph_cell_data,
	CellData dem_cell_data,
	CellData b_cell_data,
	uint	numParticles
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	uint originalIndex = sph_cell_data.grid_index[index];

	// read particle data from sorted arrays
	float3 pos = sph_cell_data.sorted_pos[index];

	// initial density
	float rho = mass[originalIndex] * sph_kernel_poly6(0, params.effective_radius);

	// get address in grid
	int3 gridPos = calcGridPos(pos);


	// traverse 27 neighbors (fluid - fluid)
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				rho += pbf_density_0(
					neighbor_pos, index,
					pos, mass,
					sph_cell_data
					//cell_start, cell_end, gridParticleIndex
				);
			}
		}
	}

	// fluid - rigid
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_gridPos = gridPos + make_int3(x, y, z);
				rho += pbf_boundary_density_coupling(
					// fluid
					neighbor_gridPos,
					pos,
					// rigid
					dem_mass,
					dem_cell_data
				);
			}
		}
	}


	// use gridPos to traverse 27 surrounding grids (fluid - boundary)
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_gridPos = gridPos + make_int3(x, y, z);
				rho += pbf_density_coupling_vol(
					// fluid
					neighbor_gridPos,
					pos,
					// boundary
					b_volume,
					b_cell_data
				);
			}
		}
	}


	// Update date density and constraint value
	density[originalIndex] = rho;
	if ((rho / params.rest_density) - 1.f > 0.f)
		C[originalIndex] = (rho / params.rest_density) - 1.f;
	else
		C[originalIndex] = 0.f;

	//printf("rho = %f\n", rho);
	//printf("C[%u]: %f\n", originalIndex, C[originalIndex]);

}

__global__
void compute_snow_dem_sph_density_d(
	// sph
	float* sph_mass,						// input: mass of fluid paritcle
	// dem
	float* dem_mass,
	float* dem_C,
	float* dem_density,					// output: boundary density
	// boundary
	float* b_vol,
	// cell data
	CellData	sph_cell_data,
	CellData	dem_cell_data,
	CellData	b_cell_data,
	uint		dem_numParticles
) 
{

	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= dem_numParticles) return;

	// original index of boundary particle
	uint originalIndex = dem_cell_data.grid_index[index];

	// read position from sorted arrays
	float3 pos = dem_cell_data.sorted_pos[index];

	// initial density 
	float rho = dem_mass[originalIndex] * sph_kernel_poly6(0, params.effective_radius);

	// get address in grid of boundary particles (basically the same as fluid particle)
	int3 gridPos = calcGridPos(pos);


	// dem-dem
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_gridPos = gridPos + make_int3(x, y, z);
				rho += pbf_density_0(
					neighbor_gridPos, index,
					pos,
					dem_mass,
					dem_cell_data
				);
			}
		}
	}
	//printf("dem_mass: %f\n", dem_mass[originalIndex]);


	// dem-sph
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_gridPos = gridPos + make_int3(x, y, z);
				rho += pbf_boundary_density_coupling(
					// boundary
					neighbor_gridPos,
					pos,
					// fluid
					sph_mass,
					sph_cell_data
				);
			}
		}
	}


	// boundary
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_gridPos = gridPos + make_int3(x, y, z);
				rho += pbf_density_coupling_vol(
					// fluid
					neighbor_gridPos,
					pos,
					// boundary
					b_vol,
					b_cell_data
				);
			}
		}
	}	
	// Update density of fluid particle
	dem_density[originalIndex] = rho;
	// Recompute constraint value of dem particle
	if ((dem_density[originalIndex] / params.rest_density) - 1.f > 0)
		dem_C[originalIndex] = (dem_density[originalIndex] / params.rest_density) - 1.f;
	else
		dem_C[originalIndex] = 0.f;

}

__global__
void compute_snow_boundary_density_d(
	// fluid
	float* sph_mass,						// input: mass of fluid paritcle
	// dem
	float* dem_mass,
	// boundary
	float* b_mass,
	float* b_volume,
	float* b_C,
	float* b_density,					// output: boundary density
	// cell data
	CellData	sph_cell_data,
	CellData	dem_cell_data,
	CellData	b_cell_data,
	uint		b_numParticles
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= b_numParticles) return;

	// original index of boundary particle
	uint originalIndex = b_cell_data.grid_index[index];

	// read position from sorted arrays
	float3 pos = b_cell_data.sorted_pos[index];

	// initial density 
	float rho = params.rest_density * b_volume[originalIndex] * sph_kernel_poly6(0, params.effective_radius);

	// get address in grid of boundary particles (basically the same as fluid particle)
	int3 gridPos = calcGridPos(pos);


	// use gridPos to traverse 27 surrounding grids (boundary - boundary)
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_gridPos = gridPos + make_int3(x, y, z);
				rho += pbf_density_1(
					neighbor_gridPos, index,
					pos,
					b_mass,
					b_volume,
					b_cell_data
				);
			}
		}
	}


	//boundary-dem
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_gridPos = gridPos + make_int3(x, y, z);
				rho += pbf_boundary_density_coupling(
					// boundary
					neighbor_gridPos,
					pos,
					// fluid
					dem_mass,
					dem_cell_data
				);
			}
		}
	}


	// boundary-sph
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_gridPos = gridPos + make_int3(x, y, z);
				rho += pbf_boundary_density_coupling(
					// boundary
					neighbor_gridPos,
					pos,
					// fluid
					sph_mass,
					sph_cell_data
				);
			}
		}
	}

	// Update density of fluid particle
	b_density[originalIndex] = rho;
	// Recompute constraint value of fluid particle
	if ((b_density[originalIndex] / params.rest_density) - 1.f > 0)
		b_C[originalIndex] = (b_density[originalIndex] / params.rest_density) - 1.f;
	else
		b_C[originalIndex] = 0;
}

inline void compute_snow_pbf_density(
	ParticleSet* sph_particles,
	ParticleSet* dem_particles,
	ParticleSet* boundary_particles,
	CellData sph_cell_data,
	CellData dem_cell_data,
	CellData b_cell_data,
	uint sph_num_particles,
	uint dem_num_particles,
	uint b_num_particles

)
{
	uint num_threads, num_blocks;
	uint dem_num_threads, dem_num_blocks;
	uint b_num_threads, b_num_blocks;
	compute_grid_size(sph_num_particles, MAX_THREAD_NUM, num_blocks, num_threads);
	compute_grid_size(dem_num_particles, MAX_THREAD_NUM, dem_num_blocks, dem_num_threads);
	compute_grid_size(b_num_particles, MAX_THREAD_NUM, b_num_blocks, b_num_threads);

	// sph-sph density
	// sph-b density
	compute_snow_sph_density_d <<< num_blocks, num_threads >>>(
		sph_particles->m_device_data.m_d_density,
		sph_particles->m_device_data.m_d_mass,
		sph_particles->m_device_data.m_d_C,
		dem_particles->m_device_data.m_d_mass,
		boundary_particles->m_device_data.m_d_volume,
		sph_cell_data,
		dem_cell_data,
		b_cell_data,
		sph_num_particles
	);

	getLastCudaError("Kernel execution failed: compute_snow_sph_density_d ");
	
	
	// dem as sph density
	compute_snow_dem_sph_density_d <<<dem_num_blocks, dem_num_threads>>>(
		sph_particles->m_device_data.m_d_mass,
		dem_particles->m_device_data.m_d_mass,
		dem_particles->m_device_data.m_d_C,
		dem_particles->m_device_data.m_d_density,
		boundary_particles->m_device_data.m_d_volume,
		sph_cell_data,
		dem_cell_data,
		b_cell_data,
		dem_num_particles
	);
	getLastCudaError("Kernel execution failed: compute_snow_dem_sph_density_d ");
	
	// boundary density
	compute_snow_boundary_density_d << <b_num_blocks, b_num_threads >> > (
		sph_particles->m_device_data.m_d_mass,
		dem_particles->m_device_data.m_d_mass,
		boundary_particles->m_device_data.m_d_mass,
		boundary_particles->m_device_data.m_d_volume,
		boundary_particles->m_device_data.m_d_C,
		boundary_particles->m_device_data.m_d_density,
		sph_cell_data,
		dem_cell_data,
		b_cell_data,
		b_num_particles
		);

	getLastCudaError("Kernel execution failed: compute_snow_boundary_density_d ");
}

__global__
void compute_snow_sph_lambdas_d(
	float* sph_lambda,						// output: computed density
	float* sph_C,							// input: contraint
	float* sph_mass,
	float* b_volume,
	CellData sph_cell_data,
	CellData dem_cell_data,
	CellData b_cell_data,
	uint	numParticles
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	uint originalIndex = sph_cell_data.grid_index[index];

	// read particle data from sorted arrays
	float3 pos = sph_cell_data.sorted_pos[index];

	// initial density
	sph_lambda[originalIndex] = -sph_C[originalIndex];

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	float3 gradientC_i = make_float3(0);
	//-(1.f / params.rest_density) *
	//Poly6_W_Gradient_CUDA(make_float3(0, 0, 0), 0, params.effective_radius);
	float gradientC_sum = dot(gradientC_i, gradientC_i);
	
	// traverse 27 neighbors

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				float res = pbf_lambda_0(
					neighbor_pos, index,
					pos,
					sph_cell_data
				);
				gradientC_sum += res;
			}
		}
	}

	// sph-dem

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				float res = pbf_boundary_lambda(
					neighbor_pos,
					pos,
					dem_cell_data
				);
				gradientC_sum += res;
			}
		}
	}

	// traverse 27 neighbors in "boundary cells"

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				float res = pbf_lambda_boundary(
					neighbor_pos,
					pos,
					sph_mass[originalIndex],  // paritcle_mass
					b_cell_data,
					b_volume
				);
				gradientC_sum += res;
			}
		}
	}

	//printf("gradientC_sum: %f\n", gradientC_sum);
	sph_lambda[originalIndex] /= gradientC_sum + params.epsilon;

	//lambda[originalIndex] = lambda_res;
}

__global__
void compute_dem_sph_lambdas_d(
	float*   dem_lambda,				// lambda of boundary particles
	float3*  dem_pos,
	float*   dem_C,
	float*   dem_mass,
	float*	 b_volume,
	// Cell data of fluid particles
	CellData sph_cell_data,
	CellData dem_cell_data,
	CellData b_cell_data,
	uint	 dem_num_particles		// number of boundary particles
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= dem_num_particles) return;

	uint originalIndex = dem_cell_data.grid_index[index];

	// read particle data from sorted arrays
	float3 pos = dem_cell_data.sorted_pos[index];

	// initial density
	dem_lambda[originalIndex] = -dem_C[originalIndex];

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	float3 gradientC_i = make_float3(0);
	//-(1.f / params.rest_density) *
	//Poly6_W_Gradient_CUDA(make_float3(0, 0, 0), 0, params.effective_radius);
	float gradientC_sum = dot(gradientC_i, gradientC_i);

	// dem-dem
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				float res = pbf_lambda_0(
					neighbor_pos, index,
					pos,
					dem_cell_data
				);
				gradientC_sum += res;
			}
		}
	}

	// dem-sph
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				float res = pbf_boundary_lambda(
					// boundary
					neighbor_pos,
					pos,
					// fluid
					sph_cell_data
				);
				gradientC_sum += res;
			}
		}
	}

	//dem-boundary

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				float res = pbf_lambda_boundary(
					neighbor_pos,
					pos,
					dem_mass[originalIndex],  // paritcle_mass
					b_cell_data,
					b_volume
					);
				gradientC_sum += res;
			}
		}
	}



	//printf("gradientC_sum: %f\n", gradientC_sum);
	dem_lambda[originalIndex] /= gradientC_sum + params.epsilon;

	//lambda[originalIndex] = lambda_res;
}

__global__
void compute_snow_boundary_lambdas_d(
	float*	b_lambda,				// lambda of boundary particles
	float*	b_vol,
	float3* b_pos,
	float*	b_C,
	float*	b_mass,
	CellData sph_cell_data,
	CellData dem_cell_data,
	CellData b_cell_data,
	// Cell data of fluid particles
	uint	b_numParticles		// number of boundary particles
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= b_numParticles) return;

	uint originalIndex = b_cell_data.grid_index[index];

	// read particle data from sorted arrays
	float3 pos = b_cell_data.sorted_pos[index];

	// initial density
	b_lambda[originalIndex] = -b_C[originalIndex];

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	float3 gradientC_i = make_float3(0);
	//-(1.f / params.rest_density) *
	//Poly6_W_Gradient_CUDA(make_float3(0, 0, 0), 0, params.effective_radius);
	float gradientC_sum = dot(gradientC_i, gradientC_i);

 	// traverse 27 neighbors in boundary cells (boundary - boundary)

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				float res = pbf_lambda_1(
					neighbor_pos, index,
					pos,
					b_mass,
					b_cell_data.sorted_pos,
					b_cell_data.cell_start, b_cell_data.cell_end, b_cell_data.grid_index,
					b_vol
				);
				gradientC_sum += res;
			}
		}
	}

	// boundary-sph

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				float res = pbf_boundary_lambda(
					// boundary
					neighbor_pos,
					pos,
					// fluid
					sph_cell_data
				);
				gradientC_sum += res;
			}
		}
	}

	// boundary-dem

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				float res = pbf_boundary_lambda(
					// boundary
					neighbor_pos,
					pos,
					// fluid
					dem_cell_data
				);
				gradientC_sum += res;
			}
		}
	}

	//printf("gradientC_sum: %f\n", gradientC_sum);
	b_lambda[originalIndex] /= gradientC_sum + params.epsilon;

	//lambda[originalIndex] = lambda_res;
}


inline void compute_snow_pbf_lambdas(
	ParticleSet* sph_particles,
	ParticleSet* dem_particles,
	ParticleSet* boundary_particles,
	CellData sph_cell_data,
	CellData dem_cell_data,
	CellData b_cell_data,
	uint sph_num_particles,
	uint dem_num_particles,
	uint b_num_particles
	)
{
	uint num_threads, num_blocks;
	uint dem_num_threads, dem_num_blocks;
	uint b_num_threads, b_num_blocks;
	compute_grid_size(sph_num_particles, MAX_THREAD_NUM, num_blocks, num_threads);
	compute_grid_size(dem_num_particles, MAX_THREAD_NUM, dem_num_blocks, dem_num_threads);
	compute_grid_size(b_num_particles, MAX_THREAD_NUM, b_num_blocks, b_num_threads);

	// sph-sph lambdas
	// sph-b lambdas
	compute_snow_sph_lambdas_d << <num_blocks, num_threads >> > (
		sph_particles->m_device_data.m_d_lambda,
		sph_particles->m_device_data.m_d_C,
		sph_particles->m_device_data.m_d_mass,
		boundary_particles->m_device_data.m_d_volume,
		sph_cell_data,
		dem_cell_data,
		b_cell_data,
		sph_num_particles
	);
	getLastCudaError("Kernel execution failed: compute_snow_sph_lambdas_d ");

	// sph-dem lambdas (ignored if not using dem density)
	compute_dem_sph_lambdas_d << <dem_num_blocks, dem_num_threads >> > (
		dem_particles->m_device_data.m_d_lambda,
		dem_particles->m_device_data.m_d_positions,
		dem_particles->m_device_data.m_d_C,
		dem_particles->m_device_data.m_d_mass,
		boundary_particles->m_device_data.m_d_volume,
		sph_cell_data,
		dem_cell_data,
		b_cell_data,
		dem_num_particles
	);
	getLastCudaError("Kernel execution failed: compute_dem_sph_lambdas_d ");
	// boundary lambdas
	compute_snow_boundary_lambdas_d << <b_num_blocks, b_num_threads >> > (
		boundary_particles->m_device_data.m_d_lambda,
		boundary_particles->m_device_data.m_d_volume,
		boundary_particles->m_device_data.m_d_positions,
		boundary_particles->m_device_data.m_d_C,
		boundary_particles->m_device_data.m_d_mass,
		sph_cell_data,
		dem_cell_data,
		b_cell_data,
		b_num_particles
		);
	getLastCudaError("Kernel execution failed: compute_snow_boundary_lambdas_d ");
}

inline __device__
float count_neighbor_cell(
	int3 grid_pos,
	uint index,
	float3 pos,
	CellData cell_data,
	bool coupling = false
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_data.cell_start[grid_hash];
	float3 correction = make_float3(0, 0, 0);

	float count = 0;
	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			// check for repeat
			if ( coupling || (j != index) )
			{
				float dist = length(pos - cell_data.sorted_pos[j]);

				// count if close enough
				if (dist <= 2.f * params.particle_radius)
				{
					count += 1.f;
				}
			}
		}
	}
	return count;
}

inline __device__
float count_neighbor_cell(
	int3 grid_pos,
	uint index,
	float3 pos,
	CellData cell_data
	)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_data.cell_start[grid_hash];
	float3 correction = make_float3(0, 0, 0);

	float count = 0;
	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			float dist = length(pos - cell_data.sorted_pos[j]);

			// count if close enough
			if (dist <= 2.f * params.particle_radius)
			{
				count += 1.f;
			}
		}
	}
	return count;
}

inline __device__
bool sphere_cd_coupling(
	int3    grid_pos,
	float3  pos,
	CellData cell_data
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_data.cell_start[grid_hash];
	float3 correction = make_float3(0, 0, 0);

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			float3 pos2 = cell_data.sorted_pos[j];
			float3 v = pos - pos2;
			float dist = length(v);

			// correct if distance is close
			if (dist <= params.effective_radius)
			{
				return true;
			}
			
		}
	}
	return false;
}

__global__
void compute_pbf_sph_correction(
	ParticleDeviceData sph_data,
	ParticleDeviceData dem_data,
	ParticleDeviceData b_data,
	// boundary
	CellData sph_cell_data,
	CellData dem_cell_data,
	CellData b_cell_data,
	uint	numParticles,
	float	dt
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	uint original_index = sph_cell_data.grid_index[index];

	//cache data
	float* sph_lambda = sph_data.m_d_lambda;
	float* sph_C = sph_data.m_d_C;
	float* sph_contrib = sph_data.m_d_contrib;
	float3* sph_correction = sph_data.m_d_correction;
	
	float* dem_lambda = dem_data.m_d_lambda;
	float* b_lambda = b_data.m_d_lambda;
	


	// read particle data from sorted arrays
	float3 pos = sph_cell_data.sorted_pos[index];

	// initial density
	float lambda_i = sph_lambda[original_index];

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	float3 corr = make_float3(0, 0, 0);
	// traverse 27 neighbors
	// sph-sph

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				corr += pbf_correction(
					neighbor_pos, index,
					pos, lambda_i, sph_lambda, sph_contrib,
					sph_cell_data,
					dt
				);
			}
		}
	}
	
	//sph-dem

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				corr += pbf_correction_coupling(
					neighbor_pos,
					index,
					pos,
					lambda_i,
					dem_cell_data,
					dem_lambda,
					dt
				);
			}
		}
	}
	
	

	// sph-boundary

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				corr += pbf_correction_boundary(
					neighbor_pos,
					index,
					pos,
					lambda_i,
					b_cell_data,
					b_lambda,
					dt
				);
			}
		}
	}

	corr = (1.f / params.rest_density) * corr;
	corr *= sph_data.m_d_contrib[original_index];
	sph_correction[original_index] += corr;
}

// This function corrects the new dem particles comes from melting to prevent sudden changes in position
__global__
void compute_pbf_new_dem_correction(
	ParticleDeviceData sph_data,
	ParticleDeviceData dem_data,
	ParticleDeviceData b_data,
	// boundary
	CellData sph_cell_data,
	CellData dem_cell_data,
	CellData b_cell_data,
	uint	numParticles,
	float	dt
	)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	uint original_index = sph_cell_data.grid_index[index];

	//cache data
	float* sph_lambda = sph_data.m_d_lambda;
	float* sph_C = sph_data.m_d_C;
	float* sph_contrib = sph_data.m_d_contrib;

	float* dem_lambda = dem_data.m_d_lambda;
	float* b_lambda = b_data.m_d_lambda;

	// read particle data from sorted arrays
	float3 pos = dem_cell_data.sorted_pos[index];

	// initial density
	float lambda_i = dem_data.m_d_lambda[original_index];

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	float3 corr = make_float3(0, 0, 0);
	
	// traverse 27 neighbors
	// dem-sph
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				corr += pbf_correction_coupling(
					neighbor_pos, index,
					pos, lambda_i, sph_cell_data,
					sph_lambda,
					dt
					);
			}
		}
	}

	//dem-dem
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				corr += pbf_correction(
					neighbor_pos,
					index,
					pos,
					lambda_i,
					dem_data.m_d_lambda,
					dem_data.m_d_contrib,
					dem_cell_data,
					dt
					);
			}
		}
	}



	// dem-boundary
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				corr += pbf_correction_boundary(
					neighbor_pos,
					index,
					pos,
					lambda_i,
					b_cell_data,
					b_lambda,
					dt
					);
			}
		}
	}

	corr = (1.f / params.rest_density) * corr;
	corr *= (1.f - dem_data.m_d_contrib[original_index]);
	dem_data.m_d_correction[original_index] += corr;
}

inline void compute_snow_pbf_correction(
	ParticleSet* sph_particles,
	ParticleSet* dem_particles,
	ParticleSet* boundary_particles,
	CellData sph_cell_data,
	CellData dem_cell_data,
	CellData b_cell_data,
	uint sph_num_particles,
	uint dem_num_particles,
	uint b_num_particles,
	float dt
)
{
	uint sph_num_threads, sph_num_blocks;
	uint dem_num_threads, dem_num_blocks;
	compute_grid_size(sph_num_particles, MAX_THREAD_NUM, sph_num_blocks, sph_num_threads);
	compute_grid_size(dem_num_particles, MAX_THREAD_NUM, dem_num_blocks, dem_num_threads);

	// sph-sph correction
	// sph-b correction
	compute_pbf_sph_correction << <sph_num_blocks, sph_num_threads >> > (
		sph_particles->m_device_data,
		dem_particles->m_device_data,
		boundary_particles->m_device_data,
		sph_cell_data,
		dem_cell_data,
		b_cell_data,
		sph_num_particles,
		dt
	);

	
	compute_pbf_new_dem_correction << <dem_num_blocks, dem_num_threads >> > (
		sph_particles->m_device_data,
		dem_particles->m_device_data,
		boundary_particles->m_device_data,
		sph_cell_data,
		dem_cell_data,
		b_cell_data,
		dem_particles->m_size,
		dt
	);
	

	getLastCudaError("Kernel execution failed: compute_snow_sph_position_correction ");
}

inline void compute_snow_distance_correction(
	ParticleSet* sph_particles,
	ParticleSet* dem_particles,
	ParticleSet* boundary_particles,
	CellData sph_cell_data,
	CellData dem_cell_data,
	CellData b_cell_data,
	uint sph_num_particles,
	uint dem_num_particles,
	uint b_num_particles,
	bool correct_dem = false,
	bool sph_sph_correction = false
)
{
	uint sph_num_threads, sph_num_blocks;
	compute_grid_size(sph_num_particles, MAX_THREAD_NUM, sph_num_blocks, sph_num_threads);
	uint dem_num_threads, dem_num_blocks;
	compute_grid_size(dem_num_particles, MAX_THREAD_NUM, dem_num_blocks, dem_num_threads);
	

	if (correct_dem)
	{
		// correct sph particles with dem particles by distance constraint
		compute_sph_dem_distance_correction << <sph_num_blocks, sph_num_threads >> > (
			sph_particles->m_device_data,
			dem_particles->m_device_data,
			sph_cell_data,
			dem_cell_data,
			sph_num_particles
			);
		getLastCudaError("Kernel execution failed: compute_sph_dem_distance_correction ");
		
		// correct dem particles with sph particles by distance constraint
		compute_dem_sph_distance_correction << <dem_num_blocks, dem_num_threads >> > (
			dem_particles->m_device_data,
			sph_particles->m_device_data,
			dem_cell_data,
			sph_cell_data,
			dem_num_particles
			);
		getLastCudaError("Kernel execution failed: compute_sph_dem_distance_correction ");
		
	}
	if (sph_sph_correction)
	{
		compute_sph_sph_distance_correction << <sph_num_blocks, sph_num_threads >> > (
			sph_particles->m_device_data,
			sph_cell_data,
			sph_num_particles
			);
	}

	// dem-dem distance correction
	// dem-boundary distance correction
	compute_dem_distance_correction << <dem_num_blocks, dem_num_threads >> > (
		dem_particles->m_device_data,
		boundary_particles->m_device_data,
		dem_cell_data,
		b_cell_data,
		dem_num_particles
		);
	getLastCudaError("Kernel execution failed: compute_dem_correction ");
	

}

inline __device__
float3 xsph_viscosity_cell(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float3	v_i,
	float3*	vel,
	float*	mass,
	float*  density,
	CellData cell_data
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_data.cell_start[grid_hash];
	float3 res = make_float3(0, 0, 0);

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			if (j != index)
			{
				uint original_index_j = cell_data.grid_index[j];

				float3 pos2 = cell_data.sorted_pos[j];
				float3 vec = pos - pos2;
				float dist = length(vec);
				float3 v_j = vel[original_index_j];
				float3 v_i_j = v_j - v_i;

				res += (mass[original_index_j]/density[original_index_j]) * v_i_j * sph_kernel_poly6(dist, params.effective_radius);
			}
			
		}

	}
	return res;
}

inline
__device__
float3 xsph_viscosity_cell_coupling(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float3  v_i,
	float3*	other_vel,
	float*  other_mass,
	float*  other_density,
	CellData other_cell_data
	)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = other_cell_data.cell_start[grid_hash];
	float3 res = make_float3(0, 0, 0);

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = other_cell_data.cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			uint original_index_j = other_cell_data.grid_index[j];

			float3 pos2 = other_cell_data.sorted_pos[j];
			float3 vec = pos - pos2;
			float dist = length(vec);
			float3 v_j = other_vel[original_index_j];
			float3 v_i_j = v_j - v_i;

			res += (other_mass[original_index_j] / other_density[original_index_j]) * v_i_j * sph_kernel_poly6(dist, params.effective_radius);
	}

	}
	return res;
}

__global__ 
void xsph_sph_viscosity(
	float3* sph_vel,
	float*  sph_mass,
	float*  sph_density,
	float3* dem_vel,
	float*  dem_mass,
	float*	dem_density,
	CellData sph_cell_data,
	CellData dem_cell_data,
	uint sph_num_particles
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= sph_num_particles) 
		return;

	uint original_index = sph_cell_data.grid_index[index];

	cg::thread_block cta = cg::this_thread_block();
	// read particle data from sorted arrays
	float3 pos = sph_cell_data.sorted_pos[index];
	float3 v_i = sph_vel[original_index];
	// get address in grid
	int3 gridPos = calcGridPos(pos);

	float3 corr = make_float3(0, 0, 0);

	// viscosity with sph particles

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				corr += xsph_viscosity_cell(
					neighbor_pos, index,
					pos, v_i,
					sph_vel,
					sph_mass,
					sph_density,
					sph_cell_data
					);
			}
		}
	}

	// viscosity with dem particles

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				corr += xsph_viscosity_cell_coupling(
					neighbor_pos, index,
					pos, v_i,
					dem_vel,
					dem_mass,
					dem_density,
					dem_cell_data
					);
			}
		}
	}
	 
	corr *= params.sph_viscosity;

	cg::sync(cta);

	sph_vel[original_index] += corr;

}

__global__
void xsph_dem_viscosity(
	float3* dem_vel,
	float* dem_mass,
	float* dem_density,
	float3* sph_vel,
	float* sph_mass,
	float* sph_density,
	CellData dem_cell_data,
	CellData sph_cell_data,
	uint num_particles
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= num_particles)
		return;

	uint original_index = dem_cell_data.grid_index[index];

	cg::thread_block cta = cg::this_thread_block();
	// read particle data from sorted arrays
	float3 pos = dem_cell_data.sorted_pos[index];
	float3 v_i = dem_vel[original_index];
	// get address in grid
	int3 gridPos = calcGridPos(pos);

	float3 corr = make_float3(0, 0, 0);

	// viscosity with sph particles

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				corr += xsph_viscosity_cell(
					neighbor_pos, index,
					pos, v_i,
					dem_vel,
					dem_mass,
					dem_density,
					dem_cell_data
				);
			}
		}
	}

	// viscosity with dem particles

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				corr += xsph_viscosity_cell_coupling(
					neighbor_pos, index,
					pos, v_i,
					sph_vel,
					sph_mass,
					sph_density,
					sph_cell_data
				);
			}
		}
	}

	corr *= params.dem_viscosity;

	cg::sync(cta);

	dem_vel[original_index] += corr;

}


void apply_XSPH_viscosity(
	ParticleSet* sph_particles,
	ParticleSet* dem_particles,
	CellData sph_cell_data,
	CellData dem_cell_data,
	uint sph_num_particles,
	uint dem_num_particles,
	bool use_dem_viscosity=false
)
{
	uint sph_num_threads, sph_num_blocks;
	compute_grid_size(sph_num_particles, MAX_THREAD_NUM, sph_num_blocks, sph_num_threads);

	xsph_sph_viscosity <<<sph_num_blocks, sph_num_threads>>> (
		sph_particles->m_device_data.m_d_velocity,
		sph_particles->m_device_data.m_d_mass,
		sph_particles->m_device_data.m_d_density,
		dem_particles->m_device_data.m_d_velocity,
		dem_particles->m_device_data.m_d_mass,
		dem_particles->m_device_data.m_d_density,
		sph_cell_data,
		dem_cell_data,
		sph_num_particles
	);
	getLastCudaError("Kernel execution failed : xsph_sph_viscosity");

	if (use_dem_viscosity)
	{
		uint dem_num_threads, dem_num_blocks;
		compute_grid_size(dem_num_particles, MAX_THREAD_NUM, dem_num_blocks, dem_num_threads);
		xsph_dem_viscosity << <sph_num_blocks, sph_num_threads >> > (
			dem_particles->m_device_data.m_d_velocity,
			dem_particles->m_device_data.m_d_mass,
			dem_particles->m_device_data.m_d_density,
			sph_particles->m_device_data.m_d_velocity,
			sph_particles->m_device_data.m_d_mass,
			sph_particles->m_device_data.m_d_density,
			dem_cell_data,
			sph_cell_data,
			dem_num_particles
			);
		getLastCudaError("Kernel execution failed : xsph_dem_viscosity");
	}
}

__global__
void counting_neighbors_d(
	float* neighbor_count0,
	CellData cell_data0,
	CellData cell_data1,
	uint num_particles
	)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= num_particles)
		return;

	uint original_index = cell_data0.grid_index[index];

	// read particle data from sorted arrays
	float3 pos = cell_data0.sorted_pos[index];
	// get address in grid
	int3 gridPos = calcGridPos(pos);


	float count = 0;
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				count += count_neighbor_cell(gridPos, index, pos, cell_data0, false);
			}
		}
	}

	// count with sph particles
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				count += count_neighbor_cell(gridPos, index, pos, cell_data1, true);
			}
		}
	}

	// update neighbor_count 
	neighbor_count0[original_index] = count;
}

inline
__device__
float heat_transfer_cell(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float   T_i,
	float   k_i,
	float	k_j,
	ParticleDeviceData other_data,
	CellData target_cell_data,
	bool	coupling=false
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = target_cell_data.cell_start[grid_hash];
	float result = 0.f;

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = target_cell_data.cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			// skip if not doing coupling (same cell)
			if (!coupling && j == index)
				continue;

			uint original_index_j = target_cell_data.grid_index[j];

			float3 pos2 = target_cell_data.sorted_pos[j];
			float3 vec = pos - pos2;
			float dist = length(vec);

			result += 
				(other_data.m_d_mass[original_index_j] / other_data.m_d_density[original_index_j]) 
				* (4.f * k_i * k_j/ (k_i + k_j)) 
				* (other_data.m_d_T[original_index_j] - T_i)
				* sph_kernel_viscosity_laplacian(dist, params.effective_radius);
		}

	}
	return result;
}

__global__
void transfer_heat_sph(
	ParticleDeviceData sph_data,
	ParticleDeviceData dem_data,
	CellData sph_cell_data,
	CellData dem_cell_data,
	uint sph_num_particles,
	float dt
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= sph_num_particles)
		return;

	uint original_index = sph_cell_data.grid_index[index];

	// read particle data from sorted arrays
	float3 pos = sph_cell_data.sorted_pos[index];
	// get address in grid
	int3 gridPos = calcGridPos(pos);

	//read data from sph_data
	float T_i = sph_data.m_d_T[original_index];


	float dT = 0.f;
	//sph-sph
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				dT += heat_transfer_cell(gridPos, index, pos, T_i, params.k_water, params.k_water, sph_data, sph_cell_data, false);
			}
		}
	}

	//sph-dem
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				dT += heat_transfer_cell(gridPos, index, pos, T_i, params.k_water, params.k_snow, dem_data, dem_cell_data, true);
			}
		}
	}

	/*
	float factor = params.C_water * sph_data.m_d_density[original_index];
	if (index == 0)
	{
		printf("Factor: %f\n", factor);
		printf("dT: %f\n", dT);
	}
	*/

	dT /= params.C_water * sph_data.m_d_density[original_index];

	//write to new_T
	sph_data.m_d_new_T[original_index] = T_i + dT * dt;
}

__global__
void transfer_heat_dem(
	ParticleDeviceData sph_data,
	ParticleDeviceData dem_data,
	CellData sph_cell_data,
	CellData dem_cell_data,
	uint dem_num_particles,
	float dt
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= dem_num_particles)
		return;

	uint original_index = dem_cell_data.grid_index[index];

	// read particle data from sorted arrays
	float3 pos = dem_cell_data.sorted_pos[index];
	// get address in grid
	int3 gridPos = calcGridPos(pos);

	//read data from sph_data
	float T_i = dem_data.m_d_T[original_index];


	float dT = 0.f;
	//dem-dem
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				dT += heat_transfer_cell(gridPos, index, pos, T_i, params.k_snow, params.k_snow, dem_data, dem_cell_data, false);
			}
		}
	}

	//dem-sph
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				dT += heat_transfer_cell(gridPos, index, pos, T_i, params.k_snow, params.k_water, sph_data, sph_cell_data, true);
			}
		}
	}

	dT /= params.C_snow * dem_data.m_d_density[original_index];

	//write to new_T
	dem_data.m_d_new_T[original_index] = T_i + dT * dt;
}

void transfer_heat(
	ParticleDeviceData sph_data, 
	ParticleDeviceData dem_data, 
	CellData sph_cell_data, 
	CellData dem_cell_data,
	uint sph_num_particles,
	uint dem_num_particles,
	float dt,
	uint sph_num_blocks,
	uint sph_num_threads,
	uint dem_num_blocks,
	uint dem_num_threads
)
{
	// Be careful to R/W problem
	// transfer heat between (sph-sph sph-dem), (dem-dem dem-sph)
	transfer_heat_sph <<<sph_num_blocks, sph_num_threads>>>(sph_data, dem_data, sph_cell_data, dem_cell_data, sph_num_particles, dt);
	transfer_heat_dem <<<dem_num_blocks, dem_num_threads>>>(sph_data, dem_data, sph_cell_data, dem_cell_data, dem_num_particles, dt);

	// Cast to device pointer
	thrust::device_ptr<float> sph_T_ptr = thrust::device_pointer_cast(sph_data.m_d_T);
	thrust::device_ptr<float> sph_new_T_ptr = thrust::device_pointer_cast(sph_data.m_d_new_T);

	thrust::device_ptr<float> dem_T_ptr = thrust::device_pointer_cast(dem_data.m_d_T);
	thrust::device_ptr<float> dem_new_T_ptr = thrust::device_pointer_cast(dem_data.m_d_new_T);

	// Write back
	//thrust::fill(sph_T_ptr, sph_T_ptr + sph_num_particles, 10.f);
	//thrust::fill(dem_T_ptr, dem_T_ptr + dem_num_particles, -20.f);
	thrust::copy(sph_new_T_ptr, sph_new_T_ptr + sph_num_particles, sph_T_ptr);
	thrust::copy(dem_new_T_ptr, dem_new_T_ptr + dem_num_particles, dem_T_ptr);
}


// copy data0[index] to data1[target_index]
inline __device__
void copy_particle_info(ParticleDeviceData src, ParticleDeviceData dst, uint index, uint target_index)
{
	dst.m_d_positions[target_index] = src.m_d_positions[index];
	dst.m_d_predict_positions[target_index] = src.m_d_predict_positions[index];
	dst.m_d_new_positions[target_index] = src.m_d_new_positions[index];
	dst.m_d_velocity[target_index] = src.m_d_velocity[index];
	dst.m_d_force[target_index] = src.m_d_force[index];
	dst.m_d_correction[target_index] = src.m_d_correction[index];
	dst.m_d_mass[target_index] = src.m_d_mass[index];
	dst.m_d_massInv[target_index] = src.m_d_massInv[index];
	dst.m_d_contrib[target_index] = src.m_d_contrib[index];
	dst.m_d_density[target_index] = src.m_d_density[index];
	dst.m_d_C[target_index] = src.m_d_C[index];
	dst.m_d_lambda[target_index] = src.m_d_lambda[index];
	dst.m_d_T[target_index] = src.m_d_T[index];
	dst.m_d_new_T[target_index] = src.m_d_new_T[index];
	dst.m_d_trackId[target_index] = src.m_d_trackId[index];

	// copy connect record
	for (uint i = 0; i < params.maximum_connection; ++i)
	{
		const uint src_record_idx = index * params.maximum_connection;
		const uint target_record_idx = target_index * params.maximum_connection;
		dst.m_d_connect_record[target_record_idx + i] = src.m_d_connect_record[src_record_idx + i];
		dst.m_d_connect_length[target_record_idx + i] = src.m_d_connect_length[src_record_idx + i];
	}
	dst.m_d_iter_end[target_index] = src.m_d_iter_end[index];
}

inline __device__
void clean_particle_info(ParticleDeviceData dst, uint target_index)
{
	/*
	dst.m_d_positions[target_index] = make_float3(INFINITY);
	dst.m_d_predict_positions[target_index] = make_float3(INFINITY);
	dst.m_d_new_positions[target_index] = make_float3(INFINITY);
	dst.m_d_velocity[target_index] = make_float3(0);
	dst.m_d_force[target_index] = make_float3(0);
	dst.m_d_correction[target_index] = make_float3(0);
	//dst.m_d_mass[target_index] = src.m_d_mass[index];
	//dst.m_d_massInv[target_index] = src.m_d_massInv[index];
	dst.m_d_density[target_index] = 0;
	dst.m_d_C[target_index] = 0;
	dst.m_d_lambda[target_index] = 0;
	dst.m_d_T[target_index] = INFINITY;
	dst.m_d_new_T[target_index] = INFINITY;
	*/
	dst.m_d_predicate[target_index] = 0;
	dst.m_d_trackId[target_index] = 0;
	dst.m_d_scan_index[target_index] = 0; // <- maybe this doesn't need to update

	//reset refreezing parameters
	for (uint i = 0; i < params.maximum_connection; ++i)
	{
		const uint target_record_idx = target_index * params.maximum_connection;
		dst.m_d_connect_record[target_record_idx + i] = UINT_MAX;
		dst.m_d_connect_length[target_record_idx + i] = 0.f;
	}
	dst.m_d_iter_end[target_index] = UINT_MAX;
}


__device__
void print_particle_info(ParticleDeviceData data, uint index, const char* str)
{
	printf("%s\n%u: pos: %f, %f, %f vel: %f corr: %f, %f, %f T: %f density: %f C:%f lambda: %f\n", 
		str,
		index, 
		data.m_d_positions[index].x, data.m_d_positions[index].y, data.m_d_positions[index].z, 
		length(data.m_d_velocity[index]),
		data.m_d_correction[index].x, data.m_d_correction[index].y, data.m_d_correction[index].z,
		data.m_d_T[index], 
		data.m_d_density[index], 
		data.m_d_C[index], 
		data.m_d_lambda[index]
	);
}

__global__
void melting(ParticleDeviceData sph_data, ParticleDeviceData dem_data, uint num_particles, uint frame_count)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= num_particles)
		return;

	// melt -> put information to sph's tail
	if (dem_data.m_d_T[index] > params.freezing_point && dem_data.m_d_contrib[index] >= 0.99f)
	{
		uint target_index;
		target_index = atomicAdd(sph_data.m_d_new_end, 1u);
		
		dem_data.m_d_contrib[index] = 0;

		//printf("(%u) DEM %u@%u \tto\t %u\n", frame_count ,dem_data.m_d_trackId[index], index, target_index);
		//print_particle_info(dem_data, index, "DEM(Before)");
		//print_particle_info(sph_data, target_index, "SPH");
		copy_particle_info(dem_data, sph_data, index, target_index);
		//print_particle_info(dem_data, index, "DEM(After)");
		//print_particle_info(sph_data, target_index, "SPH");

		dem_data.m_d_predicate[index] = 0;
		//set track id to 0 => not using
		dem_data.m_d_trackId[index] = 0; 
		sph_data.m_d_predicate[target_index] = 1;
	}
}

__global__
void freezing(ParticleDeviceData sph_data, ParticleDeviceData dem_data, uint num_particles, uint frame_count)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= num_particles)
		return;
	
	// freeze -> put this particle's information to dem's tail
	if (sph_data.m_d_T[index] <= params.freezing_point)// && sph_data.m_d_contrib[index] >= 0.99f)
	{
		uint target_index;
		target_index = atomicAdd(dem_data.m_d_new_end, 1u);
		//printf("target_index: %u\n", target_index);
		sph_data.m_d_contrib[index] = 0;

		//printf("(%u) SPH %u@%u \tto\t %u\n", frame_count, sph_data.m_d_trackId[index], index, target_index);

		copy_particle_info(sph_data, dem_data, index, target_index);

		sph_data.m_d_predicate[index] = 0;
		sph_data.m_d_trackId[index] = 0;

		dem_data.m_d_predicate[target_index] = 1;
		// reset refreezing data on dem particles
		for (uint i = 0; i < params.maximum_connection; ++i)
		{
			const uint record_target_index = target_index * params.maximum_connection + i;
			dem_data.m_d_connect_record[record_target_index] = UINT_MAX;
			dem_data.m_d_connect_length[record_target_index] = 0.f;
		}
		dem_data.m_d_iter_end[target_index] = target_index;
	}
}


__global__
void copy_to_target(ParticleDeviceData src, ParticleDeviceData dst, uint num_particles)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= num_particles)
		return;

	copy_particle_info(src, dst, index, index);
}

__global__
void scatter(ParticleDeviceData target_data, ParticleDeviceData buffer, uint num_particles)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= num_particles)
		return;

	if (target_data.m_d_predicate[index] == 1)
	{
		uint target_index = target_data.m_d_scan_index[index];
		copy_particle_info(buffer, target_data, index, target_index);
	}
}

__global__
void clean_tail(ParticleDeviceData data, uint num_particles)
{
	// offset to new end
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x + (*data.m_d_new_end);

	if (index >= num_particles)
		return;

	clean_particle_info(data, index);

}

template <typename T>
void print_vec(std::vector<T> vec0, std::vector<T> vec1)
{
	for (size_t i=0;i<vec0.size();++i)
		std::cout << vec0[i] << " " << vec1[i] << "\t";
	std::cout << "\n";
}

void verification(ParticleSet* sph_particles, ParticleSet* dem_particles)
{
	if (sph_particles->m_full_size != dem_particles->m_full_size)
		return;

	uint ans_count = 0;
	uint n = sph_particles->m_full_size;


	std::vector<uint> sph_id_vec(n, 0u);
	std::vector<uint> dem_id_vec(n, 0u);

	std::vector<uint> sph_predicate_vec(n, 0u);
	std::vector<uint> dem_predicate_vec(n, 0u);


	// copy to host vector
	cudaMemcpy(sph_id_vec.data(), sph_particles->m_device_data.m_d_trackId, n * sizeof(uint), cudaMemcpyDeviceToHost);
	cudaMemcpy(dem_id_vec.data(), dem_particles->m_device_data.m_d_trackId, n * sizeof(uint), cudaMemcpyDeviceToHost);

	cudaMemcpy(sph_predicate_vec.data(), sph_particles->m_device_data.m_d_predicate, n * sizeof(uint), cudaMemcpyDeviceToHost);
	cudaMemcpy(dem_predicate_vec.data(), dem_particles->m_device_data.m_d_predicate, n * sizeof(uint), cudaMemcpyDeviceToHost);

	//thrust::device_ptr<uint> sph_id_ptr = thrust::device_pointer_cast(sph_particles->m_device_data.m_d_trackId);
	//thrust::device_ptr<uint> dem_id_ptr = thrust::device_pointer_cast(dem_particles->m_device_data.m_d_trackId);

	// Is all ID unique?
	for (uint i = 1; i <= n; ++i)
	{
		uint tmp = 0;
		//printf("searching %u... ", i);
		//search in host vector
		for (auto it = sph_id_vec.cbegin(); it != sph_id_vec.cend(); ++it)
		{
			if (*it == i) 
				ans_count++, tmp++;
		}

		for (auto it = dem_id_vec.cbegin(); it != dem_id_vec.cend(); ++it)
		{
			if (*it == i)
				ans_count++, tmp++;
		}
		if (tmp > 1)
			printf("\nExceed at %u", i);
	}

	if (ans_count == n)
		printf("Verification success\n");
	else if (ans_count > n)
	{
		printf("Verification failed (exceed) n=%u, ans_count=%u\n", n, ans_count);
		printf("SPH ");
		print_vec(sph_id_vec, sph_predicate_vec);
		printf("DEM ");
		print_vec(dem_id_vec, dem_predicate_vec);
	}
	else if (ans_count < n)
	{
		printf("Verification failed (less) n=%u, ans_count=%u\n", n, ans_count);
		printf("SPH\n");
		print_vec(sph_id_vec, sph_predicate_vec);
		printf("DEM\n");
		print_vec(dem_id_vec, dem_predicate_vec);
	}


}


void compact_and_clean(ParticleSet* sph_particles, ParticleSet* dem_particles, ParticleDeviceData buffer)
{
	static uint count=0, sph_size=0, dem_size=0;
	static bool debug_flag = false;

	uint num_threads, num_blocks, full_num_threads, full_num_blocks;
	uint sph_new_size, dem_new_size;
	uint full_size = sph_particles->m_full_size;

	thrust::device_ptr<uint> sph_predicate_ptr = thrust::device_pointer_cast(sph_particles->m_device_data.m_d_predicate);
	thrust::device_ptr<uint> dem_predicate_ptr = thrust::device_pointer_cast(dem_particles->m_device_data.m_d_predicate);

	thrust::device_ptr<uint> sph_scan_ptr = thrust::device_pointer_cast(sph_particles->m_device_data.m_d_scan_index);
	thrust::device_ptr<uint> dem_scan_ptr = thrust::device_pointer_cast(dem_particles->m_device_data.m_d_scan_index);

	sph_new_size = thrust::count(sph_predicate_ptr, sph_predicate_ptr + full_size, 1u);
	dem_new_size = thrust::count(dem_predicate_ptr, dem_predicate_ptr + full_size, 1u);

	thrust::exclusive_scan(sph_predicate_ptr, sph_predicate_ptr + full_size, sph_scan_ptr, 0u);
	thrust::exclusive_scan(dem_predicate_ptr, dem_predicate_ptr + full_size, dem_scan_ptr, 0u);

	/*
	if (debug_flag)
		printf("Before verification: "),  verification(sph_particles, dem_particles), debug_flag=false;
	*/
	// copy sph to tmp
	compute_grid_size(full_size, MAX_THREAD_NUM, full_num_blocks, full_num_threads);
	copy_to_target << <full_num_blocks, full_num_threads >> > (sph_particles->m_device_data, buffer, full_size);
	compute_grid_size(sph_particles->m_size, MAX_THREAD_NUM, num_blocks, num_threads);
	scatter <<<full_num_blocks, full_num_threads >>> (sph_particles->m_device_data, buffer, full_size);
	getLastCudaError("Kernel execution failed: scatter ");
	cudaDeviceSynchronize();

	// copy dem to tmp
	copy_to_target << <full_num_blocks, full_num_threads >> > (dem_particles->m_device_data, buffer, full_size);
	compute_grid_size(dem_particles->m_size, MAX_THREAD_NUM, num_blocks, num_threads);
	scatter <<< full_num_blocks, full_num_threads >>> (dem_particles->m_device_data, buffer, full_size);
	getLastCudaError("Kernel execution failed: scatter ");
	cudaDeviceSynchronize();
	// reset size on CPU for draw call
	sph_particles->setSize(sph_new_size);
	dem_particles->setSize(dem_new_size);
	
	// copy data to GPU so that we can know how much particles are simulating now
	cudaMemcpy(sph_particles->m_device_data.m_d_new_end, &sph_new_size, sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemcpy(dem_particles->m_device_data.m_d_new_end, &dem_new_size, sizeof(uint), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	// clean
	
	// fill front (reserve)
	thrust::fill(sph_predicate_ptr, sph_predicate_ptr + sph_new_size, 1u);
	thrust::fill(dem_predicate_ptr, dem_predicate_ptr + dem_new_size, 1u);
	
	// fill tail (clean)
	//thrust::fill(sph_predicate_ptr + sph_new_size, sph_predicate_ptr + full_size, 0u);
	//thrust::fill(dem_predicate_ptr + dem_new_size, dem_predicate_ptr + full_size, 0u);
	
	
	uint sph_tail_size = full_size - sph_new_size, dem_tail_size = full_size - dem_new_size;
	compute_grid_size(sph_tail_size, MAX_THREAD_NUM, num_blocks, num_threads);
	clean_tail <<<num_blocks, num_threads>>> (sph_particles->m_device_data, sph_tail_size);
	getLastCudaError("Kernel execution failed: clean_tail ");

	compute_grid_size(dem_tail_size, MAX_THREAD_NUM, num_blocks, num_threads);
	clean_tail <<<num_blocks, num_threads>>> (dem_particles->m_device_data, dem_tail_size);
	getLastCudaError("Kernel execution failed: clean_tail ");
	cudaDeviceSynchronize();
	
	/*
	if (sph_size != sph_particles->m_size)
		printf("(%u)New SPH size: %u\n", count, sph_particles->m_size), sph_size = sph_particles->m_size;
	if (dem_size != dem_particles->m_size)
		printf("(%u)New DEM size: %u\nAfter verification: ", count, dem_particles->m_size), dem_size = dem_particles->m_size, count++, verification(sph_particles, dem_particles), debug_flag=true;
	*/
	
}

void phase_change(
	ParticleSet* sph_particles,
	ParticleSet* dem_particles,
	ParticleDeviceData  buffer,
	bool simulate_freezing = true,
	bool simulate_melting = true
)
{
	static uint frame_count = 0;
	if (sph_particles->m_full_size != dem_particles->m_full_size)
	{
		printf("Invalid GPU mem size (predicate)\n");
		return;
	}

	//uint full_size = sph_particles->m_full_size;
	uint num_threads, num_blocks;
	//predicate_and_fill << <num_blocks, num_threads >> > (sph_particles->m_device_data, dem_particles->m_device_data,  full_size);
	if (simulate_freezing)
	{
		compute_grid_size(sph_particles->m_size, MAX_THREAD_NUM, num_blocks, num_threads);
		freezing << <num_blocks, num_threads >> > (sph_particles->m_device_data, dem_particles->m_device_data, sph_particles->m_size, frame_count);
		cudaDeviceSynchronize();
	}
	if (simulate_melting)
	{
		compute_grid_size(dem_particles->m_size, MAX_THREAD_NUM, num_blocks, num_threads);
		melting << <num_blocks, num_threads >> > (sph_particles->m_device_data, dem_particles->m_device_data, dem_particles->m_size, frame_count);
		cudaDeviceSynchronize();
	}
	compact_and_clean(sph_particles, dem_particles, buffer);
	frame_count++;
}

void compute_mass_scale_factor(
	ParticleSet* sph_particles,
	ParticleSet* dem_particles,
	ParticleSet* b_particles
)
{
}


void snow_simulation(
	ParticleSet* sph_particles,
	ParticleSet* dem_particles, 
	ParticleSet* boundary_particles, 
	ParticleDeviceData* phase_change_buffer,
	CellData sph_cell_data, 
	CellData dem_cell_data, 
	CellData b_cell_data, 
	float dt,
	int iterations,
	bool correct_dem,
	bool sph_sph_correction,
	//bool compute_wetness,
	bool dem_friction,
	bool compute_temperature,
	bool change_phase,
	bool simulate_freezing,
	bool simulate_melting,
	bool dem_viscosity,
	bool cd_on
)
{
	uint sph_num_threads, sph_num_blocks;
	compute_grid_size(sph_particles->m_size, MAX_THREAD_NUM, sph_num_blocks, sph_num_threads);
	uint dem_num_threads, dem_num_blocks;
	compute_grid_size(dem_particles->m_size, MAX_THREAD_NUM, dem_num_blocks, dem_num_threads);

	sort_and_reorder(
		dem_particles->m_device_data.m_d_predict_positions,
		dem_cell_data,
		dem_particles->m_size
	);

	sort_and_reorder(
		sph_particles->m_device_data.m_d_predict_positions,
		sph_cell_data,
		sph_particles->m_size
	);

	compute_snow_pbf_density(
		sph_particles,
		dem_particles,
		boundary_particles,
		sph_cell_data,
		dem_cell_data,
		b_cell_data,
		sph_particles->m_size,
		dem_particles->m_size,
		boundary_particles->m_size
	);

	if (compute_temperature)
	{
		// transfer heat
		transfer_heat(
			sph_particles->m_device_data,
			dem_particles->m_device_data,
			sph_cell_data,
			dem_cell_data,
			sph_particles->m_size,
			dem_particles->m_size,
			dt,
			sph_num_blocks,
			sph_num_threads,
			dem_num_blocks,
			dem_num_threads
		);
	}

	if(change_phase)
		phase_change(sph_particles, dem_particles, *phase_change_buffer, simulate_freezing, simulate_melting);


	integrate_pbd(sph_particles, dt, sph_particles->m_size, cd_on);
	integrate_pbd(dem_particles, dt, dem_particles->m_size, cd_on);


	for (int i = 0; i < iterations; ++i)
	{
		// CAUTION: Must do sort() again after phase_change()
		sort_and_reorder(
			dem_particles->m_device_data.m_d_predict_positions,
			dem_cell_data,
			dem_particles->m_size
		);

		sort_and_reorder(
			sph_particles->m_device_data.m_d_predict_positions,
			sph_cell_data,
			sph_particles->m_size
		);
		
		compute_snow_pbf_density(
			sph_particles,
			dem_particles,
			boundary_particles,
			sph_cell_data,
			dem_cell_data,
			b_cell_data,
			sph_particles->m_size,
			dem_particles->m_size,
			boundary_particles->m_size
			);

		compute_snow_pbf_lambdas(
			sph_particles,
			dem_particles,
			boundary_particles,
			sph_cell_data,
			dem_cell_data,
			b_cell_data,
			sph_particles->m_size,
			dem_particles->m_size,
			boundary_particles->m_size
			); 

		compute_snow_pbf_correction(
			sph_particles,
			dem_particles,
			boundary_particles,
			sph_cell_data,
			dem_cell_data,
			b_cell_data,
			sph_particles->m_size,
			dem_particles->m_size,
			boundary_particles->m_size,
			dt
			);
		
		compute_snow_distance_correction(
			sph_particles,
			dem_particles,
			boundary_particles,
			sph_cell_data,
			dem_cell_data,
			b_cell_data,
			sph_particles->m_size,
			dem_particles->m_size,
			boundary_particles->m_size,
			correct_dem,
			sph_sph_correction
			);

		// refreezing proccess
		// refreezing();

		
		apply_correction << <sph_num_blocks, sph_num_threads >> > (
			sph_particles->m_device_data,
			sph_cell_data,
			sph_particles->m_size
		);
		getLastCudaError("Kernel execution failed: apply_correction ");

		apply_correction << <dem_num_blocks, dem_num_threads >> > (
			dem_particles->m_device_data,
			dem_cell_data,
			dem_particles->m_size
			);
		getLastCudaError("Kernel execution failed: apply_correction ");

	}

	if (dem_friction)
	{
		sort_and_reorder(
			dem_particles->m_device_data.m_d_predict_positions,
			dem_cell_data,
			dem_particles->m_size
		);

		sort_and_reorder(
			sph_particles->m_device_data.m_d_predict_positions,
			sph_cell_data,
			sph_particles->m_size
		);

		compute_friction_correction << <dem_num_blocks, dem_num_threads >> > (
			dem_particles->m_device_data.m_d_correction,
			dem_particles->m_device_data.m_d_new_positions,
			dem_particles->m_device_data.m_d_positions,
			dem_particles->m_device_data.m_d_massInv,
			boundary_particles->m_device_data.m_d_massInv,
			dem_cell_data,
			b_cell_data,
			dem_particles->m_size
			);
		getLastCudaError("Kernel execution failed: compute_friction_correction ");

		apply_correction << <dem_num_blocks, dem_num_threads >> > (
			dem_particles->m_device_data,
			dem_cell_data,
			dem_particles->m_size
			);
		getLastCudaError("Kernel execution failed: apply_correction ");
	}
	/*
	*/
	// finialize corrections and compute velocity for next integration 
	finalize_correction << <sph_num_blocks, sph_num_threads >> > (
		sph_particles->m_device_data.m_d_positions,
		sph_particles->m_device_data.m_d_new_positions,
		sph_particles->m_device_data.m_d_predict_positions,
		sph_particles->m_device_data.m_d_velocity,
		sph_particles->m_size,
		dt
	);
	getLastCudaError("Kernel execution failed: finalize_correction ");

	finalize_correction << <dem_num_blocks, dem_num_threads >> > (
		dem_particles->m_device_data.m_d_positions,
		dem_particles->m_device_data.m_d_new_positions,
		dem_particles->m_device_data.m_d_predict_positions,
		dem_particles->m_device_data.m_d_velocity,
		dem_particles->m_size,
		dt
		);
	getLastCudaError("Kernel execution failed: finalize_correction ");

	
	apply_XSPH_viscosity(
		sph_particles, 
		dem_particles,
		sph_cell_data,
		dem_cell_data,
		sph_particles->m_size,
		dem_particles->m_size,
		dem_viscosity
	);


}
