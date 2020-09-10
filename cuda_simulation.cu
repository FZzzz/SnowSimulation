#include <cuda_runtime.h>
#include <cstdlib>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <device_atomic_functions.h>
#include <helper_math.h>
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <cooperative_groups.h>
#include "cuda_simulation.cuh"
#include <chrono>
#include "imgui/imgui.h"

#define NUM_CELLS 262144

namespace cg = cooperative_groups;

/*SPH Kernels*/
inline __device__ float sph_kernel_Poly6_W_CUDA(float distance, float effective_radius)
{
	if (distance >= 0 && distance <= effective_radius)
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

inline __device__ float3 sph_kernel_Poly6_W_Gradient_CUDA(float3 diff, float distance, float effective_radius)
{
	if (distance >= 0 && distance <= effective_radius)
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


inline __device__ float sph_kernel_Spiky_W_CUDA(float distance, float effective_radius)
{
	if (distance >= 0 && distance <= effective_radius)
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


inline __device__ float3 sph_kernel_Spiky_W_Gradient_CUDA(float3 diff, float distance, float effective_radius)
{
	if (distance >= 0 && distance <= effective_radius)
	{
		const float h = effective_radius;
		//float h6 = pow(h, 6);
		float q = h - distance;
		float q2 = q * q;

		//float scalar = (-45.0f / (CUDART_PI*h6)) * (q2 / distance);
		float scalar = params.spiky_G * (q2 / distance);
		float3 result = scalar * diff;// make_float3(diff.x, diff.y, diff.z);

		return result;
	}
	else
	{
		return make_float3(0, 0, 0);
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
				rho += mass[original_index] * sph_kernel_Poly6_W_CUDA(dist, params.effective_radius);
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
/*
__global__
void reorderData_boundary_D(
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
			cell_data.cellStart[hash] = index;

			if (index > 0)
				cell_data.cell_end[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1)
		{
			cell_data.cell_end[hash] = index + 1;
		}

		// Now use the sorted index to reorder the pos data
		uint sortedIndex = cell_data.grid_index[index];
		float3 pos = oldPos[sortedIndex];

		cell_data.sorted_pos[index] = pos;
	}
}
*/

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
	float rho = mass[originalIndex] * sph_kernel_Poly6_W_CUDA(0, params.effective_radius);

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


/*
void reorderData_boundary(
	CellData cell_data, 
	float3* oldPos, 
	uint numParticles, 
	uint numCells)
{
	uint num_threads, num_blocks;
	compute_grid_size(numParticles, MAX_THREAD_NUM, num_blocks, num_threads);

	// set all cells to empty
	checkCudaErrors(cudaMemset(cell_data.cellStart, 0xffffffff, numCells * sizeof(uint)));

	uint smemSize = sizeof(uint) * (num_threads + 1);
	reorderData_boundary_D << < num_blocks, num_threads, smemSize >> > (
		cell_data,
		oldPos,
		numParticles);
	getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

}
*/

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
	float3* pos, float3* vel, float3* force, float* massInv,
	float3* predict_pos, float3* new_pos,
	float dt,
	uint numParticles)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	float3 t_vel = vel[index] + dt * params.gravity;
	t_vel = t_vel * params.global_damping;
	float3 t_pos = pos[index] + dt * t_vel;

	
	if (t_pos.x >= 1.0f)
	{
		t_pos.x = 1.f;
		t_vel.x =  -abs(t_vel.x);
		t_vel *= params.boundary_damping;
	}

	if (t_pos.x <= -1.0f)
	{
		t_pos.x = -1.f;
		t_vel.x = abs(t_vel.x);
		t_vel *= params.boundary_damping;
	}

	if (t_pos.z >= 1.0f)
	{
		t_pos.z = 1.f;
		t_vel.z = -abs(t_vel.z);
		t_vel *= params.boundary_damping;
	}

	if (t_pos.z <= -1.0f)
	{
		t_pos.z = -1.f;
		t_vel.z = abs(t_vel.z);
		t_vel *= params.boundary_damping;
	}
	
	
	if (pos[index].y <= 0.f + params.particle_radius)
	{
		pos[index].y = 0.f;
		t_vel.y = abs(t_vel.y);
		t_vel *= params.boundary_damping;
	}
	

	// Velocity limitation
	
	const float limit = params.maximum_speed;
	if (length(t_vel) > limit)
	{
		t_vel = (limit / length(t_vel)) * t_vel ;
	}
	

	predict_pos[index] = pos[index] + dt * t_vel;
	vel[index] = t_vel;
	new_pos[index] = predict_pos[index];


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

				rho = mass[original_index] * sph_kernel_Poly6_W_CUDA(dist, params.effective_radius);

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

				rho = params.rest_density * b_volume[original_index] * sph_kernel_Poly6_W_CUDA(dist, params.effective_radius);

				density += rho;
			}
		}
	}

	return density;
}


inline __device__
float pbf_density_boundary(
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

			float rho = params.rest_density * volume[original_index] * sph_kernel_Poly6_W_CUDA(dist, params.effective_radius);

			density += rho;	
		}
	}

	// return contributions of boundary paritcles
	return density;
}

// boundary - fluid
inline __device__
float pbf_boundary_density(
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

			float rho = mass[original_index] * sph_kernel_Poly6_W_CUDA(dist, params.effective_radius);

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
					sph_kernel_Poly6_W_Gradient_CUDA(vec, dist, params.effective_radius);

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
						sph_kernel_Poly6_W_Gradient_CUDA(vec, dist, params.effective_radius);

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
				sph_kernel_Poly6_W_Gradient_CUDA(vec, dist, params.effective_radius);

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
				sph_kernel_Poly6_W_Gradient_CUDA(vec, dist, params.effective_radius);

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
	/*
	float3* sorted_pos,
	
	uint*	cell_start,
	uint*	cell_end,
	uint*	gridParticleIndex,
	*/
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

				const float3 gradient = sph_kernel_Poly6_W_Gradient_CUDA(vec, dist, params.effective_radius);
				
				float x = sph_kernel_Poly6_W_CUDA(dist, params.effective_radius) / 
					sph_kernel_Poly6_W_CUDA(0.3f * params.effective_radius, params.effective_radius);
				x = pow(x, 4);
				const float scorr = -params.scorr_coeff * x * dt * dt * dt;
				
				//printf("scorr: %f\n", scorr);

				//float3 res = //(1.f / params.rest_density) *
					
				correction += (lambda_i + lambdas[original_index] + scorr) * gradient;
				;
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

			float3 gradient = sph_kernel_Poly6_W_Gradient_CUDA(vec, dist, params.effective_radius);

			float scorr = -0.1f;
			float x = sph_kernel_Poly6_W_CUDA(dist, params.effective_radius) /
				sph_kernel_Poly6_W_CUDA(0.3f * params.effective_radius, params.effective_radius);
			x = pow(x, 4);
			scorr = scorr * x * dt * dt;

			//printf("scorr: %f\n", scorr);

			float3 res = //(1.f / params.rest_density) *
				(lambda_i + lambda_j) *// +scorr)*
				gradient;

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
	CellData b_cell_data,
	float*   b_lambdas,
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
			if (j != index)                // check not colliding with self
			{
				uint original_index = b_cell_data.grid_index[j];

				float lambda_j = b_lambdas[original_index];
				float3 pos2 = b_cell_data.sorted_pos[j];
				float3 vec = pos - pos2;
				float dist = length(vec);

				if (dist <= 2.f * params.particle_radius)
				{
					float3 gradient = sph_kernel_Poly6_W_Gradient_CUDA(vec, dist, params.effective_radius);

					float scorr = -0.1f;
					float x = sph_kernel_Poly6_W_CUDA(dist, params.effective_radius) /
						sph_kernel_Poly6_W_CUDA(0.3f * params.effective_radius, params.effective_radius);
					x = pow(x, 4);
					scorr = scorr * x * dt * dt;

					//printf("scorr: %f\n", scorr);

					float3 res = //(1.f / params.rest_density) *
						(lambda_i + lambda_j) *// +scorr)*
						gradient;

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
	float rho = mass[originalIndex] * sph_kernel_Poly6_W_CUDA(0, params.effective_radius);

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
				rho += pbf_density_boundary(
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
	float rho = params.rest_density * b_volume[originalIndex] * sph_kernel_Poly6_W_CUDA(0, params.effective_radius);

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
				rho += pbf_boundary_density(
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
void compute_position_correction(
	float*	lambda,						// output: computed density
	float*	b_lambda,					// input: lambdas in boundary particles
	//float3* sorted_pos,					// input: sorted mass
	//float3* new_pos,					// output: new_pos
	float3* correction,					// output: accumulated correction
	//uint*	gridParticleIndex,			// input: sorted particle indices
	//uint*	cell_start,
	//uint*	cell_end,
	// boundary
	
	CellData cell_data,
	CellData b_cell_data,
	
	uint	numParticles,
	float	dt
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	uint originalIndex = cell_data.grid_index[index];

	// read particle data from sorted arrays
	float3 pos = cell_data.sorted_pos[index];

	// initial density
	float lambda_i = lambda[originalIndex];

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	float3 corr = make_float3(0, 0, 0);


	// traverse 27 neighbors
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				corr += pbf_correction(
					neighbor_pos, index,
					pos, lambda_i, lambda, 
					cell_data,
					dt
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
	correction[originalIndex] += corr;
}

__global__
void apply_correction(
	float3* new_pos,
	float3* predict_pos,
	float3* correction,
	CellData cell_data,
	uint numParticles
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	//predict_pos[index] = new_pos[index];
	
	uint original_index = cell_data.grid_index[index];

	new_pos[original_index] = cell_data.sorted_pos[index] + params.sor_coeff * correction[original_index];
	predict_pos[original_index] = new_pos[original_index];
	// write back to sorted_pos for next iteration
	//cell_data.sorted_pos[index] = new_pos[original_index];
	correction[original_index] = make_float3(0, 0, 0);
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

	float3 t_pos = new_pos[index];
	float3 t_vel = (t_pos - pos[index]) / dt;
	
	

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
			particles->m_d_positions,
			particles->m_d_velocity,
			particles->m_d_force,
			particles->m_d_massInv,
			particles->m_d_predict_positions,
			particles->m_d_new_positions,
			deltaTime,
			numParticles
			);
	}
	else
	{
		integrate_pbd_d << <num_blocks, num_threads >> > (
			particles->m_d_positions,
			particles->m_d_velocity,
			particles->m_d_force,
			particles->m_d_massInv,
			particles->m_d_predict_positions,
			particles->m_d_new_positions,
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
	thrust::sort_by_key(
		thrust::device_ptr<uint>(grid_hash),
		thrust::device_ptr<uint>(grid_hash + numParticles), 
		thrust::device_ptr<uint>(grid_index)
	);
}

void solve_sph_fluid(
	ParticleSet*	sph_particles,
	CellData		sph_cell_data,
	uint			numParticles,
	ParticleSet*	boundary_particles,
	CellData		b_cell_data,
	uint			b_num_particles,
	float			dt,
	int				iterations
)
{
	//std::chrono::steady_clock::time_point t1, t2, t3, t4, t5;
	uint num_threads, num_blocks;
	compute_grid_size(numParticles, MAX_THREAD_NUM, num_blocks, num_threads);

	uint b_num_threads, b_num_blocks;
	compute_grid_size(b_num_particles, MAX_THREAD_NUM, b_num_blocks, b_num_threads);

	for (int i = 0; i < iterations; ++i)
	{
		if (i != 0)
		{
			calculate_hash(
				sph_cell_data,
				sph_particles->m_d_predict_positions,
				numParticles
				);
			sort_particles(
				sph_cell_data,
				numParticles
				);
			reorder_data(
				sph_cell_data,
				//particles->m_d_positions,
				sph_particles->m_d_predict_positions,
				numParticles,
				(64 * 64 * 64)
				);
		}

		// CUDA SPH Kernel
		// compute density
		//t1 = std::chrono::high_resolution_clock::now();
		compute_density_d << <num_blocks, num_threads >> > (
			sph_particles->m_d_density, 
			sph_particles->m_d_mass, 
			sph_particles->m_d_C,
			boundary_particles->m_d_volume,
			sph_cell_data,
			b_cell_data,
			numParticles
			);
		getLastCudaError("Kernel execution failed: compute_density_d ");
		// compute density contributed by boundary particles
		compute_boundary_density_d << <b_num_blocks, b_num_threads >> > (
			sph_particles->m_d_mass,
			boundary_particles->m_d_mass,
			boundary_particles->m_d_volume,
			boundary_particles->m_d_C,
			boundary_particles->m_d_density,
			sph_cell_data,
			b_cell_data,
			b_num_particles
			);
		// compute density of bounary particles
		getLastCudaError("Kernel execution failed: compute_density_boundary_d ");
		//t2 = std::chrono::high_resolution_clock::now();
		// compute lambda
 		compute_lambdas_d << <num_blocks, num_threads >> > (
			sph_particles->m_d_lambda,
			sph_particles->m_d_C,
			sph_particles->m_d_mass,
			boundary_particles->m_d_volume,
			sph_cell_data,
			b_cell_data,
			numParticles
			);
		getLastCudaError("Kernel execution failed: compute_lambdas_d ");
		compute_boundary_lambdas_d << <b_num_blocks, b_num_threads >> > (
			boundary_particles->m_d_lambda,
			boundary_particles->m_d_volume,
			boundary_particles->m_d_positions,
			boundary_particles->m_d_C,
			boundary_particles->m_d_mass,
			b_cell_data,
			sph_cell_data,
			b_num_particles
		);
		getLastCudaError("Kernel execution failed: compute_boundary_lambdas_d ");
		//t3 = std::chrono::high_resolution_clock::now();
		// compute new position
		compute_position_correction << <num_blocks, num_threads >> > (
			sph_particles->m_d_lambda,
			boundary_particles->m_d_lambda,
			sph_particles->m_d_correction,
			sph_cell_data,
			b_cell_data,
			numParticles,
			dt
		);
		getLastCudaError("Kernel execution failed: compute_position_correction ");
		// correct this iteration
		apply_correction << <num_blocks, num_threads >> > (
			sph_particles->m_d_new_positions, 
			sph_particles->m_d_predict_positions, 
			sph_particles->m_d_correction,
			sph_cell_data,
			numParticles
		);
		getLastCudaError("Kernel execution failed: apply_correction ");

		//t4 = std::chrono::high_resolution_clock::now();
	}
	// finalize correction
	finalize_correction << <num_blocks, num_threads >> > (
		sph_particles->m_d_positions, 
		sph_particles->m_d_new_positions, 
		sph_particles->m_d_predict_positions, 
		sph_particles->m_d_velocity,
		numParticles, 
		dt
	);
	getLastCudaError("Kernel execution failed: finalize_correction ");
	/*
	t5 = std::chrono::high_resolution_clock::now();
	{
		ImGui::Begin("CUDA Performance");
		ImGui::Text("Density:     %.5lf (ms)", (t2 - t1).count() / 1000000.0f);
		ImGui::Text("Lambda:      %.5lf (ms)", (t3 - t2).count() / 1000000.0f);
		ImGui::Text("Correction:  %.5lf (ms)", (t4 - t3).count() / 1000000.0f);
		ImGui::Text("Finalize:    %.5lf (ms)", (t5 - t4).count() / 1000000.0f);
		ImGui::End();
	}
	*/
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
				if (dist <= 2.f * params.particle_radius)
				{
					// Non-penetration correction
					const float w1 = invMass[original_index_j];

					float w_sum = w0 + w1;
					C = dist - 2.f * params.particle_radius;
					
					// normalize v + 0.000001f for vanish problem
					float3 n = v / dist + params.pbd_epsilon;// +0.000001f);

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
			if (dist <= 2.f * params.particle_radius)
			{
				// Non-penetration correction
				const float w1 = b_invMass[original_index_j];

				float w_sum = w0 + w1;
				C = dist - 2.f * params.particle_radius;

				// normalize v + 0.000001f to prevent not becoming infinite
				float3 n = v / (dist) + params.pbd_epsilon;// +0.000001f);

				correction_j = -w0 * (1.f / w_sum) * C * n;
			}

			correction += correction_j;
		}

		//printf("Num neighbors: %u\n", end_index - start_index);
	}
	return correction;
}

__global__
void compute_distance_correction(
	float3*		correction,		// output: corrected pos
	float*		invMass,		// input: mass
	float*		b_invMass,
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
				corr += pbd_distance_correction(
					neighbor_pos, index,
					pos, w0,
					invMass,
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
				corr += pbd_distance_correction_boundary(
					neighbor_pos, index,
					pos, w0,
					b_invMass,
					b_cell_data
				);
			}
		}
	}

	correction[original_index] += corr;
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
					float3 n = v / (dist);// +0.000001f);

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
						float coeff = min(params.kinematic_friction * penetration / len, 1.f);
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

		//printf("Num neighbors: %u\n", end_index - start_index);
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
				float3 n = v / (dist);// +0.000001f);

				float penetration = 2.f * params.particle_radius - dist;
				float3 dx = (predict_pos0 - original_pos0);
				float3 dx_t = dx - (dot(dx, n) * n);

				//printf("dx: %f, %f, %f\n", dx.x, dx.y, dx.z);


				float threshold = params.static_friction * penetration;
				float len = length(dx_t);

				// if exceed threshold use kinematic friction model
				if (length(dx_t) > threshold)
				{
					float coeff = min(params.kinematic_friction * penetration / len, 1.f);
					dx_t = coeff * dx_t;
				}

				dx_t = -(w0 / w_sum) * dx_t;
				correction_j += dx_t;
			}
			result += correction_j;
		}

		//printf("Num neighbors: %u\n", end_index - start_index);
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
	
	 
	//printf("corr: %f %f %f\n", corr.x, corr.y, corr.z);
	

	//corr = 0.5f * (correction_i + corr);
	correction[original_index] = corr;
	//new_pos[original_index] = pos + corr;
}

void solve_pbd_dem(
	ParticleSet* dem_particles,
	ParticleSet* boundary_particles,
	CellData cell_data,
	CellData b_cell_data,
	uint numParticles, 
	uint b_numParticles,
	float dt,
	int iteration
)
{
	uint num_threads, num_blocks;
	compute_grid_size(numParticles, MAX_THREAD_NUM, num_blocks, num_threads);
	for (int i = 0; i < iteration; ++i)
	{
		if (i != 0)
		{
			calculate_hash(
				cell_data,
				dem_particles->m_d_predict_positions,
				numParticles
			);
			sort_particles(
				cell_data,
				numParticles
			);
			reorder_data(
				cell_data,
				//particles->m_d_positions,
				dem_particles->m_d_predict_positions,
				numParticles,
				NUM_CELLS
			);
		}
		compute_distance_correction << <num_blocks, num_threads >> > (
			dem_particles->m_d_correction,
			dem_particles->m_d_massInv,
			boundary_particles->m_d_massInv,
			cell_data,
			b_cell_data,
			numParticles
			);
		getLastCudaError("Kernel execution failed: compute_dem_correction ");
		apply_correction << <num_blocks, num_threads >> > (
			dem_particles->m_d_new_positions,
			dem_particles->m_d_predict_positions,
			dem_particles->m_d_correction,
			cell_data,
			numParticles
			);
		getLastCudaError("Kernel execution failed: apply_correction ");

	}

	calculate_hash(
		cell_data,
		dem_particles->m_d_predict_positions,
		numParticles
	);
	sort_particles(
		cell_data,
		numParticles
	);
	reorder_data(
		cell_data,
		//particles->m_d_positions,
		dem_particles->m_d_predict_positions,
		numParticles,
		NUM_CELLS
	);

	compute_friction_correction << <num_blocks, num_threads >> > (
		dem_particles->m_d_correction,
		dem_particles->m_d_new_positions,
		dem_particles->m_d_positions,
		dem_particles->m_d_massInv,
		boundary_particles->m_d_massInv,
		cell_data,
		b_cell_data,
		numParticles
		);

	getLastCudaError("Kernel execution failed: compute_friction_correction ");
	apply_correction << <num_blocks, num_threads >> > (
		dem_particles->m_d_new_positions,
		dem_particles->m_d_predict_positions,
		dem_particles->m_d_correction,
		cell_data,
		numParticles
		);
	getLastCudaError("Kernel execution failed: apply_correction ");


	// finalize correction
	finalize_correction << <num_blocks, num_threads >> > (
		dem_particles->m_d_positions,
		dem_particles->m_d_new_positions,
		dem_particles->m_d_predict_positions,
		dem_particles->m_d_velocity,
		numParticles,
		dt
		);
	getLastCudaError("Kernel execution failed: finalize_correction ");
}

/* 
 * Compute corrections of DEM particles contributed by SPH particles 
 * Treat DEM particles as SPH particles
 */
void solve_dem_sph(
	ParticleSet* dem_particles, 
	ParticleSet* sph_particles, 
	CellData sph_cell_data, 
	CellData dem_cell_data, 
	uint num_dem_particles, 
	uint num_sph_particles, 
	float dt
)
{
	uint num_threads, num_blocks;
	compute_grid_size(num_dem_particles, MAX_THREAD_NUM, num_blocks, num_threads);

	// compute density (treat all of them as sph particles)
	// compute dem_sph_density
	// 

}

__global__
void compute_sph_dem_distance_correction(
	float3* correction,		// output: corrected pos
	float* invMass,		// input: mass
	float* dem_invMass,
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
				corr += pbd_distance_correction_boundary(
					neighbor_pos, index,
					pos, w0,
					dem_invMass,
					dem_cell_data
					);
			}
		}
	}

	correction[original_index] += corr;
}

__global__
void compute_sph_sph_distance_correction(
	float3* correction,		// output: corrected pos
	float* invMass,		// input: mass
	CellData	sph_cell_data,		// input: cell data of dem particles
	uint		numParticles	// input: number of sph particles
	)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

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
				corr += pbd_distance_correction(
					neighbor_pos, index,
					pos, w0,
					invMass,
					sph_cell_data
					);
			}
		}
	}

	correction[original_index] += corr;
}



/*
 * Compute corrections of SPH particles contributed by DEM particles
 * Treat SPH particles as DEM particles
 */
void solve_sph_dem(
	ParticleSet* sph_particles, 
	ParticleSet* dem_particles, 
	CellData sph_cell_data, 
	CellData dem_cell_data, 
	uint num_sph_particles, 
	uint num_dem_particles, 
	float dt
)
{
	uint num_threads, num_blocks;
	compute_grid_size(num_sph_particles, MAX_THREAD_NUM, num_blocks, num_threads);
	
	compute_sph_dem_distance_correction << <num_blocks, num_threads >> > (
		sph_particles->m_d_correction,
		sph_particles->m_d_massInv,
		dem_particles->m_d_massInv,
		sph_cell_data,
		dem_cell_data,
		num_sph_particles
		);
	getLastCudaError("Kernel execution failed: compute_sph_dem_distance_correction ");
	
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
	float rho = mass[originalIndex] * sph_kernel_Poly6_W_CUDA(0, params.effective_radius);

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

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_gridPos = gridPos + make_int3(x, y, z);
				rho += pbf_boundary_density(
					// fluid
					neighbor_gridPos,
					pos,
					// boundary
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
				rho += pbf_density_boundary(
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
	float rho = dem_mass[originalIndex] * sph_kernel_Poly6_W_CUDA(0, params.effective_radius);

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
				rho += pbf_boundary_density(
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
				rho += pbf_density_boundary(
					// fluid
					neighbor_gridPos,
					pos,
					// boundary
					b_vol,
					b_cell_data
				);
			}
		}

		// Update density of fluid particle
		dem_density[originalIndex] = rho;
		// **repeated code**
		// Recompute constraint value of fluid particle
		if ((dem_density[originalIndex] / params.rest_density) - 1.f > 0)
			dem_C[originalIndex] = (dem_density[originalIndex] / params.rest_density) - 1.f;
		else
			dem_C[originalIndex] = 0.f;

	}	
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
	float rho = params.rest_density * b_volume[originalIndex] * sph_kernel_Poly6_W_CUDA(0, params.effective_radius);

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
				rho += pbf_boundary_density(
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
				rho += pbf_boundary_density(
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
	// **repeated code**
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
		sph_particles->m_d_density,
		sph_particles->m_d_mass,
		sph_particles->m_d_C,
		dem_particles->m_d_mass,
		boundary_particles->m_d_volume,
		sph_cell_data,
		dem_cell_data,
		b_cell_data,
		sph_num_particles
	);

	// sph-dem density
	compute_snow_dem_sph_density_d <<<dem_num_blocks, dem_num_threads>>>(
		sph_particles->m_d_mass,
		dem_particles->m_d_mass,
		dem_particles->m_d_C,
		dem_particles->m_d_density,
		boundary_particles->m_d_volume,
		sph_cell_data,
		dem_cell_data,
		b_cell_data,
		dem_num_particles
	);

	// boundary density
	compute_snow_boundary_density_d << <b_num_blocks, b_num_threads >> > (
		sph_particles->m_d_mass,
		dem_particles->m_d_mass,
		boundary_particles->m_d_mass,
		boundary_particles->m_d_volume,
		boundary_particles->m_d_C,
		boundary_particles->m_d_density,
		sph_cell_data,
		dem_cell_data,
		b_cell_data,
		b_num_particles
		);
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
		sph_particles->m_d_lambda,
		sph_particles->m_d_C,
		sph_particles->m_d_mass,
		boundary_particles->m_d_volume,
		sph_cell_data,
		dem_cell_data,
		b_cell_data,
		sph_num_particles
	);

	// sph-dem lambdas (ignored if not using dem density)
	compute_dem_sph_lambdas_d << <dem_num_blocks, dem_num_threads >> > (
		dem_particles->m_d_lambda,
		dem_particles->m_d_positions,
		dem_particles->m_d_C,
		dem_particles->m_d_mass,
		boundary_particles->m_d_volume,
		sph_cell_data,
		dem_cell_data,
		b_cell_data,
		dem_num_particles
	);

	// boundary lambdas
	compute_snow_boundary_lambdas_d << <b_num_blocks, b_num_threads >> > (
		boundary_particles->m_d_lambda,
		boundary_particles->m_d_volume,
		boundary_particles->m_d_positions,
		boundary_particles->m_d_C,
		boundary_particles->m_d_mass,
		sph_cell_data,
		dem_cell_data,
		b_cell_data,
		b_num_particles
		);
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
void compute_snow_sph_position_correction(
	float* sph_lambda,						// output: computed density
	float* dem_lambda,
 	float* b_lambda,					// input: lambdas in boundary particles
	float3* sph_correction,					// output: accumulated correction

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

	uint originalIndex = sph_cell_data.grid_index[index];

	// read particle data from sorted arrays
	float3 pos = sph_cell_data.sorted_pos[index];

	// initial density
	float lambda_i = sph_lambda[originalIndex];

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
					pos, lambda_i, sph_lambda,
					sph_cell_data,
					dt
				);
			}
		}
	}

	
	//sph-dem is corrected by non-penetration constraint
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
	sph_correction[originalIndex] += corr;
}

__global__
void compute_dem_sph_position_correction(
	float*	 sph_lambda,					// input: lambdas in boundary particles
	float*	 dem_lambda,					// output: computed density
	float*	 b_lambda,
	float3*	 dem_correction,				// output: accumulated correction
	CellData sph_cell_data,
	CellData dem_cell_data,
	CellData b_cell_data,
	uint	 dem_num_particles,
	float	 dt
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= dem_num_particles) return;

	uint originalIndex = dem_cell_data.grid_index[index];

	// read particle data from sorted arrays
	float3 pos = dem_cell_data.sorted_pos[index];

	// initial density
	float lambda_i = dem_lambda[originalIndex];

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	float3 corr = make_float3(0, 0, 0);

	bool proceed = false;
	
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				proceed |= sphere_cd_coupling(
					neighbor_pos, 
					pos, 
					sph_cell_data
				);
			}
		}
	}

	// test if collided with sph particles (affected by fluid movement)
	if (proceed)
	{
		// dem-dem
		for (int z = -1; z <= 1; z++)
		{
			for (int y = -1; y <= 1; y++)
			{
				for (int x = -1; x <= 1; x++)
				{
					int3 neighbor_pos = gridPos + make_int3(x, y, z);
					corr += pbf_correction(
						neighbor_pos, index,
						pos, lambda_i, dem_lambda,
						dem_cell_data,
						dt
					);
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
					corr += pbf_correction_boundary(
						neighbor_pos,
						index,
						pos,
						lambda_i,
						sph_cell_data,
						sph_lambda,
						dt
					);
				}
			}
		}
		
	}

	// dem-boundary is corrected by non penetration constraint
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
	dem_correction[originalIndex] += corr;
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
	uint num_threads, num_blocks;
	uint dem_num_threads, dem_num_blocks;
	compute_grid_size(sph_num_particles, MAX_THREAD_NUM, num_blocks, num_threads);
	compute_grid_size(dem_num_particles, MAX_THREAD_NUM, dem_num_blocks, dem_num_threads);

	// sph-sph correction
	// sph-b correction
	compute_snow_sph_position_correction << <num_blocks, num_threads >> > (
		sph_particles->m_d_lambda,
		dem_particles->m_d_lambda,
		boundary_particles->m_d_lambda,
		sph_particles->m_d_correction,
		sph_cell_data,
		dem_cell_data,
		b_cell_data,
		sph_num_particles,
		dt
	);
	 
	/*
	// dem-sph correction (treat dem particles as sph particles) (affected by fluid movment)
	compute_dem_sph_position_correction << <dem_num_blocks, dem_num_threads >> > (
		sph_particles->m_d_lambda,
		dem_particles->m_d_lambda,
		boundary_particles->m_d_lambda,
		dem_particles->m_d_correction,
		sph_cell_data,
		dem_cell_data,
		b_cell_data,
		dem_num_particles,
		dt
	);
	*/
	

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
	
	// sph-dem distance correction (treat sph particles as dem particles)
	compute_sph_dem_distance_correction <<<sph_num_blocks, sph_num_threads>>>(
		sph_particles->m_d_correction,
		sph_particles->m_d_massInv,
		dem_particles->m_d_massInv,
		sph_cell_data,
		dem_cell_data,
		sph_num_particles
	);
	getLastCudaError("Kernel execution failed: compute_sph_dem_distance_correction ");
	
	if (correct_dem)
	{
		//dem-sph distance correction (reversed parameters for the same function)
		compute_sph_dem_distance_correction << <dem_num_blocks, dem_num_threads >> > (
			dem_particles->m_d_correction,
			dem_particles->m_d_massInv,
			sph_particles->m_d_massInv,
			dem_cell_data,
			sph_cell_data,
			dem_num_particles
			);
		getLastCudaError("Kernel execution failed: compute_sph_dem_distance_correction ");
	}

	if (sph_sph_correction)
	{
		compute_sph_sph_distance_correction << <sph_num_blocks, sph_num_threads >> > (
			sph_particles->m_d_correction,
			sph_particles->m_d_massInv,
			sph_cell_data,
			sph_num_particles
			);
	}

	// dem-dem distance correction
	// dem-boundary distance correction
	compute_distance_correction << <dem_num_blocks, dem_num_threads >> > (
		dem_particles->m_d_correction,
		dem_particles->m_d_massInv,
		boundary_particles->m_d_massInv,
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

				res += (mass[original_index_j]/density[original_index_j]) * v_i_j * sph_kernel_Poly6_W_CUDA(dist, params.effective_radius);
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

			res += (other_mass[original_index_j] / other_density[original_index_j]) * v_i_j * sph_kernel_Poly6_W_CUDA(dist, params.effective_radius);
	}

	}
	return res;
}

__global__ 
void xsph_viscosity(
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
	 
	corr *= params.viscosity;

	cg::sync(cta);

	sph_vel[original_index] += corr;

}

void apply_XSPH_viscosity(
	ParticleSet* sph_particles,
	ParticleSet* dem_particles,
	CellData sph_cell_data,
	CellData dem_cell_data,
	uint sph_num_particles
)
{
	uint sph_num_threads, sph_num_blocks;
	compute_grid_size(sph_num_particles, MAX_THREAD_NUM, sph_num_blocks, sph_num_threads);

	xsph_viscosity <<<sph_num_blocks, sph_num_threads>>> (
		sph_particles->m_d_velocity,
		sph_particles->m_d_mass,
		sph_particles->m_d_density,
		dem_particles->m_d_velocity,
		dem_particles->m_d_mass,
		dem_particles->m_d_density,
		sph_cell_data,
		dem_cell_data,
		sph_num_particles
	);

	getLastCudaError("Kernel execution failed : apply_correction");

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


inline __device__
float propagate_wetness_cell(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float*	wetness,
	float*	neighbor_count,
	CellData cell_data,
	float dt,
	bool coupling=false
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_data.cell_start[grid_hash];
	float res = 0;

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			if (coupling || (j != index))
			{
				uint original_index_j = cell_data.grid_index[j];

				float3 pos2 = cell_data.sorted_pos[j];
				//float3 vec = pos - pos2;
				float dist = length(pos - pos2);
				
				if (dist <= 2.f * params.particle_radius)
				{
					float dw = wetness[original_index_j] - params.wetness_threshold;
					res += params.k_p * dt * dw / neighbor_count[original_index_j];
				}
				
			}
			
		}

	}
	return res;
}

__global__
void propagate_wetness_d(
	float* sph_wetness,
	float* dem_wetness,
	float* sph_neighbor_count,
	float* dem_neighbor_count,
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
	float w_i = dem_wetness[original_index];

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	// lock critical section
	cg::thread_block cta = cg::this_thread_block();
	float change = 0.f;

	// wet propagation between neighbors (dem->dem)
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				change += propagate_wetness_cell(
					gridPos,
					index,
					pos,
					dem_wetness,
					dem_neighbor_count,
					dem_cell_data, dt
					);
			}
		}
	}

	// (sph->dem)
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				change += propagate_wetness_cell(
					gridPos,
					index,
					pos,
					sph_wetness,
					sph_neighbor_count,
					sph_cell_data, dt
					);
			}
		}
	}

	// wait & sync
	cg::sync(cta);

	if (w_i + change >= params.wetness_max)
		dem_wetness[original_index] = params.wetness_max;
	else
		dem_wetness[original_index] += change;
	
}

inline __device__
float3 wetness_force_cell(
	int3    grid_pos,
	uint    index,
	float3  pos0,
	float3  v_i,
	float   w_i,
	float3* velocity,
	float*  wetness,
	CellData cell_data,
	float dt,
	bool  coupling = false
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
			if (coupling || (j != index))
			{
				uint original_index_j = cell_data.grid_index[j];

				float3 pos1 = cell_data.sorted_pos[j];
				//float3 vec = pos - pos2;
				const float3 dx = pos1 - pos0;
				const float3 dv = velocity[original_index_j] - v_i;

				const float w_j = wetness[original_index_j];
				float dist = length(dx);
				if (dist < 2.f * params.particle_radius && dot(dv, dx) > 0)
				{
					res += max(0.f, (params.wetness_threshold - (w_i + w_j) / 2.f) ) * dv;
				}

			}

		}

	}
	return res;
}


__global__
void compute_wetness_correction(
	float3* dem_correction,
	float3* sph_velocity,
	float3* dem_velocity,
	float* dem_mass,
	float* sph_wetness,
	float* dem_wetness,
	float* sph_neighbor_count,
	float* dem_neighbor_count,
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
	float3 v_i = dem_velocity[original_index];
	float w_i = dem_wetness[original_index];
	
	// get address in grid
	int3 gridPos = calcGridPos(pos);

	float change = 0.f;
	float3 force = make_float3(0, 0, 0);

	float3 corr = make_float3(0, 0, 0);

	// wet propagation between neighbors (dem->dem)
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				force += wetness_force_cell(
					gridPos, index, 
					pos, v_i, w_i,
					dem_velocity, dem_wetness,
					dem_cell_data, 
					dt,
					false
					);
			}
		}
	}

	// (sph->dem)
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				force += wetness_force_cell(
					gridPos, index, 
					pos, v_i, w_i,
					sph_velocity, sph_wetness, 
					sph_cell_data, 
					dt,
					true
					);
			}
		}
	}

	// integration
	float3 vel = (force / dem_mass[original_index]) * dt;
	corr = vel * dt;

	dem_correction[original_index] += corr;
}


void propagate_wetness(
	ParticleSet* sph_particles,
	ParticleSet* dem_particles,
	CellData sph_cell_data,
	CellData dem_cell_data,
	uint sph_num_particles,
	uint dem_num_particles,
	float dt
)
{
	uint sph_num_threads, sph_num_blocks;
	compute_grid_size(sph_num_particles, MAX_THREAD_NUM, sph_num_blocks, sph_num_threads);
	uint dem_num_threads, dem_num_blocks;
	compute_grid_size(dem_num_particles, MAX_THREAD_NUM, dem_num_blocks, dem_num_threads);

	counting_neighbors_d <<<dem_num_blocks, dem_num_threads>>>(
		dem_particles->m_d_num_neighbors, 
		dem_cell_data, sph_cell_data, 
		dem_num_particles
	);
	counting_neighbors_d <<<sph_num_blocks, sph_num_threads>>>(
		sph_particles->m_d_num_neighbors, 
		sph_cell_data, dem_cell_data, 
		sph_num_particles
	);

	// propagation only exist between dem->dem sph->dem (sph are considered as max in any time)
	propagate_wetness_d << <dem_num_blocks, dem_num_threads >> > (
		sph_particles->m_d_wetness,
		dem_particles->m_d_wetness,
		sph_particles->m_d_num_neighbors,
		dem_particles->m_d_num_neighbors,
		sph_cell_data,
		dem_cell_data,
		dem_num_particles,
		dt
	);

	//correct_with_wetness
	compute_wetness_correction << <dem_num_blocks, dem_num_threads >> > (
		dem_particles->m_d_correction,
		sph_particles->m_d_velocity,
		dem_particles->m_d_velocity,
		dem_particles->m_d_mass,
		sph_particles->m_d_wetness,
		dem_particles->m_d_wetness,
		sph_particles->m_d_num_neighbors,
		dem_particles->m_d_num_neighbors,
		sph_cell_data,
		dem_cell_data,
		dem_num_particles,
		dt
	);
}

void snow_simulation(
	ParticleSet* sph_particles,
	ParticleSet* dem_particles, 
	ParticleSet* boundary_particles, 
	CellData sph_cell_data, 
	CellData dem_cell_data, 
	CellData b_cell_data, 
	uint sph_num_particles, 
	uint dem_num_particles, 
	uint b_num_particles, 
	float dt,
	int iterations,
	bool correct_dem,
	bool sph_sph_correction,
	bool compute_wetness
)
{
	uint sph_num_threads, sph_num_blocks;
	compute_grid_size(sph_num_particles, MAX_THREAD_NUM, sph_num_blocks, sph_num_threads);
	uint dem_num_threads, dem_num_blocks;
	compute_grid_size(dem_num_particles, MAX_THREAD_NUM, dem_num_blocks, dem_num_threads);

	for (int i = 0; i < iterations; ++i)
	{
		//Search again for stability
		/*
		if (i != 0)
		{
			sort_and_reorder(
				dem_particles->m_d_predict_positions,
				dem_cell_data,
				dem_num_particles
			);

			sort_and_reorder(
				sph_particles->m_d_predict_positions,
				sph_cell_data,
				sph_num_particles
			);
		}
		*/
		compute_snow_pbf_density(
			sph_particles,
			dem_particles,
			boundary_particles,
			sph_cell_data,
			dem_cell_data,
			b_cell_data,
			sph_num_particles,
			dem_num_particles,
			b_num_particles
			);

		compute_snow_pbf_lambdas(
			sph_particles,
			dem_particles,
			boundary_particles,
			sph_cell_data,
			dem_cell_data,
			b_cell_data,
			sph_num_particles,
			dem_num_particles,
			b_num_particles
			); 

		compute_snow_pbf_correction(
			sph_particles,
			dem_particles,
			boundary_particles,
			sph_cell_data,
			dem_cell_data,
			b_cell_data,
			sph_num_particles,
			dem_num_particles,
			b_num_particles,
			dt
			);

		compute_snow_distance_correction(
			sph_particles,
			dem_particles,
			boundary_particles,
			sph_cell_data,
			dem_cell_data,
			b_cell_data,
			sph_num_particles,
			dem_num_particles,
			b_num_particles,
			correct_dem,
			sph_sph_correction
			);

		apply_correction << <sph_num_blocks, sph_num_threads >> > (
			sph_particles->m_d_new_positions,
			sph_particles->m_d_predict_positions,
			sph_particles->m_d_correction,
			sph_cell_data,
			sph_num_particles
		);
		getLastCudaError("Kernel execution failed: apply_correction ");

		apply_correction << <dem_num_blocks, dem_num_threads >> > (
			dem_particles->m_d_new_positions,
			dem_particles->m_d_predict_positions,
			dem_particles->m_d_correction,
			dem_cell_data,
			dem_num_particles
			);
		getLastCudaError("Kernel execution failed: apply_correction ");

	}

	sort_and_reorder(
		dem_particles->m_d_predict_positions,
		dem_cell_data,
		dem_num_particles
	);

	sort_and_reorder(
		sph_particles->m_d_predict_positions,
		sph_cell_data,
		sph_num_particles
	);

	compute_friction_correction << <dem_num_blocks, dem_num_threads >> > (
		dem_particles->m_d_correction,
		dem_particles->m_d_new_positions,
		dem_particles->m_d_positions,
		dem_particles->m_d_massInv,
		boundary_particles->m_d_massInv,
		dem_cell_data,
		b_cell_data,
		dem_num_particles
		);
	getLastCudaError("Kernel execution failed: compute_friction_correction ");

	apply_correction << <dem_num_blocks, dem_num_threads >> > (
		dem_particles->m_d_new_positions,
		dem_particles->m_d_predict_positions,
		dem_particles->m_d_correction,
		dem_cell_data,
		dem_num_particles
		);
	getLastCudaError("Kernel execution failed: apply_correction ");


	sort_and_reorder(
		dem_particles->m_d_predict_positions,
		dem_cell_data,
		dem_num_particles
	);
	sort_and_reorder(
		sph_particles->m_d_predict_positions,
		sph_cell_data,
		sph_num_particles
		);
	if (compute_wetness)
	{
		// force-based wetness correction for dem particles
		propagate_wetness(
			sph_particles,
			dem_particles,
			sph_cell_data,
			dem_cell_data,
			sph_num_particles,
			dem_num_particles,
			dt
			);

		apply_correction << <dem_num_blocks, dem_num_threads >> > (
			dem_particles->m_d_new_positions,
			dem_particles->m_d_predict_positions,
			dem_particles->m_d_correction,
			dem_cell_data,
			dem_num_particles
			);
		getLastCudaError("Kernel execution failed: apply_correction ");
	}

	// finialize corrections and compute velocity for next integration 
	finalize_correction << <sph_num_blocks, sph_num_threads >> > (
		sph_particles->m_d_positions,
		sph_particles->m_d_new_positions,
		sph_particles->m_d_predict_positions,
		sph_particles->m_d_velocity,
		sph_num_particles,
		dt
	);
	getLastCudaError("Kernel execution failed: finalize_correction ");

	finalize_correction << <dem_num_blocks, dem_num_threads >> > (
		dem_particles->m_d_positions,
		dem_particles->m_d_new_positions,
		dem_particles->m_d_predict_positions,
		dem_particles->m_d_velocity,
		dem_num_particles,
		dt
		);
	getLastCudaError("Kernel execution failed: finalize_correction ");

	
	apply_XSPH_viscosity(
		sph_particles, 
		dem_particles,
		sph_cell_data,
		dem_cell_data,
		sph_num_particles
	);
}
