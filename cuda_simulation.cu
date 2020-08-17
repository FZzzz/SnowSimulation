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
#include "sph_kernel.cuh"
#include <chrono>
#include "imgui/imgui.h"

namespace cg = cooperative_groups;

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

// collide two spheres using DEM method
inline __device__
float3 collideSpheres(
	float3 posA, float3 posB,
	float3 velA, float3 velB,
	float radiusA, float radiusB,
	float attraction)
{
	// calculate relative position
	float3 relPos = posB - posA;

	float dist = length(relPos);
	float collideDist = radiusA + radiusB;

	float3 force = make_float3(0.0f);

	//printf("dist: %f\ncollideDist: %f", dist, collideDist);

	if (dist < collideDist)
	{
		float3 norm = relPos / (dist+0.00001f);

		// relative velocity
		float3 relVel = velB - velA;

		// relative tangential velocity
		float3 tanVel = relVel - (dot(relVel, norm) * norm);

		// spring force
		force = -params.spring * (collideDist - dist) * norm;
		// dashpot (damping) force
		force += params.damping * relVel;
		// tangential shear force
		force += params.shear * tanVel;
		// attraction
		force += attraction * relPos;

		//printf("%f %f %f\n", force.x, force.y, force.z);
	}

	return force;
}

inline __device__
float3 collideCell(
	int3    gridPos,
	uint    index,
	float3  pos,
	float3  vel,
	float3* oldPos,
	float3* oldVel,
	uint* cellStart,
	uint* cellEnd)
{
	uint gridHash = calcGridHash(gridPos);

	// get start of bucket for this cell
	uint startIndex = cellStart[gridHash];

	float3 force = make_float3(0.0f);

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = cellEnd[gridHash];

		for (uint j = startIndex; j < endIndex; j++)
		{
			if (j != index)                // check not colliding with self
			{
				float3 pos2 = oldPos[j];
				float3 vel2 = oldVel[j];

				// collide two spheres
				force += collideSpheres(
					pos, pos2,
					vel, vel2,
					params.particle_radius, params.particle_radius,
					params.attraction);
			}
		}
	}

	return force;
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

	uint start_index = data.cellStart[grid_hash];

	float rho = 0.f;

	if (start_index != 0xffffffff)
	{
		uint end_index = data.cellEnd[grid_hash];

		for (uint j = start_index; j < end_index; ++j)
		{
			if (j != index)
			{
				uint original_index = data.grid_index[j];
				float3 pos2 = data.sorted_pos[j];
				float3 vec = pos1 - pos2;
				float dist = length(vec);
				rho += mass[original_index] * Poly6_W_CUDA(dist, params.effective_radius);
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
			cell_data.cellStart[hash] = index;

			if (index > 0)
				cell_data.cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1)
		{
			cell_data.cellEnd[hash] = index + 1;
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
				cell_data.cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1)
		{
			cell_data.cellEnd[hash] = index + 1;
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
	float rho = mass[originalIndex] * Poly6_W_CUDA(0, params.effective_radius);

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

void compute_grid_size(uint n, uint block_size, uint& num_blocks, uint& num_threads)
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
	uint numThreads, numBlocks;
	compute_grid_size(numParticles, MAX_THREAD_NUM, numBlocks, numThreads);

	// set all cells to empty
	checkCudaErrors(cudaMemset(cell_data.cellStart, 0xffffffff, numCells * sizeof(uint)));

	uint smemSize = sizeof(uint) * (numThreads + 1);
	reorderDataAndFindCellStartD << < numBlocks, numThreads, smemSize >> > (
		cell_data,
		oldPos,
		numParticles);
	getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

}
/*
void reorderData_boundary(
	CellData cell_data, 
	float3* oldPos, 
	uint numParticles, 
	uint numCells)
{
	uint numThreads, numBlocks;
	compute_grid_size(numParticles, MAX_THREAD_NUM, numBlocks, numThreads);

	// set all cells to empty
	checkCudaErrors(cudaMemset(cell_data.cellStart, 0xffffffff, numCells * sizeof(uint)));

	uint smemSize = sizeof(uint) * (numThreads + 1);
	reorderData_boundary_D << < numBlocks, numThreads, smemSize >> > (
		cell_data,
		oldPos,
		numParticles);
	getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

}
*/

void compute_boundary_volume(CellData data, float* mass, float* volume, uint numParticles)
{
	uint numThreads, numBlocks;
	compute_grid_size(numParticles, MAX_THREAD_NUM, numBlocks, numThreads);

	compute_boundary_volume_d << <numBlocks, numThreads >> > (
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
	
	if (t_pos.y <= 0.f)
	{
		t_pos.y = 0.f;
		t_vel.y = abs(t_vel.y);
		t_vel *= params.boundary_damping;
	}
	
	
	/* Velocity limitation
	if (length(t_vel) > 5.f)
	{
		t_vel = (5.f / length(t_vel)) * t_vel ;
	}
	*/
	
	predict_pos[index] = t_pos;// pos[index] + dt * t_vel;
	vel[index] = t_vel; 
	new_pos[index] = predict_pos[index];


}

// collide a particle against all other particles in a given cell
/* Collision device code */
__global__
void collideD(
	float3* newVel,               // output: new velocity
	float3* oldPos,               // input: sorted positions
	float3* oldVel,               // input: sorted velocities
	uint* gridParticleIndex,      // input: sorted particle indices
	uint* cellStart,
	uint* cellEnd,
	uint  numParticles,
	float dt)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	// read particle data from sorted arrays
	float3 pos = oldPos[index];
	float3 vel = oldVel[index];

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	// examine neighbouring cells
	float3 force = make_float3(0.0f);

	// traverse 27 neighbors
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbor_pos = gridPos + make_int3(x, y, z);
				force += collideCell(neighbor_pos, index, pos, vel, oldPos, oldVel, cellStart, cellEnd);
			}
		}
	}

	// write new velocity back to original unsorted location
	uint originalIndex = gridParticleIndex[index];
	newVel[originalIndex] = vel + force * dt; // + force/mass * dt ?
}


inline __device__
float pbf_density_0(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float3* sorted_pos,
	float*	mass,
	float*	rest_density,
	uint*	cell_start,
	uint*	cell_end,
	uint*	gridParticleIndex
) // type: 0->fluid fluid 1->boundary boundary 
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_start[grid_hash];
	float density = 0.0f;

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			if (j != index)                // check not colliding with self
			{
				uint original_index = gridParticleIndex[j];
				
				float3 pos2 = sorted_pos[j];
				float3 vec = pos - pos2;
				float dist = length(vec);
				float rho = 0.f;

				rho = mass[original_index] * Poly6_W_CUDA(dist, params.effective_radius);

				density += rho;
			}
		}
	}

	return density;
}

inline __device__
float pbf_density_1(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float3* sorted_pos,
	float* mass,
	float* rest_density,
	uint* cell_start,
	uint* cell_end,
	uint* gridParticleIndex,
	float* b_volume = nullptr) // type: 0->fluid fluid 1->boundary boundary 
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_start[grid_hash];
	float density = 0.0f;

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			if (j != index)                // check not colliding with self
			{
				uint original_index = gridParticleIndex[j];

				float3 pos2 = sorted_pos[j];
				float3 vec = pos - pos2;
				float dist = length(vec);
				float rho = 0.f;

				rho = (*rest_density) * b_volume[original_index] * Poly6_W_CUDA(dist, params.effective_radius);

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
	float* rest_density,
	float* volume,
	CellData cell_data
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_data.cellStart[grid_hash];
	float density = 0.0f;

	// if cell of boundary cell data is not empty
	if (start_index != 0xffffffff)
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cellEnd[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{	
			// no need to check collision (boundary cell data is not the same as fluid cell data)
			uint original_index = cell_data.grid_index[j];

			float3 pos2 = cell_data.sorted_pos[j];
			float3 vec = pos1 - pos2;
			float dist = length(vec);

			float rho = (*rest_density) * volume[original_index] * Poly6_W_CUDA(dist, params.effective_radius);

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
	float3*		sorted_pos,
	uint*		cell_start,
	uint*		cell_end,
	uint*		gridParticleIndex
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_start[grid_hash];
	float density = 0.0f;

	// if cell of boundary cell data is not empty
	if (start_index != 0xffffffff)
	{
		// iterate over particles in this cell
		uint end_index = cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			// no need to check collision (boundary cell data is not the same as fluid cell data)
			uint original_index = gridParticleIndex[j];

			float3 pos2 = sorted_pos[j];
			float3 vec = pos1 - pos2;
			float dist = length(vec);

			float rho = mass[original_index] * Poly6_W_CUDA(dist, params.effective_radius);

			density += rho;
		}
	}

	// return contributions of boundary paritcles
	return density;
}

inline __device__
float pbf_lambda_0(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float*	rest_density,
	float*	mass,
	float3* sorted_pos,
	uint*	cell_start,
	uint*	cell_end,
	uint*	gridParticleIndex
)
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
				//float particle_mass = mass[original_index];
				float3 pos2 = sorted_pos[j];
				float3 vec = pos - pos2;
				float dist = length(vec);

				float3 gradientC_j;

				gradientC_j = (1.f / (*rest_density)) *
					Poly6_W_Gradient_CUDA(vec, dist, params.effective_radius);

				float dot_val = dot(gradientC_j, gradientC_j);
				gradientC_sum += dot_val;
			}
		}
	}
	return gradientC_sum;
}

inline __device__
float pbf_lambda_1(
	int3    grid_pos,
	uint    index,
	float3  pos,
	float* rest_density,
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

				gradientC_j = (1.f / (*rest_density)) *
						((*rest_density) * vol / particle_mass) *
						Poly6_W_Gradient_CUDA(vec, dist, params.effective_radius);

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
	float*		rest_density,
	float		particle_mass,
	CellData	cell_data,	// cell data of boundary particle,
	float*		volume
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_data.cellStart[grid_hash];
	float gradientC_sum = 0.f;

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cellEnd[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			uint original_index = cell_data.grid_index[j];
			float vol = volume[original_index];

			float3 pos2 = cell_data.sorted_pos[j];
			float3 vec = pos1 - pos2;
			float dist = length(vec);

			float3 gradientC_j = (1.f / (*rest_density)) * 
				((*rest_density) * vol / particle_mass) *  
				Poly6_W_Gradient_CUDA(vec, dist, params.effective_radius);

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
	float*		rest_density,
	float		particle_mass,
	float		volume,
	// fluid
	float3*		sorted_pos,  
	uint*		cell_start,
	uint*		cell_end,
	uint*		gridParticleIndex
)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_start[grid_hash];
	float gradientC_sum = 0.f;

	// search in fluid cell
	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			float3 pos2 = sorted_pos[j];
			float3 vec = pos1 - pos2;
			float dist = length(vec);

			float3 gradientC_j = (1.f / (*rest_density)) *
				Poly6_W_Gradient_CUDA(vec, dist, params.effective_radius);

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
	float*	rest_density,
	float3* sorted_pos,
	float*	lambda,
	uint*	cell_start,
	uint*	cell_end,
	uint*	gridParticleIndex,
	float	dt)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = cell_start[grid_hash];
	float3 correction = make_float3(0, 0, 0);

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_end[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			if (j != index)                // check not colliding with self
			{
				uint original_index = gridParticleIndex[j];

				float3 pos2 = sorted_pos[j];
				float3 vec = pos - pos2;
				float dist = length(vec);

				float3 gradient = Poly6_W_Gradient_CUDA(vec, dist, params.effective_radius);
				
				float scorr = -0.1f;
				float x = Poly6_W_CUDA(dist, params.effective_radius) / 
					Poly6_W_CUDA(0.3f * params.effective_radius, params.effective_radius);
				x = pow(x, 4);
				scorr = scorr * x * dt * dt * dt;
				
				//printf("scorr: %f\n", scorr);

				float3 res = //(1.f / (*rest_density)) *
					(lambda_i + lambda[original_index] +scorr)*
					gradient;
				
				correction += res;
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
	float*	rest_density,
	// boundary
	CellData b_cell_data,
	float*	b_lambda,
	float	dt)
{
	uint grid_hash = calcGridHash(grid_pos);

	// get start of bucket for this cell
	uint start_index = b_cell_data.cellStart[grid_hash];
	float3 correction = make_float3(0, 0, 0);

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = b_cell_data.cellEnd[grid_hash];

		for (uint j = start_index; j < end_index; j++)
		{
			if (j != index)                // check not colliding with self
			{
				uint original_index = b_cell_data.grid_index[j];

				float lambda_j = b_lambda[original_index];
				float3 pos2 = b_cell_data.sorted_pos[j];
				float3 vec = pos - pos2;
				float dist = length(vec);

				float3 gradient = Poly6_W_Gradient_CUDA(vec, dist, params.effective_radius);

				float scorr = -0.1f;
				float x = Poly6_W_CUDA(dist, params.effective_radius) /
					Poly6_W_CUDA(0.3f * params.effective_radius, params.effective_radius);
				x = pow(x, 4);
				scorr = scorr * x * dt * dt;

				//printf("scorr: %f\n", scorr);

				float3 res = //(1.f / (*rest_density)) *
					(lambda_i + lambda_j) *// +scorr)*
					gradient;

				correction += res;
			}
		}

		//printf("Num neighbors: %u\n", end_index - start_index);
	}
	return correction;
}

__global__
void compute_density_d(
	float*	density,					// output: computed density
	float*	rest_density,				// input: rest density
	float3* sorted_pos,					// input: sorted mass
	float*	mass,						// input: mass
	float*	C,							// input: contraint
	uint*	gridParticleIndex,			// input: sorted particle indices
	uint*	cellStart,
	uint*	cellEnd,
	//boundary
	CellData cell_data,
	float*	b_volume,
	uint	numParticles
	)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	
	if (index >= numParticles) return;

	uint originalIndex = gridParticleIndex[index];

	// read particle data from sorted arrays
	float3 pos = sorted_pos[index];
	
	// initial density
	float rho = mass[originalIndex] * Poly6_W_CUDA(0, params.effective_radius);

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
					pos, sorted_pos, mass, 
					rest_density,
					cellStart, cellEnd, gridParticleIndex
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
					rest_density,
					// boundary
					b_volume,
					cell_data
				);
			}
		}
	}


	// Update date density and constraint value
	density[originalIndex] = rho;
	C[originalIndex] = (rho / (*rest_density)) - 1.f;

	//printf("rho = %f\n", rho);
	//printf("C[%u]: %f\n", originalIndex, C[originalIndex]);

}

__global__
void compute_boundary_density_d(
	// fluid
	float*		rest_density,				// input: rest density
	float3*		sorted_pos,					// input: sorted pos of fluid particle
	float*		mass,						// input: mass of fluid paritcle
	uint*		cellStart,
	uint*		cellEnd,
	uint*		gridParticleIndex,			// input: sorted particle indices (for original_index of fluid particles)
	// boundary
	CellData	b_cell_data,
	float*		b_mass,
	float*		b_volume,
	float*		b_C,
	float*		b_density,					// output: boundary density
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
	float rho = (*rest_density) * b_volume[originalIndex] * Poly6_W_CUDA(0, params.effective_radius);

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
					pos, b_cell_data.sorted_pos,
					b_mass,
					rest_density,
					b_cell_data.cellStart,
					b_cell_data.cellEnd,
					b_cell_data.grid_index,
					b_volume
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
					sorted_pos,
					cellStart,
					cellEnd,
					gridParticleIndex
				);
			}
		}
	}

	// Update density of fluid particle
	b_density[originalIndex] = rho;
	// **repeated code**
	// Recompute constraint value of fluid particle
	b_C[originalIndex] = (b_density[originalIndex] / (*rest_density)) - 1.f;
}

/* fluid - boundary */
__global__
void compute_lambdas_d(
	float*	lambda,						// output: computed density
	float*	rest_density,				// input: rest density
	float3* sorted_pos,					// input: sorted mass
	float*	C,							// input: contraint
	float*  mass,
	uint*	gridParticleIndex,			// input: sorted particle indices
	uint*	cellStart,
	uint*	cellEnd,
	CellData cell_data,
	
	float*	b_volume,
	uint	numParticles
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	uint originalIndex = gridParticleIndex[index];

	// read particle data from sorted arrays
	float3 pos = sorted_pos[index];

	// initial density
	lambda[originalIndex] = -C[originalIndex];

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	float3 gradientC_i = make_float3(0);
		//-(1.f / (*rest_density)) *
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
					pos, rest_density,
					mass, sorted_pos,
					cellStart, cellEnd, 
					gridParticleIndex
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
					pos, rest_density,
					mass[originalIndex],  // paritcle_mass
					cell_data,
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
	float3* sorted_pos,
	uint*	gridParticleIndex,			// input: sorted particle indices
	uint*	cellStart,
	uint*	cellEnd,
	float*	rest_density,
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
	float particle_mass = b_mass[originalIndex];

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	float3 gradientC_i = make_float3(0);
	//-(1.f / (*rest_density)) *
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
					pos, rest_density,
					b_mass,
					b_cell_data.sorted_pos,
					b_cell_data.cellStart, b_cell_data.cellEnd, b_cell_data.grid_index,
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
					pos, rest_density,
					particle_mass,			// paritcle_mass
					b_vol[originalIndex],	// volume
					// fluid
					sorted_pos,
					cellStart,
					cellEnd,
					gridParticleIndex
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
	float*	rest_density,				// input: rest density
	float3* sorted_pos,					// input: sorted mass
	//float3* new_pos,					// output: new_pos
	float3* correction,					// output: accumulated correction
	uint*	gridParticleIndex,			// input: sorted particle indices
	uint*	cellStart,
	uint*	cellEnd,
	// boundary
	CellData b_cell_data,
	float*	b_lambda,
	uint	numParticles,
	float	dt
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	uint originalIndex = gridParticleIndex[index];

	// read particle data from sorted arrays
	float3 pos = sorted_pos[index];

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
					pos, lambda_i, rest_density,
					sorted_pos, lambda,
					cellStart, cellEnd, gridParticleIndex,
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
					rest_density,
					b_cell_data,
					b_lambda,
					dt
				);
			}
		}
	}
	corr = (1.f / (*rest_density)) * corr;
	correction[originalIndex] = corr;
	//compute new position
	//new_pos[originalIndex] = pos + corr;
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

	new_pos[original_index] = cell_data.sorted_pos[index] + correction[original_index];
	predict_pos[original_index] = new_pos[original_index];
	// write back to sorted_pos for next iteration
	cell_data.sorted_pos[index] = new_pos[original_index];
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

void allocateArray(void** devPtr, size_t size)
{
	checkCudaErrors(cudaMalloc(devPtr, size));
}

void setParams(SimParams* param_in)
{
	checkCudaErrors(cudaMemcpyToSymbol(params, param_in, sizeof(SimParams)));
}

/* Integration for Position based Dynamics */
void integratePBD(
	float3* pos, float3* vel,  
	float3* force, float* massInv,
	float3* predict_pos, float3* new_pos,
	float deltaTime,
	uint numParticles
)
{
	uint numThreads, numBlocks;
	compute_grid_size(numParticles, MAX_THREAD_NUM, numBlocks, numThreads);

	integrate_pbd_d << <numBlocks, numThreads >> > (
		pos, vel, force, massInv,
		predict_pos, new_pos,
		deltaTime,
		numParticles
		);
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

void solve_dem_collision(
	float3* newVel,
	float3* sortedPos,
	float3* sortedVel,
	uint*	gridParticleIndex,
	uint*	cellStart,
	uint*	cellEnd,
	uint	numParticles,
	uint	numCells,
	float	dt)
{

	// thread per particle
	uint numThreads, numBlocks;
	compute_grid_size(numParticles, MAX_THREAD_NUM, numBlocks, numThreads);

	// execute the kernel
	collideD << < numBlocks, numThreads >> > (
		newVel,
		sortedPos,
		sortedVel,
		gridParticleIndex,
		cellStart,
		cellEnd,
		numParticles,
		dt
	);

	// check if kernel invocation generated an error
	getLastCudaError("Kernel execution failed");

}

void solve_sph_fluid(
	float*			rest_density,
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
	std::chrono::steady_clock::time_point t1, t2, t3, t4, t5;
	uint numThreads, numBlocks;
	compute_grid_size(numParticles, MAX_THREAD_NUM, numBlocks, numThreads);

	for (int i = 0; i < iterations; ++i)
	{
		// CUDA SPH Kernel
		// compute density
		t1 = std::chrono::high_resolution_clock::now();
		compute_density_d << <numBlocks, numThreads >> > (
			sph_particles->m_d_density, 
			rest_density,
			sph_cell_data.sorted_pos,
			sph_particles->m_d_mass, sph_particles->m_d_C,
			sph_cell_data.grid_index,
			sph_cell_data.cellStart,
			sph_cell_data.cellEnd,
			b_cell_data,
			boundary_particles->m_d_volume,
			numParticles
			);
		getLastCudaError("Kernel execution failed: compute_density_d ");
		// compute density contributed by boundary particles
		compute_boundary_density_d << <numBlocks, numThreads >> > (
			rest_density,
			sph_cell_data.sorted_pos,
			sph_particles->m_d_mass,
			sph_cell_data.cellStart,
			sph_cell_data.cellEnd,
			sph_cell_data.grid_index,
			b_cell_data,
			boundary_particles->m_d_mass,
			boundary_particles->m_d_volume,
			boundary_particles->m_d_C,
			boundary_particles->m_d_density,
			b_num_particles
			);
		// compute density of bounary particles
		// compute_boundary_density_d();
		getLastCudaError("Kernel execution failed: compute_density_boundary_d ");
		t2 = std::chrono::high_resolution_clock::now();
		// compute lambda
 		compute_lambdas_d << <numBlocks, numThreads >> > (
			sph_particles->m_d_lambda,
			rest_density,
			sph_cell_data.sorted_pos,
			sph_particles->m_d_C,
			sph_particles->m_d_mass,
			sph_cell_data.grid_index,
			sph_cell_data.cellStart,
			sph_cell_data.cellEnd,
			b_cell_data,
			boundary_particles->m_d_volume,
			numParticles
			);
		getLastCudaError("Kernel execution failed: compute_lambdas_d ");
		compute_boundary_lambdas_d << <numBlocks, numThreads >> > (
			boundary_particles->m_d_lambda,
			boundary_particles->m_d_volume,
			boundary_particles->m_d_positions,
			boundary_particles->m_d_C,
			boundary_particles->m_d_mass,
			b_cell_data,
			sph_cell_data.sorted_pos,
			sph_cell_data.grid_index,
			sph_cell_data.cellStart,
			sph_cell_data.cellEnd,
			rest_density,
			b_num_particles
		);
		getLastCudaError("Kernel execution failed: compute_boundary_lambdas_d ");
		t3 = std::chrono::high_resolution_clock::now();
		// compute new position
		compute_position_correction << <numBlocks, numThreads >> > (
			sph_particles->m_d_lambda,
			rest_density,
			sph_cell_data.sorted_pos,
			//sph_particles->m_d_new_positions,
			sph_particles->m_d_correction,
			sph_cell_data.grid_index,
			sph_cell_data.cellStart,
			sph_cell_data.cellEnd,
			b_cell_data,
			boundary_particles->m_d_lambda,
			numParticles,
			dt
		);
		getLastCudaError("Kernel execution failed: compute_position_correction ");
		// correct this iteration
		apply_correction << <numBlocks, numThreads >> > (
			sph_particles->m_d_new_positions, 
			sph_particles->m_d_predict_positions, 
			sph_particles->m_d_correction,
			sph_cell_data,
			numParticles
		);
		getLastCudaError("Kernel execution failed: apply_correction ");

		t4 = std::chrono::high_resolution_clock::now();
	}
	// finalize correction
	finalize_correction << <numBlocks, numThreads >> > (
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

__device__
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
	uint start_index = cell_data.cellStart[grid_hash];
	float3 correction = make_float3(0, 0, 0);

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cellEnd[grid_hash];

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
					float3 n = v / (dist);// +0.000001f);

					correction_j = -w0 * (1.f / w_sum) * C * n;

					/*
					// Tangential correction
					// project on tangential direction
					float penetration = abs(C);
					float3 correction_j_t = correction_j - (dot(correction_j, n) * n);
					float threshold = params.static_friction * penetration;
					float len = length(correction_j_t);
					
					//printf("penetration: %f\n", penetration);
					//printf("Correction: %f, %f, %f\n", correction_j_t.x, correction_j_t.y, correction_j_t.z);
					// use kinematic friction model
					if (length(correction_j_t) < threshold)
					{
						float coeff = min(params.kinematic_friction * penetration / len, 1.f);
						correction_j_t = coeff * correction_j_t;
					}
					
					correction_j_t = (w0 / w_sum) * correction_j_t;
					correction_j += correction_j_t;
					*/
					
				 }
			}
			correction += correction_j;
		}

		//printf("Num neighbors: %u\n", end_index - start_index);
	}
	return correction;
}

__device__
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
	uint start_index = b_cell_data.cellStart[grid_hash];
	float3 correction = make_float3(0, 0, 0);

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = b_cell_data.cellEnd[grid_hash];

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

				// normalize v + 0.000001f for vanish problem
				float3 n = v / (dist);// +0.000001f);

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

	correction[original_index] = corr;
}

__device__
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
	uint start_index = cell_data.cellStart[grid_hash];
	float3 result = make_float3(0,0,0);// correction_i;

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = cell_data.cellEnd[grid_hash];

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

__device__
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
	uint start_index = b_cell_data.cellStart[grid_hash];
	float3 result = make_float3(0, 0, 0);// correction_i;

	if (start_index != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint end_index = b_cell_data.cellEnd[grid_hash];

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
	float3 new_pos0 = new_pos[original_index];
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
	uint numThreads, numBlocks;
	compute_grid_size(numParticles, MAX_THREAD_NUM, numBlocks, numThreads);
	for (int i = 0; i < iteration; ++i)
	{
		compute_distance_correction << <numBlocks, numThreads >> > (
			dem_particles->m_d_correction,
			dem_particles->m_d_massInv,
			boundary_particles->m_d_massInv,
			cell_data,
			b_cell_data,
			numParticles
			);
		getLastCudaError("Kernel execution failed: compute_dem_correction ");
		apply_correction << <numBlocks, numThreads >> > (
			dem_particles->m_d_new_positions,
			dem_particles->m_d_predict_positions,
			dem_particles->m_d_correction,
			cell_data,
			numParticles
			);
		getLastCudaError("Kernel execution failed: apply_correction ");
		
		
	}

	compute_friction_correction << <numBlocks, numThreads >> > (
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
	apply_correction << <numBlocks, numThreads >> > (
		dem_particles->m_d_new_positions,
		dem_particles->m_d_predict_positions,
		dem_particles->m_d_correction,
		cell_data,
		numParticles
		);
	getLastCudaError("Kernel execution failed: apply_correction ");
		
	// finalize correction
	finalize_correction << <numBlocks, numThreads >> > (
		dem_particles->m_d_positions,
		dem_particles->m_d_new_positions,
		dem_particles->m_d_predict_positions,
		dem_particles->m_d_velocity,
		numParticles,
		dt
		);
	getLastCudaError("Kernel execution failed: finalize_correction ");
}
