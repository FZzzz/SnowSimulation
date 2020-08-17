#ifndef _CUDA_SIMULATION_CUH_
#define _CUDA_SIMULATION_CUH_

#include <cuda_runtime.h>
#include "Particle.h"
#include "Simulation.h"
#include "NeighborSearch.h"
//#include <helper_math.h>

#define MAX_THREAD_NUM 512

// simulation parameters
__constant__ SimParams params;

void allocateArray(void** devPtr, size_t size);

void cuda_test_offset(unsigned int block_num, unsigned int thread_num, float3* positions);

void setParams(SimParams* param_in);

void integratePBD(
    float3* pos, float3* vel,
    float3* force, float* massInv,
    float3* predict_pos, float3* new_pos,
    float deltaTime,
    uint numParticles
);

void compute_grid_size(uint n, uint block_size, uint& num_blocks, uint& num_threads);

void calculate_hash(
    CellData cell_data,
    float3* pos,
    uint    num_particles
);

void sort_particles(
    CellData cell_data,
    uint numParticles
);

void reorder_data(
    CellData cell_data,
    float3* oldPos,
    uint	numParticles,
    uint	numCells
);

void compute_boundary_volume(
    CellData data,
    float* mass,
    float* volume,          // output: volume of particle
    uint numParticles
);


void solve_dem_collision(
    float3* newVel,
    float3* sortedPos,
    float3* sortedVel,
    uint* gridParticleIndex,
    uint* cellStart,
    uint* cellEnd,
    uint   numParticles,
    uint   numCells,
    float dt  
);

void solve_sph_fluid(
    float*          rest_density,
    ParticleSet*    sph_particles,
    CellData		sph_cell_data,
    uint			numParticles,
    ParticleSet*    boundary_particles,
    CellData		b_cell_data,
    uint			b_num_particles,
    float			dt,
    int             iterations=1
);

void solve_pbd_dem(
    ParticleSet* dem_particles,
    ParticleSet* boundary_particles,
    CellData     cell_data,
    CellData     b_cell_data,
    uint         numParticles,
    uint         b_numParticles,
    float        dt,
    int          iteration
);

#endif
