#ifndef _CUDA_SIMULATION_CUH_
#define _CUDA_SIMULATION_CUH_

#include <cuda_runtime.h>
#include "Particle.h"
#include "SimParams.h"
#include "NeighborSearch.h"
//#include <helper_math.h>

#define MAX_THREAD_NUM 512

// simulation parameters
__constant__ SimParams params;

void allocate_array(void** devPtr, size_t size);

void set_sim_params(SimParams* param_in);

void integrate_pbd(
    ParticleSet* particles,
    float deltaTime,
    uint numParticles,
    bool cd_on
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

void solve_sph_fluid(
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

/* solve dem particle as sph particles */
void solve_dem_sph(
    ParticleSet* dem_particles,
    ParticleSet* sph_particles,
    CellData     sph_cell_data,
    CellData     dem_cell_data,
    uint         num_dem_particles,
    uint         num_sph_particles,
    float        dt
);

/* solve sph particles as dem particles */
void solve_sph_dem(
    ParticleSet* sph_particles,
    ParticleSet* dem_particles,
    CellData     sph_cell_data,
    CellData     dem_cell_data,
    uint         num_sph_particles,
    uint         num_dem_particles,
    float        dt
);

void snow_simulation(
    ParticleSet* sph_particles,
    ParticleSet* dem_particles,
    ParticleSet* boundary_particles,
    ParticleDeviceData* phase_change_buffer,
    CellData     sph_cell_data,
    CellData     dem_cell_data,
    CellData     b_cell_data,
    float        dt,
    int          iterations,
    bool         correct_dem,
    bool         sph_sph_correction,
    bool         dem_friction,
    bool         change_phase,
    bool         cd_on
);

#endif
