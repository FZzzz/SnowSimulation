#include <omp.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include "NeighborSearch.h"

#define PRE_ALLOCATE_ENTRY_SIZE 100

/*
 * NavieSearch()
 * @return The neighbors of the particle 
 */


NeighborSearch::NeighborSearch(std::shared_ptr<ParticleSystem> particle_system, uint3 grid_size):
    m_particle_system(particle_system), 
    m_grid_spacing(1.f),
    m_grid_size(grid_size)
{
    m_num_grid_cells = m_grid_size.x * m_grid_size.y * m_grid_size.z;
}

NeighborSearch::~NeighborSearch()
{
}

void NeighborSearch::Initialize()
{
#ifdef _DEBUG
    assert(m_particle_system != nullptr);
#endif
    /*
    const auto& particles = m_particle_system->getParticles();
    m_search_cache.resize(particles.size(), std::vector<size_t>());
    */
    const ParticleSet* const particles = m_particle_system->getSPHParticles();
    m_search_cache.resize(particles->m_size, std::vector<size_t>());

}

void NeighborSearch::InitializeCUDA()
{
#ifdef _DEBUG
    assert(m_particle_system != nullptr);
#endif

    const ParticleSet* const sph_particles = m_particle_system->getSPHParticles();
    const ParticleSet* const dem_particles = m_particle_system->getDEMParticles();
    const ParticleSet* const boundary_particles = m_particle_system->getBoundaryParticles();

    /*Allocate CUDA memory*/
    /* SPH cell data initialization */
    if (sph_particles != nullptr)
    {
        cudaMalloc((void**)&m_d_sph_cell_data.grid_hash, sph_particles->m_size * sizeof(uint));
        cudaMalloc((void**)&m_d_sph_cell_data.grid_index, sph_particles->m_size * sizeof(uint));

        cudaMalloc((void**)&m_d_sph_cell_data.cell_start, m_num_grid_cells * sizeof(uint));
        cudaMalloc((void**)&m_d_sph_cell_data.cell_end, m_num_grid_cells * sizeof(uint));

        cudaMalloc((void**)&m_d_sph_cell_data.sorted_pos, sph_particles->m_size * sizeof(float3));
    }

    if (dem_particles != nullptr)
    {
        cudaMalloc((void**)&m_d_dem_cell_data.grid_hash, dem_particles->m_size * sizeof(uint));
        cudaMalloc((void**)&m_d_dem_cell_data.grid_index, dem_particles->m_size * sizeof(uint));

        cudaMalloc((void**)&m_d_dem_cell_data.cell_start, m_num_grid_cells * sizeof(uint));
        cudaMalloc((void**)&m_d_dem_cell_data.cell_end, m_num_grid_cells * sizeof(uint));

        cudaMalloc((void**)&m_d_dem_cell_data.sorted_pos, dem_particles->m_size * sizeof(float3));
    }

    /* Boundary particle cell data initialization */
    if (boundary_particles != nullptr)
    {
        cudaMalloc((void**)&m_d_boundary_cell_data.grid_hash, boundary_particles->m_size * sizeof(uint));
        cudaMalloc((void**)&m_d_boundary_cell_data.grid_index, boundary_particles->m_size * sizeof(uint));

        cudaMalloc((void**)&m_d_boundary_cell_data.cell_start, m_num_grid_cells * sizeof(uint));
        cudaMalloc((void**)&m_d_boundary_cell_data.cell_end, m_num_grid_cells * sizeof(uint));

        cudaMalloc((void**)&m_d_boundary_cell_data.sorted_pos, boundary_particles->m_size * sizeof(float3));
    }
}

void NeighborSearch::Release()
{
    const ParticleSet* const sph_particles = m_particle_system->getSPHParticles();
    const ParticleSet* const boundary_particles = m_particle_system->getBoundaryParticles();

    if (sph_particles != nullptr)
    {
        cudaFree(m_d_sph_cell_data.grid_hash);
        cudaFree(m_d_sph_cell_data.grid_index);
        cudaFree(m_d_sph_cell_data.cell_start);
        cudaFree(m_d_sph_cell_data.cell_end);
        cudaFree(m_d_sph_cell_data.sorted_pos);
    }
    if (boundary_particles != nullptr)
    {
        cudaFree(m_d_boundary_cell_data.grid_hash);
        cudaFree(m_d_boundary_cell_data.grid_index);
        cudaFree(m_d_boundary_cell_data.cell_start);
        cudaFree(m_d_boundary_cell_data.cell_end);
        cudaFree(m_d_boundary_cell_data.sorted_pos);
    }

}

/*
 * Perform the most naive search (O(n^2)) (full search)
 */
void NeighborSearch::NaiveSearch(float effective_radius)
{
    if (m_search_cache.size() == 0 || effective_radius < 0)
        return;

    for (size_t i = 0; i < m_search_cache.size(); ++i)
        m_search_cache[i].clear();

    const float square_h = effective_radius * effective_radius;
    const ParticleSet* const particles = m_particle_system->getSPHParticles();
    //#pragma omp parallel default(shared) num_threads(8)// Personally I think this is useless... (cannot prevent race condition)
    {
        //#pragma omp for schedule(dynamic)  // Using round-robin scheduling
        for (int i = 0; i < particles->m_size; ++i)
        {
            for (int j = i + 1; j < particles->m_size; ++j)
            {
                float distance2 = glm::distance2(particles->m_new_positions[i], particles->m_new_positions[j]);
                if (distance2 <= square_h)
                {
                    m_search_cache[i].push_back(static_cast<size_t>(j));
                    m_search_cache[j].push_back(static_cast<size_t>(i));
                }
            }
        }
    }
}

void NeighborSearch::SpatialSearch(float effective_radius)
{
#ifdef _DEBUG
    assert(effective_radius > 0);
#endif
    if (m_search_cache.size() == 0)
        return;

    const float square_h = effective_radius * effective_radius;
    const ParticleSet* const particles = m_particle_system->getSPHParticles();

    for (auto entry : m_hashtable)
    {
        if (entry.second != nullptr)
            entry.second->particles.clear();
    }

    for (size_t i = 0; i < m_search_cache.size(); ++i)
        m_search_cache[i].clear();

    //#pragma omp parallel default(shared)
    {
        //#pragma omp for schedule(dynamic)
        for (int i = 0; i < particles->m_size; ++i)
        {
            // Flooring 
            glm::i32vec3 grid_index = Flooring(particles->m_predict_positions[i]);
            // Hashing
            uint32_t hash_value = GetHashValue(grid_index);
            // Filling in table
            if (m_hashtable.find(hash_value) == m_hashtable.end())
            {
                HashEntry* entry = new HashEntry();

                // Particles tend to occur nearby, so we allocate a small space for other particles
                entry->particles.reserve(PRE_ALLOCATE_ENTRY_SIZE);
                entry->particles.push_back(i);
                m_hashtable.emplace(hash_value, entry);
            }
            else
            {
                if (m_hashtable[hash_value] != nullptr)
                {
                    HashEntry* entry = m_hashtable[hash_value];
                    entry->particles.push_back(i);
                }
                else
                {
                    HashEntry* entry = new HashEntry();
                    entry->particles.reserve(PRE_ALLOCATE_ENTRY_SIZE);
                    entry->particles.push_back(i);
                    m_hashtable.emplace(hash_value, entry);
                }
            }
        }
    }
    #pragma omp parallel default(shared) num_threads(8)
    {
        // Filling search cache
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < particles->m_size; ++i)
        {
            // Search 27 neighbor cells
            for (int32_t x = -1; x < 2; ++x)
            {
                for (int32_t y = -1; y < 2; ++y)
                {
                    for (int32_t z = -1; z < 2; ++z)
                    {
                        glm::i32vec3 grid_index = Flooring(particles->m_predict_positions[i]);
                        glm::i32vec3 search_idx = grid_index;
                        search_idx.x = grid_index.x + x;
                        search_idx.y = grid_index.y + y;
                        search_idx.z = grid_index.z + z;

                        uint32_t hash_value = GetHashValue(search_idx);

                        if (m_hashtable.find(hash_value) != m_hashtable.end())
                        {
                            HashEntry* entry = m_hashtable[hash_value];

                            for (int j = 0; j < entry->particles.size(); ++j)
                            {
                                size_t p_j = entry->particles[j];

                                if (p_j == i)
                                    continue;

                                float distance2 = glm::distance2(
                                    particles->m_predict_positions[i],
                                    particles->m_predict_positions[p_j]
                                );

                                if (distance2 <= square_h)
                                {
                                    m_search_cache[i].push_back(static_cast<size_t>(p_j));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

}

const std::vector<size_t>& NeighborSearch::FetchNeighbors(size_t i)
{
	return m_search_cache[i];
}

glm::i32vec3 NeighborSearch::Flooring(const glm::vec3& position)
{
    glm::i32vec3 grid_index;
    
    grid_index.x = static_cast<int32_t>(std::floorf(position.x / m_grid_spacing.x));
    grid_index.y = static_cast<int32_t>(std::floorf(position.y / m_grid_spacing.y));
    grid_index.z = static_cast<int32_t>(std::floorf(position.z / m_grid_spacing.z));
    
    return grid_index;
}

/* Hash function from https://github.com/InteractiveComputerGraphics/PositionBasedDynamics */
uint32_t NeighborSearch::GetHashValue(const glm::i32vec3& key)
{
    uint32_t value;
    
    const int p1 = 73856093 * key.x;
    const int p2 = 19349663 * key.y;
    const int p3 = 83492791 * key.z;
    
    value = p1 + p2 + p3;

    return value;
}
