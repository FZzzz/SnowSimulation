#ifndef _NEIGHBOR_SEARCH_H_
#define _NEIGHBOR_SEARCH_H_

#include <vector>
#include <memory>
#include <unordered_map>
#include <atomic>
#include <cstdint>
#include <cuda_runtime.h>
#include <helper_math.h>
#include "common.h"
#include "ParticleSystem.h"

struct HashEntry
{
	std::vector<size_t> particles;
};

struct CellData
{
	uint* grid_hash;
	uint* grid_index;
	uint* cellStart;
	uint* cellEnd;
	float3* sorted_pos;
};

class NeighborSearch
{
public:
	NeighborSearch(std::shared_ptr<ParticleSystem> particle_system, uint3 grid_size);
	~NeighborSearch();

	void Initialize();
	void InitializeCUDA();

	void Release();

	void NaiveSearch(float effective_radius);
	void SpatialSearch(float effective_radius);

	const std::vector<size_t>& FetchNeighbors(size_t i);

	uint3 m_grid_size;
	uint  m_num_grid_cells;
	/*
	uint* m_d_grid_particle_hash;
	uint* m_d_grid_particle_index;
	uint* m_d_cellStart;
	uint* m_d_cellEnd;
	*/

	struct CellData m_d_boundary_cell_data;
	struct CellData m_d_sph_cell_data;

private:

	// Flooring function
	glm::i32vec3 Flooring(const glm::vec3& position);
	// Hashing function
	uint32_t GetHashValue(const glm::i32vec3& key);

	std::shared_ptr<ParticleSystem>	 m_particle_system;
	std::vector<std::vector<size_t>> m_search_cache;

	glm::vec3 m_grid_spacing;
	std::unordered_map<unsigned int, HashEntry*> m_hashtable;
	
};

#endif
