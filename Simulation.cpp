#include "Simulation.h"
#include "imgui/imgui.h"
#include "CollisionDetection.h"
#include "SPHKernel.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <omp.h>
#include <chrono>
#include <cstdlib>
#include "cuda_simulation.cuh"

Simulation::Simulation()
	: m_solver(nullptr), m_particle_system(nullptr), m_neighbor_searcher(nullptr),
	m_initialized(false), m_world_desc(SimWorldDesc(-9.8f, 0.f)), m_pause(true),
	m_first_frame(true),
	m_rest_density(0.8f),
	m_iterations(1),
	m_clip_length(1000)
{
	

}
Simulation::Simulation(SimWorldDesc desc)
	: m_solver(nullptr), m_particle_system(nullptr), m_neighbor_searcher(nullptr), 
	m_initialized(false), m_world_desc(desc), m_pause(false)
{
}

Simulation::~Simulation()
{
}

void Simulation::Initialize(PBD_MODE mode, std::shared_ptr<ParticleSystem> particle_system)
{
	m_particle_system = particle_system;
	
	uint3 grid_size = make_uint3(64, 64, 64);
	glm::vec3 fluid_half_extends = glm::vec3(0.9998f, 0.1f, 0.9998f);
	glm::vec3 snow_half_extends = glm::vec3(0.25f, 0.25f, 0.25f);
	glm::vec3 fluid_origin = glm::vec3(0.0f, 0.12f, 0.0f);
	glm::vec3 snow_origin = glm::vec3(0.f, 1.251f, 0.0f);
	
	m_neighbor_searcher = std::make_shared<NeighborSearch>(m_particle_system, grid_size);
	m_solver = std::make_shared<ConstraintSolver>(mode);

	SetupSimParams();
	GenerateParticleCube(fluid_half_extends, fluid_origin, 0, false);
	GenerateParticleCube(snow_half_extends, snow_origin, 1, false);
	InitializeBoundaryParticles();

#ifdef _USE_CUDA_
	m_particle_system->InitializeCUDA();
	m_neighbor_searcher->InitializeCUDA();
	InitializeBoundaryCudaData();
#else
	m_particle_system->Initialize();
	m_neighbor_searcher->Initialize();
#endif
	
	m_initialized = true;
	
}

/*
 * Step simulation
 * @param[in] dt time step
 * @retval true  Successfully project constraints
 * @retval false Failed on constraint projection
 */
bool Simulation::Step(float dt)
{
	if (!m_initialized)
		return false;

	if (m_pause)
		return true;

	const float effective_radius = 1.f;
	std::chrono::steady_clock::time_point t1, t2, t3, t4, t5;

	PredictPositions(dt);

	t1 = std::chrono::high_resolution_clock::now();
	CollisionDetection(dt);
	HandleCollisionResponse();
	GenerateCollisionConstraint();

	t2 = std::chrono::high_resolution_clock::now();
	FindNeighborParticles(effective_radius);

	t3 = std::chrono::high_resolution_clock::now();
	
	if (m_first_frame)
	{
		ComputeRestDensity();
		std::cout << "Rest density: " << m_rest_density << std::endl;
	}
	
	
	//m_rest_density = 10.f / 12.f;

	for (uint32_t i = 0; i < m_solver->getSolverIteration(); ++i)
	{		
		ComputeDensity(effective_radius);
		ComputeLambdas(effective_radius);
		ComputeSPHParticlesCorrection(effective_radius, dt);
		UpdatePredictPosition();
	}
	t4 = std::chrono::high_resolution_clock::now();
	
 	if (!ProjectConstraints(dt))
		return false;

	ApplySolverResults(dt);

	{
		ImGui::Begin("Performance");
		ImGui::Text("Collision:\t %.5lf (ms)", (t2 - t1).count() / 1000000.0);
		ImGui::Text("Searching:\t %.5lf (ms)", (t3 - t2).count() / 1000000.0);
		ImGui::Text("Correction:\t%.5lf (ms)", (t4 - t3).count() / 1000000.0);
		ImGui::Text("GL update:\t%.5lf (ms)", m_particle_system->getUpdateTime());
		ImGui::End();
	}

	//m_pause = true;

	return true;
}

bool Simulation::StepCUDA(float dt)
{
	if (!m_initialized)
		return false;

	if (m_pause)
		return true;

	bool cd_on = true;
	bool correct_dem = true;
	bool sph_sph_correction = false;

	std::chrono::steady_clock::time_point t1, t2, t3, t4, t5;

	ParticleSet* sph_particles = m_particle_system->getSPHParticles();
	ParticleSet* dem_particles = m_particle_system->getDEMParticles();
	ParticleSet* boundary_particles = m_particle_system->getBoundaryParticles();

	cudaGraphicsResource** sph_vbo_resource = m_particle_system->getSPHCUDAGraphicsResource();
	cudaGraphicsResource** dem_vbo_resource = m_particle_system->getDEMCUDAGraphicsResource();
	cudaGraphicsResource** b_vbo_resource = m_particle_system->getBoundaryCUDAGraphicsResource();
	
	glm::vec3* positions = sph_particles->m_positions.data();

	unsigned int sph_num_particles = sph_particles->m_size;
	unsigned int dem_num_particles = dem_particles->m_size;
	unsigned int b_num_particles = boundary_particles->m_size;

	// Map vbo to m_d_positinos
	cudaGraphicsMapResources(1, sph_vbo_resource, 0);
	cudaGraphicsMapResources(1, dem_vbo_resource, 0);
	cudaGraphicsMapResources(1, b_vbo_resource, 0);

	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void**)&(sph_particles->m_d_positions), &num_bytes, *sph_vbo_resource);
	cudaGraphicsResourceGetMappedPointer((void**)&(dem_particles->m_d_positions), &num_bytes, *dem_vbo_resource);
	cudaGraphicsResourceGetMappedPointer((void**)&(boundary_particles->m_d_positions), &num_bytes, *b_vbo_resource);
	
	// Integrate
	//integrate(particles->m_d_positions, particles->m_d_velocity, dt, particles->m_size);

	t1 = std::chrono::high_resolution_clock::now();
	integrate_pbd(
		sph_particles,
		dt,
		sph_num_particles,
		cd_on
	);
	
	integrate_pbd(
		dem_particles,
		dt,
		dem_num_particles,
		cd_on
	);

	t2 = std::chrono::high_resolution_clock::now();
	// Neighbor search
	calculate_hash(
		m_neighbor_searcher->m_d_sph_cell_data,
		sph_particles->m_d_predict_positions,
		sph_num_particles
	);
	sort_particles(
		m_neighbor_searcher->m_d_sph_cell_data,
		sph_num_particles
	);
	reorder_data(
		m_neighbor_searcher->m_d_sph_cell_data,
		//particles->m_d_positions,
		sph_particles->m_d_predict_positions,
		sph_num_particles,
		m_neighbor_searcher->m_num_grid_cells
	);

	// DEM particles
	calculate_hash(
		m_neighbor_searcher->m_d_dem_cell_data,
		dem_particles->m_d_predict_positions,
		dem_num_particles
		);
	sort_particles(
		m_neighbor_searcher->m_d_dem_cell_data,
		dem_num_particles
		);
	reorder_data(
		m_neighbor_searcher->m_d_dem_cell_data,
		//particles->m_d_positions,
		dem_particles->m_d_predict_positions,
		dem_num_particles,
		m_neighbor_searcher->m_num_grid_cells
		);

	t3 = std::chrono::high_resolution_clock::now();

	snow_simulation(
		sph_particles,
		dem_particles,
		boundary_particles,
		m_neighbor_searcher->m_d_sph_cell_data,
		m_neighbor_searcher->m_d_dem_cell_data,
		m_neighbor_searcher->m_d_boundary_cell_data,
		sph_num_particles,
		dem_num_particles,
		b_num_particles,
		dt,
		m_iterations,
		correct_dem,
		sph_sph_correction
		);

	/*
	solve_pbd_dem(
		particles,
		boundary_particles,
		m_neighbor_searcher->m_d_sph_cell_data,
		m_neighbor_searcher->m_d_boundary_cell_data,
		numParticles,
		b_numParticles,
		dt,
		m_iterations
	);
	*/
	
	/*
	solve_sph_fluid(
		sph_particles,
		m_neighbor_searcher->m_d_sph_cell_data,
		sph_num_particles,
		boundary_particles,
		m_neighbor_searcher->m_d_boundary_cell_data,
		b_num_particles,
		dt,
		m_iterations
	);
	*/
	t4 = std::chrono::high_resolution_clock::now();

	{
		ImGui::Begin("CUDA Performance");
		ImGui::Text("Integrate:   %.5lf (ms)", (t2 - t1).count() / 1000000.0f);
		ImGui::Text("Search:      %.5lf (ms)", (t3 - t2).count() / 1000000.0f);
		ImGui::Text("Solve:       %.5lf (ms)", (t4 - t3).count() / 1000000.0f);
		ImGui::End();
	}
	
	// Unmap CUDA buffer object
	cudaGraphicsUnmapResources(1, sph_vbo_resource, 0);
	cudaGraphicsUnmapResources(1, dem_vbo_resource, 0);
	cudaGraphicsUnmapResources(1, b_vbo_resource, 0);
	//m_pause = true;

	static int count = 0;
	if(!m_pause) count++;
	
	if (count == m_clip_length)
		m_pause = true, count = 0;
	
	return true;
}

void Simulation::AddCollider(Collider* collider)
{
	m_colliders.push_back(collider);
	m_collision_table.push_back(std::vector<Collider*>());
}

void Simulation::AddStaticConstraint(Constraint* constraint)
{
	m_static_constraints.push_back(constraint);
}

void Simulation::AddStaticConstraints(std::vector<Constraint*> constraints)
{
	m_static_constraints.insert(m_static_constraints.end(), constraints.begin(), constraints.end());
}

void Simulation::SetSolverIteration(uint32_t iter_count)
{
	m_solver->setSolverIteration(iter_count);
	m_iterations = iter_count;
}

void Simulation::ComputeRestDensity()
{
	const float effective_radius = 1.f;
	ParticleSet* const particles = m_particle_system->getSPHParticles();

	m_rest_density = 0;
	
	//#pragma omp parallel default(shared) num_threads(8)
	{
		//#pragma omp for schedule(dynamic)
		for (int i = 0; i < particles->m_size; ++i)
		{
			auto neighbors = m_neighbor_searcher->FetchNeighbors(static_cast<size_t>(i));
			particles->m_density[i] = particles->m_mass[i] * SPHKernel::Poly6_W(0, effective_radius);


			for (int j = 0; j < neighbors.size(); ++j)
			{
				float distance = glm::distance(particles->m_positions[i], particles->m_positions[neighbors[j]]);
				particles->m_density[i] += particles->m_mass[neighbors[j]] * SPHKernel::Poly6_W(distance, effective_radius);
			}
			m_rest_density += particles->m_density[i];
		}
	}

	m_rest_density /= static_cast<float>(particles->m_size);
	m_first_frame = false;
}

void Simulation::Pause()
{
	m_pause = !m_pause;
}

void Simulation::setGravity(float gravity)
{
	m_world_desc.gravity = gravity;
}

void Simulation::setClipLength(int length)
{
	m_clip_length = length;
}

void Simulation::SetupSimParams()
{
	//const size_t n_particles = 1000;
	const float particle_mass = 0.015f;
	const float n_kernel_particles = 20.f;	
	const float dem_sph_ratio = 1.25f;
	// water density = 1000 kg/m^3
	m_rest_density = 1000.f; 
	m_sph_particle_mass = particle_mass;
	m_dem_particle_mass = dem_sph_ratio * particle_mass;

	float effective_radius, particle_radius;
	
	/* Compute parameters from mass and n_particles*/
	m_volume = n_kernel_particles * particle_mass / m_rest_density;
	effective_radius = powf(((3.0f * m_volume) / (4.0f * M_PI)), 1.0f / 3.0f);
	particle_radius = powf((M_PI / (6.0f * n_kernel_particles)), 1.0f / 3.0f) * effective_radius;

	std::cout << "Particle mass: " << particle_mass << std::endl;
	std::cout << "Effective radius: " << effective_radius << std::endl;
	std::cout << "Particle radius: " << particle_radius << std::endl;

	m_sim_params = new SimParams();

	m_sim_params->gravity = make_float3(0.f, -9.8f, 0.f);
	m_sim_params->global_damping = 1.f;
	m_sim_params->maximum_speed = 500.f;

	m_sim_params->particle_radius = particle_radius;
	m_sim_params->effective_radius = effective_radius;
	m_sim_params->rest_density = m_rest_density;
	m_sim_params->epsilon = 1000.f;
	m_sim_params->pbd_epsilon = 0.8f * particle_radius;
	m_sim_params->grid_size = m_neighbor_searcher->m_grid_size;
	m_sim_params->num_cells = m_neighbor_searcher->m_num_grid_cells;
	m_sim_params->world_origin = make_float3(0, 0, 0);
	m_sim_params->cell_size = make_float3(m_sim_params->effective_radius);
	m_sim_params->boundary_damping = 0.5f;
	
	//coupling coefficients
	//m_sim_params->sph_dem_corr = 0.05f;

	m_sim_params->static_friction = 1.0f;
	m_sim_params->kinematic_friction = 0.75f;

	m_sim_params->scorr_coeff = 0.1f;
	m_sim_params->sor_coeff = 1.0f * (1.f/4.f);
	m_sim_params->viscosity = 0.001f;

	m_sim_params->poly6 = (315.0f / (64.0f * M_PI * glm::pow(effective_radius, 9)));
	m_sim_params->poly6_G = (-945.0f / (32.0f * M_PI * glm::pow(effective_radius, 9)));
	m_sim_params->spiky = (15.0f / (M_PI * glm::pow(effective_radius, 6)));
	m_sim_params->spiky_G = (-45.0f / (M_PI * glm::pow(effective_radius, 6)));

	m_particle_system->setParticleRadius(particle_radius);

	

	set_sim_params(m_sim_params);
}

void Simulation::InitializeBoundaryParticles()
{
	const int thickness = 1;
	const float diameter = 2.f * m_sim_params->particle_radius;
	// number of particles on x,y,z
	int nx, ny, nz;
	//size_t n_particles = 0;
	// fluid cube extends
	glm::vec3 half_extend1(1.0f, diameter * static_cast<float>(thickness)/2.f, 1.0f);
	glm::vec3 half_extend2(diameter * static_cast<float>(thickness)/2.f, 1.0f, 1.0f);
	glm::vec3 half_extend3(1.0f, 1.0f, diameter * static_cast<float>(thickness)/2.f);
	
	// Initialize boundary particles
	ParticleSet* particles = m_particle_system->AllocateBoundaryParticles();

	// Initialize positions
	float x, y, z;
	std::vector<glm::vec3> positions;
	// Buttom boundary
	size_t idx = 0;
	nx = static_cast<int>(half_extend1.x / diameter);
	ny = static_cast<int>(half_extend1.y / diameter);
	nz = static_cast<int>(half_extend1.z / diameter);

	// buttom
	float left_margin = INFINITY, right_margin = -INFINITY;
	glm::vec3 buttom_origin(0, -0.5f*diameter, 0);

	for (int i = -nx-thickness; i < nx+thickness; ++i)
	{
		for (int j = 0; j < thickness; ++j)
		{
			for (int k = -nz-thickness; k < nz+thickness; ++k)
			{
				//int idx = k + j * 10 + i * 100;
				x = buttom_origin.x + diameter * static_cast<float>(i);
				y = buttom_origin.y - diameter * static_cast<float>(j);
				z = buttom_origin.z + diameter * static_cast<float>(k);
				glm::vec3 pos(x, y, z);
				positions.push_back(pos);

				if ( x < left_margin) left_margin = x;
				if ( x > right_margin) right_margin = x;
				idx++;
			}
		}
	}
	
	left_margin -= diameter;
	right_margin += diameter;
	
	nx = static_cast<int>(half_extend2.x / diameter);
	ny = static_cast<int>(half_extend2.y / diameter);
	nz = static_cast<int>(half_extend2.z / diameter);

	// left && right
	float back_margin = INFINITY, front_margin = -INFINITY;
	glm::vec3 left_origin(left_margin, ny * diameter, 0);
	glm::vec3 right_origin(right_margin, ny * diameter, 0);


	for (int i = 0; i < thickness; ++i)
	{
		for (int j = -ny - thickness; j < ny; ++j)
		{
			for (int k = -nz-thickness; k < nz+thickness; ++k)
			{
				// left
				x = -diameter * static_cast<float>(i);
				y = diameter * static_cast<float>(j);
				z = diameter * static_cast<float>(k);
				glm::vec3 pos(x, y, z);
				pos += left_origin;
				positions.push_back(pos);
				idx++;

				// right	
				x = diameter * static_cast<float>(i);
				pos = glm::vec3(x, y, z);
				pos += right_origin;
				positions.push_back(pos);
				idx++;

				if (pos.z < back_margin) back_margin = pos.z;
				if (pos.z > front_margin) front_margin = pos.z;

			}
		}
	}
	 
	back_margin -= diameter;
	front_margin += diameter;

	nx = static_cast<int>(half_extend3.x / diameter);
	ny = static_cast<int>(half_extend3.y / diameter);
	nz = static_cast<int>(half_extend3.z / diameter);

	// back && front boundary
	glm::vec3 back_origin(0, ny * diameter, back_margin);
	glm::vec3 front_origin(0, ny* diameter, front_margin);
	for (int i = -nx-2*thickness; i < nx+2*thickness; ++i)
	{
		for (int j = -ny-thickness; j < ny; ++j)
		{
			for (int k = 0; k < thickness; ++k)
			{
				// back
				x = diameter * static_cast<float>(i);
				y = diameter * static_cast<float>(j);
				z = -diameter * static_cast<float>(k);
				glm::vec3 pos(x, y, z);
				pos += back_origin;
				positions.push_back(pos);
				idx++;

				// front
				z = diameter * static_cast<float>(k);
				pos = glm::vec3(x, y, z);
				pos += front_origin;
				positions.push_back(pos);
				idx++;
			}
		}
	}

	std::cout << "Boundary particles: " << idx << std::endl;
	particles->ResetPositions(positions, m_sph_particle_mass);

}

void Simulation::InitializeBoundaryCudaData()
{
	auto boundary_particles = m_particle_system->getBoundaryParticles();
	size_t num_particles = boundary_particles->m_size;
	cudaGraphicsResource** vbo_resource = m_particle_system->getBoundaryCUDAGraphicsResource();
	
	// Map vbo to m_d_positinos
	cudaGraphicsMapResources(1, vbo_resource, 0);

	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void**)&(boundary_particles->m_d_positions), &num_bytes, *vbo_resource);
	//std::cout << "num_bytes " << num_bytes << std::endl;
	// Precompute hash
	calculate_hash(m_neighbor_searcher->m_d_boundary_cell_data, boundary_particles->m_d_positions, num_particles);
	// Sort
	sort_particles(m_neighbor_searcher->m_d_boundary_cell_data, num_particles);
	// Reorder
	reorder_data(
		m_neighbor_searcher->m_d_boundary_cell_data,
		boundary_particles->m_d_positions, 
		num_particles,
		m_neighbor_searcher->m_num_grid_cells
	);

	// Compute boundary particle volume
	compute_boundary_volume(
		m_neighbor_searcher->m_d_boundary_cell_data,
		boundary_particles->m_d_mass,
		boundary_particles->m_d_volume,
		num_particles
	);

	// Unmap CUDA buffer object
	cudaGraphicsUnmapResources(1, vbo_resource, 0);
}

void Simulation::GenerateParticleCube(glm::vec3 half_extends, glm::vec3 origin, int opt, bool use_jitter=false)
{
	std::srand(time(NULL));
	// diameter of particle
	const float diameter = 2.f * m_sim_params->particle_radius;
	// number of particles on x,y,z
	int nx, ny, nz;
	// fluid cube extends
	
	nx = static_cast<int>(half_extends.x / diameter);
	ny = static_cast<int>(half_extends.y / diameter);
	nz = static_cast<int>(half_extends.z / diameter);

	float x, y, z;
	
	//const float diameter = 0.5f;

	size_t n_particles = 8 * nx * ny * nz;
	ParticleSet* particles = nullptr;

	// 0: sph particles
	// 1: dem particles
	if (opt == 0)
		particles = m_particle_system->AllocateSPHParticles(n_particles, m_sph_particle_mass);
	else if (opt == 1)
		particles = m_particle_system->AllocateDEMParticles(n_particles, m_dem_particle_mass);
	else
		return;
	

	std::cout << ((opt==0)?"SPH": "DEM") << " particles: " << n_particles << std::endl;
	// set positions
	size_t idx = 0;
	for (int i = -nx; i < nx; ++i)
	{
		for (int j = -ny; j < ny; ++j)
		{
			for (int k = -nz; k < nz; ++k)
			{
				x = origin.x + diameter * static_cast<float>(i);
				y = origin.y + diameter * static_cast<float>(j);
				z = origin.z + diameter * static_cast<float>(k);

				if (use_jitter)
				{
					float x_jitter = 0.001f * diameter * static_cast<float>(rand() % 3);
					float y_jitter = 0.001f * diameter * static_cast<float>(rand() % 3);
					float z_jitter = 0.001f * diameter * static_cast<float>(rand() % 3);
					x += x_jitter;
					y += y_jitter;
					z += z_jitter;
				}

				glm::vec3 pos(x, y, z);
				particles->m_positions[idx] = pos;
				particles->m_new_positions[idx] = pos;
				particles->m_predict_positions[idx] = pos;
				idx++;
			}
		}
	}
	//std::cout << "idx " << idx << std::endl;

}


void Simulation::PredictPositions(float dt)
{
	/*
	 * Update position and velocity
	 * 1. forall vertices i do v_i = v_i + dt * w_i * f_{ext}
	 * 2. damp velocities 	
	 */
	ParticleSet* particles = m_particle_system->getSPHParticles();

#pragma omp parallel default(shared) num_threads(8)
	{
		#pragma omp for schedule(static)
		for (int i = 0; i < particles->m_size; ++i)
		{
			particles->m_force[i] = particles->m_mass[i] * glm::vec3(0, m_world_desc.gravity, 0);
			particles->m_velocity[i] = particles->m_velocity[i] + dt * particles->m_massInv[i] * particles->m_force[i];
			particles->m_predict_positions[i] = particles->m_positions[i] + dt * particles->m_velocity[i];
			particles->m_new_positions[i] = particles->m_predict_positions[i];
		}
	}

}

void Simulation::FindNeighborParticles(float effective_radius)
{
	//m_neighbor_searcher->NaiveSearch(effective_radius);
	m_neighbor_searcher->SpatialSearch(effective_radius);
}

void Simulation::ComputeDensity(float effective_radius)
{
	/*
	{
		const auto& neighbors = m_neighbor_searcher->FetchNeighbors(0);
		ImGui::Begin("Neighbor test");
		ImGui::Text("Object Array Size: %u", neighbors.size());
		ImGui::End();
	}
	*/
	ParticleSet* const particles = m_particle_system->getSPHParticles();

	#pragma omp parallel default(shared) num_threads(8)
	{
		#pragma omp for schedule(dynamic)
		for (int i = 0; i < particles->m_size; ++i)
		{
			auto neighbors = m_neighbor_searcher->FetchNeighbors(static_cast<size_t>(i));
			particles->m_density[i] = particles->m_mass[i] * SPHKernel::Poly6_W(0, effective_radius);


			for (int j = 0; j < neighbors.size(); ++j)
			{
				float distance = glm::distance(particles->m_predict_positions[i], particles->m_predict_positions[neighbors[j]]);
				
				float res = SPHKernel::Poly6_W(distance, effective_radius);
				particles->m_density[i] += particles->m_mass[neighbors[j]] * res;
			}
			particles->m_C[i] = particles->m_density[i] / m_rest_density - 1.f;
			std::cout << "C: " << particles->m_C[i] << std::endl;
		}
	}

}

void Simulation::ComputeLambdas(float effective_radius)
{
	const float epsilon = 1.0e-6f;
	/* Compute density constraints */
	ParticleSet* const particles = m_particle_system->getSPHParticles();

	#pragma omp parallel default(shared) num_threads(8)
	{
		#pragma omp for schedule(dynamic)
		for (int i = 0; i < particles->m_size; ++i)
		{
			auto neighbors = m_neighbor_searcher->FetchNeighbors(static_cast<size_t>(i));

			// Reset Lagragian multiplier
			particles->m_lambda[i] = -particles->m_C[i];
			glm::vec3 gradientC_i = (1.f / m_rest_density) * SPHKernel::Poly6_W_Gradient(glm::vec3(0), 0, effective_radius);
			float gradientC_sum = glm::dot(gradientC_i, gradientC_i);

			for (int j = 0; j < neighbors.size(); ++j)
			{
				glm::vec3 diff = particles->m_predict_positions[i] - particles->m_predict_positions[neighbors[j]];
				float distance = glm::distance(particles->m_predict_positions[i], particles->m_predict_positions[neighbors[j]]);

				glm::vec3 gradientC_j = (1.f / m_rest_density) * SPHKernel::Poly6_W_Gradient(diff, distance, effective_radius);

				float dot_value = glm::dot(gradientC_j, gradientC_j);

				//gradientC_i += gradientC_j;
				gradientC_sum += dot_value;
			}
			std::cout << "gradientC_sum: " << gradientC_sum << std::endl;
			//float dot_value = glm::dot(gradientC_i, gradientC_i);
			//gradientC_sum += dot_value;
			particles->m_lambda[i] /= gradientC_sum + epsilon;
		}
	}

}

void Simulation::ComputeSPHParticlesCorrection(float effective_radius, float dt)
{
	ParticleSet* const particles = m_particle_system->getSPHParticles();

	#pragma omp parallel default(shared) num_threads(8)
	{
		#pragma omp for schedule(dynamic)
		for (int i = 0; i < particles->m_size; ++i)
		{
			auto neighbors = m_neighbor_searcher->FetchNeighbors(static_cast<size_t>(i));

			for (int j = 0; j < neighbors.size(); ++j)
			{
				glm::vec3 diff = particles->m_predict_positions[i] - particles->m_predict_positions[neighbors[j]];
				float distance = glm::distance(particles->m_predict_positions[i], particles->m_predict_positions[neighbors[j]]);
				
				// Artificial pressure
				double scorr = -0.1;
				double x = SPHKernel::Poly6_W(distance, effective_radius) / 
					SPHKernel::Poly6_W(0.3f * effective_radius, effective_radius);
				x = glm::pow(x, 4);
				scorr = scorr * x * dt;

				glm::vec3 result = (1.f / m_rest_density) *
					(particles->m_lambda[i] + particles->m_lambda[neighbors[j]] + static_cast<float>(scorr)) *
					SPHKernel::Poly6_W_Gradient(diff, distance, effective_radius);

				particles->m_new_positions[i] += result;
			}
		}
	}
}

void Simulation::UpdatePredictPosition()
{
	ParticleSet* const particles = m_particle_system->getSPHParticles();
	for (size_t i = 0; i < particles->m_size; ++i)
	{
		particles->m_predict_positions[i] = particles->m_new_positions[i];
	}
}

void Simulation::CollisionDetection(float dt)
{	
	// Clean previous CD result
	for (auto vec : m_collision_table)
	{
		vec.clear();
	}

	ParticleSet* const particles = m_particle_system->getSPHParticles();
	for (size_t i = 0; i < particles->m_size; ++i)
	{
		for (size_t j = 0; j < m_colliders.size(); ++j)
		{
			if (particles->TestCollision(i, m_colliders[j]))
				particles->OnCollision(i, m_colliders[j], dt);
		}
	}

	// TODO: Change to m_rigidbodies[i], m_rigidbodies[j]
	for (size_t i = 0; i < m_colliders.size(); ++i)
	{
		for (size_t j = i+1; j < m_colliders.size(); ++j)
		{
			/* Record result if there's contact between two objects */
			if (m_colliders[i]->TestCollision(m_colliders[j]))
				m_collision_table[i].push_back(m_colliders[j]);
		}
	}
}

/*
 * This function handles collision response for specific collision pairs.
 * (Particle v.s. Static plane),  (Particle v.s. Static AABB), (Particle v.s Static OBB)
 */
void Simulation::HandleCollisionResponse()
{
}

/*
 * In jelly simulation the only collision is particle hitting the plane or other static BBs.
 * 
*/
void Simulation::GenerateCollisionConstraint()
{
	/* 
	for(pairs : collision_pairs)
	{
		// Generate collision constraint
	}
	*/
}

bool Simulation::ProjectConstraints(const float &dt)
{
	m_solver->SolveConstraints(dt, m_static_constraints, m_collision_constraints);

	return true;
}

void Simulation::AddCollisionConstraint(Constraint* constraint)
{
	m_collision_constraints.push_back(constraint);
}

void Simulation::ApplySolverResults(float dt)
{
	m_particle_system->getSPHParticles()->Update(dt);
	m_particle_system->Update();
}
