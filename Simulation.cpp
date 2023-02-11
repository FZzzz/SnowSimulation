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
#include <algorithm>
#include "cuda_simulation.cuh"

Simulation::Simulation()
	: m_solver(nullptr), m_particle_system(nullptr), m_neighbor_searcher(nullptr),
	m_initialized(false), m_world_desc(SimWorldDesc(-9.8f, 0.f)), m_pause(true),
	m_first_frame(true),
	m_rest_density(0.8f),
	m_iterations(1),
	m_clip_length(1000)
{
	m_scene_params.current_time = 0.f;
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

	//m_sim_params = new SimParams();

	/*
	// Debug
	glm::vec3 fluid_half_extends = glm::vec3(0.75f, 0.05f, 0.05f);
	glm::vec3 snow_half_extends = glm::vec3(0.75f, 0.05f, 0.25f);
	glm::vec3 fluid_origin = glm::vec3(-0.0f, 0.4f, 0.0f);
	glm::vec3 snow_origin = glm::vec3(0.0f, 0.0623f, 0.0f);

	const float sph_temperature = 50.f;
	const float dem_temperature = -15.f;

	m_scene_params.fluid_start_time = 0.0f;
	m_scene_params.solid_start_time = 0.0f;
	*/
	
	/*
	//plain snow drop
	glm::vec3 fluid_half_extends = glm::vec3(0.1f, 0.1f, 0.1f);
	glm::vec3 snow_half_extends = glm::vec3(0.25f, 0.25f, 0.25f);
	glm::vec3 fluid_origin = glm::vec3(-0.0f, 100.f, 0.0f); // invisible in this scene
	glm::vec3 snow_origin = glm::vec3(0.0f, 0.63f, 0.0f);

	const float sph_temperature = 25.f;
	const float dem_temperature = -10.f;

	m_scene_params.fluid_start_time = 10000.0f;
	m_scene_params.solid_start_time = 0.0f;
	*/

	/*
	// water drop melt snow 
	glm::vec3 fluid_half_extends = glm::vec3(0.98f, 0.15f, 0.98f);
	glm::vec3 snow_half_extends = glm::vec3(0.25f, 0.25f, 0.25f);
	glm::vec3 fluid_origin = glm::vec3(-0.0f, 0.151f, 0.0f); // invisible in this scene
	glm::vec3 snow_origin = glm::vec3(0.0f, 0.75f, 0.0f);

	const float sph_temperature = 1.f;
	const float dem_temperature = -100.f;

	m_scene_params.fluid_start_time = 0.0f;
	m_scene_params.solid_start_time = 0.1f;
	*/

	/*
	//snow melt in water
	glm::vec3 fluid_half_extends = glm::vec3(0.98f, 0.1f, 0.98f);
	glm::vec3 snow_half_extends = glm::vec3(0.25f, 0.25f, 0.25f);
	glm::vec3 fluid_origin = glm::vec3(-0.0f, 0.11f, 0.0f);
	glm::vec3 snow_origin = glm::vec3(0.0f, 0.63f, 0.0f);

	const float sph_temperature = 25.f;
	const float dem_temperature = -10.f;

	m_scene_params.fluid_start_time = 0.0f;
	m_scene_params.solid_start_time = 0.0f;
	*/


	// two set melt	
	/*
	glm::vec3 fluid_half_extends = glm::vec3(0.15f, 0.25f, 0.15f);
	glm::vec3 snow_half_extends = glm::vec3(0.25f, 0.25f, 0.25f);
	glm::vec3 fluid_origin = glm::vec3(0.0f, 5.75f, 0.0f);
	glm::vec3 snow_origin = glm::vec3(0.0f, 0.26f, 0.0f);

	const float sph_temperature = 500.f;
	const float dem_temperature = -15.f;

	m_scene_params.fluid_start_time = 0.3f;
	m_scene_params.solid_start_time = 0.0f;
	*/
	
	/*
	// snow melt (water line)
	glm::vec3 fluid_half_extends = glm::vec3(0.75f, 0.05f, 0.1f);
	glm::vec3 snow_half_extends = glm::vec3(0.8f, 0.125f, 0.8f);
	glm::vec3 fluid_origin = glm::vec3(-0.0f, 0.35f, 0.0f);
	glm::vec3 snow_origin = glm::vec3(0.0f, 0.13f, 0.0f);
	
	const float sph_temperature = 25.f;
	const float dem_temperature = -10.f;
	
	m_scene_params.fluid_start_time = 0.2f;
	m_scene_params.solid_start_time = 0.0f;
	*/

	
	// water drop
	glm::vec3 fluid_half_extends = glm::vec3(0.05f, 1.5f, 0.05f);
	glm::vec3 snow_half_extends = glm::vec3(1.f, 0.125f, 1.f);
	glm::vec3 fluid_origin = glm::vec3(0.0f, 2.5f, 0.0f);
	glm::vec3 snow_origin = glm::vec3(0.f, 0.26f, 0.0f);

	const float sph_temperature = 500.f;
	const float dem_temperature = -5.f;

	m_scene_params.fluid_start_time = 0.0f;
	m_scene_params.solid_start_time = 0.0f;
	

	m_particle_system->setHottestTemperature(sph_temperature + 0.1f * glm::abs(sph_temperature));
	m_particle_system->setCoolestTemperature(dem_temperature - 0.1f * glm::abs(dem_temperature));

	m_neighbor_searcher = std::make_shared<NeighborSearch>(m_particle_system, grid_size);
	m_solver = std::make_shared<ConstraintSolver>(mode);

	/*Set up parameters*/
	SetupSimParams();
	GenerateParticleCube(fluid_half_extends, fluid_origin, 0, true);
	GenerateParticleCube(snow_half_extends, snow_origin, 1, false);

	m_particle_system->setSPHInitialVelocity(glm::vec3(0.f,0,0));
	m_particle_system->setDEMInitialVelocity(glm::vec3(0));

	InitializeTemperature(m_particle_system->getSPHParticles()->m_temperature, sph_temperature);
	InitializeTemperature(m_particle_system->getDEMParticles()->m_temperature, dem_temperature);
	AppendParticleSets();
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

bool Simulation::StepCUDA(float dt)
{
	m_dt = dt;

	if (!m_initialized)
		return false;

	static bool cd_on = true;
	static bool sph_dem_correction = true;
	static bool sph_sph_correction = false;
	static bool compute_temperature = true;
	static bool change_phase = true;
	static bool simulate_freezing = true;
	static bool simulate_melting = true;
	static bool dem_friction = true;
	static bool dem_viscosity = true;

	static bool use_interlink = true;
	static bool dynamic_connections = false;
	static float temperature_variation = 0.f;

	{
		ImGui::Begin("Controls");
		ImGui::SetWindowPos(ImVec2(100, 300), ImGuiSetCond_FirstUseEver);
		ImGui::Checkbox("boundary CD", &cd_on);
		ImGui::Checkbox("SPH-SPH distance correction", &sph_sph_correction);
		ImGui::Checkbox("Phase change", &change_phase);
		ImGui::Checkbox("Freezing", &simulate_freezing);
		ImGui::Checkbox("Melting", &simulate_melting);
		ImGui::Checkbox("DEM friction", &dem_friction);
		ImGui::Checkbox("DEM viscosity", &dem_viscosity);
		ImGui::Checkbox("Enable interlinks", &use_interlink);
		ImGui::Checkbox("Dynamic connection number", &dynamic_connections);
		ImGui::SliderFloat("dT", &temperature_variation, -0.01f, 0.01f, "%.3f");
		ImGui::End();
	}


	if (m_pause)
		return true;


	std::chrono::steady_clock::time_point t1, t2, t3, t4, t5;

	ParticleSet* sph_particles = m_particle_system->getSPHParticles();
	ParticleSet* dem_particles = m_particle_system->getDEMParticles();
	ParticleSet* boundary_particles = m_particle_system->getBoundaryParticles();

	cudaGraphicsResource** sph_vbo_resource = m_particle_system->getSPHCUDAGraphicsResource();
	cudaGraphicsResource** dem_vbo_resource = m_particle_system->getDEMCUDAGraphicsResource();
	cudaGraphicsResource** b_vbo_resource = m_particle_system->getBoundaryCUDAGraphicsResource();
	
	//glm::vec3* positions = sph_particles->m_positions.data();

	unsigned int sph_num_particles = sph_particles->m_size;
	unsigned int dem_num_particles = dem_particles->m_size;
	unsigned int b_num_particles = boundary_particles->m_size;

	// Map vbo to m_d_positinos
	cudaGraphicsMapResources(1, &sph_vbo_resource[0], 0);
	cudaGraphicsMapResources(1, &sph_vbo_resource[1], 0);

	cudaGraphicsMapResources(1, &dem_vbo_resource[0], 0);
	cudaGraphicsMapResources(1, &dem_vbo_resource[1], 0);

	cudaGraphicsMapResources(1, b_vbo_resource, 0);

	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void**)&(sph_particles->m_device_data.m_d_positions), &num_bytes, sph_vbo_resource[0]);
	cudaGraphicsResourceGetMappedPointer((void**)&(sph_particles->m_device_data.m_d_T), &num_bytes, sph_vbo_resource[1]);
	cudaGraphicsResourceGetMappedPointer((void**)&(dem_particles->m_device_data.m_d_positions), &num_bytes, dem_vbo_resource[0]);
	cudaGraphicsResourceGetMappedPointer((void**)&(dem_particles->m_device_data.m_d_T), &num_bytes, dem_vbo_resource[1]);
	cudaGraphicsResourceGetMappedPointer((void**)&(boundary_particles->m_device_data.m_d_positions), &num_bytes, *b_vbo_resource);
	
	t3 = std::chrono::high_resolution_clock::now();

	snow_simulation(
		sph_particles,
		dem_particles,
		boundary_particles,
		m_particle_system->getBufferParticleDeviceData(),
		m_neighbor_searcher->m_d_sph_cell_data,
		m_neighbor_searcher->m_d_dem_cell_data,
		m_neighbor_searcher->m_d_boundary_cell_data,
		m_scene_params,
		dt,
		m_iterations,
		sph_dem_correction,
		sph_sph_correction,
		dem_friction,
		compute_temperature,
		change_phase,
		simulate_freezing,
		simulate_melting,
		dem_viscosity,
		use_interlink,
		dynamic_connections,
		cd_on,
		temperature_variation
		);

	t4 = std::chrono::high_resolution_clock::now();

	{
		ImGui::Begin("Log");
		ImGui::Text("SPH size: %u", sph_particles->m_size);
		ImGui::Text("DEM size: %u", dem_particles->m_size);
		ImGui::End();
	}

	// Unmap CUDA buffer object
	cudaGraphicsUnmapResources(1, &sph_vbo_resource[0], 0);
	cudaGraphicsUnmapResources(1, &sph_vbo_resource[1], 0);
	cudaGraphicsUnmapResources(1, &dem_vbo_resource[0], 0);
	cudaGraphicsUnmapResources(1, &dem_vbo_resource[1], 0);
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
	std::cout << "Iterations: " << iter_count << std::endl;
	m_solver->setSolverIteration(iter_count);
	m_iterations = iter_count;
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
	const float particle_mass = 0.01f;
	const float n_kernel_particles = 20.f;	
	const float dem_sph_ratio = 1.0f;
	// water density = 1000 kg/m^3
	m_rest_density = 1000.f; 
	m_sph_particle_mass = particle_mass;
	m_dem_particle_mass = dem_sph_ratio * particle_mass;

	float effective_radius, particle_radius;
	
	/* Compute parameters from mass and n_particles*/
	m_volume = n_kernel_particles * particle_mass / m_rest_density;
	effective_radius = powf(((3.0f * m_volume) / (4.0f * M_PI)), 1.0f / 3.0f);
	particle_radius = powf((M_PI / (6.0f * n_kernel_particles)), 1.0f / 3.0f) * effective_radius;
	m_particle_system->setParticleRadius(particle_radius);

	/*Maximum interlink connections*/
	m_particle_system->setMaximumConnection(25);

	std::cout << "Particle mass: " << particle_mass << std::endl;
	std::cout << "Effective radius: " << effective_radius << std::endl;
	std::cout << "Particle radius: " << particle_radius << std::endl;
	std::cout << "Maximum connection: " << m_particle_system->getMaximumConnection() << std::endl;

	//if(m_sim_params == nullptr)
		//m_sim_params = new SimParams();

	m_sim_params.gravity = make_float3(0.f, -9.8f, 0.f);
	m_sim_params.global_damping = 1.0;
	m_sim_params.maximum_speed = 10.f;
	m_sim_params.minimum_speed = 0.0f;// 01f * particle_radius * m_dt * m_dt;

	m_sim_params.particle_radius = particle_radius;
	m_sim_params.effective_radius = effective_radius;
	m_sim_params.rest_density = m_rest_density;
	m_sim_params.epsilon = 100.f;
	m_sim_params.pbd_epsilon = 0.01f * particle_radius;
	m_sim_params.kernel_epsilon = 0.01f * effective_radius;
	m_sim_params.grid_size = m_neighbor_searcher->m_grid_size;
	m_sim_params.num_cells = m_neighbor_searcher->m_num_grid_cells;
	m_sim_params.world_origin = make_float3(0, 0, 0);
	m_sim_params.cell_size = make_float3(m_sim_params.effective_radius);
	m_sim_params.boundary_damping = 0.15f;
	
	//coupling coefficients
	//m_sim_params.sph_dem_corr = 0.05f;

	m_sim_params.static_friction = 0.5f;
	m_sim_params.kinematic_friction = 0.35f;

	m_sim_params.scorr_coeff = 0.f;// .1f;
	m_sim_params.sor_coeff = 1.0f * (1.f/4.f);
	m_sim_params.sph_viscosity = 0.01f;
	m_sim_params.dem_viscosity = 0.1f;

	m_sim_params.k_stretch = 0.25f;

	//set up heat conduction constants
	m_sim_params.C_snow = 2090.f;
	m_sim_params.C_water = 4182.f;
	m_sim_params.k_snow = 25.f;
	m_sim_params.k_water = 6.f;
	m_sim_params.freezing_point = 0.f;
	m_sim_params.T_homogeneous = -30.0f;

	m_sim_params.blending_speed = 0.01f;

	// set up sph kernel constants
	m_sim_params.poly6 = (315.0f / (64.0f * M_PI * glm::pow(effective_radius, 9)));
	m_sim_params.poly6_G = (-945.0f / (32.0f * M_PI * glm::pow(effective_radius, 9)));
	m_sim_params.spiky = (15.0f / (M_PI * glm::pow(effective_radius, 6)));
	m_sim_params.spiky_G = (-45.0f / (M_PI * glm::pow(effective_radius, 6)));
	m_sim_params.viscosity_laplacian = (45.f / (M_PI * glm::pow(effective_radius, 6)));
	m_sim_params.scorr_divisor = SPHKernel::Poly6_W(0.3f * effective_radius, effective_radius);

	m_sim_params.maximum_connection = m_particle_system->getMaximumConnection();
	m_sim_params.k_refreezing = 0.75f; // PBD based stiffness (better with XPBD (haven't impelment yet))
	m_sim_params.break_threshold = 1.05f;
	m_particle_system->setParticleRadius(particle_radius);

	set_sim_params(m_sim_params);
}

void Simulation::InitializeBoundaryParticles()
{
	const int thickness = 1;
	const float diameter = 2.f * m_sim_params.particle_radius;
	// number of particles on x,y,z
	int nx, ny, nz;
	//size_t n_particles = 0;
	// fluid cube extends
	glm::vec3 half_extend1(1.0f, diameter * static_cast<float>(thickness)/2.f, 1.0f);
	glm::vec3 half_extend2(diameter * static_cast<float>(thickness)/2.f, 1.0f, 1.0f);
	glm::vec3 half_extend3(1.0f, 1.0f, diameter * static_cast<float>(thickness)/2.f);
	
	// Initialize boundary particles
	ParticleSet* particles = m_particle_system->AllocateBoundaryParticles();

	// Set particle color 
	particles->m_color = glm::vec3(1, 1, 1);

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
	cudaGraphicsResourceGetMappedPointer((void**)&(boundary_particles->m_device_data.m_d_positions), &num_bytes, *vbo_resource);
	//std::cout << "num_bytes " << num_bytes << std::endl;
	// Precompute hash
	calculate_hash(m_neighbor_searcher->m_d_boundary_cell_data, boundary_particles->m_device_data.m_d_positions, num_particles);
	// Sort
	sort_particles(m_neighbor_searcher->m_d_boundary_cell_data, num_particles);
	// Reorder
	reorder_data(
		m_neighbor_searcher->m_d_boundary_cell_data,
		boundary_particles->m_device_data.m_d_positions,
		num_particles,
		m_neighbor_searcher->m_num_grid_cells
	);

	// Compute boundary particle volume
	compute_boundary_volume(
		m_neighbor_searcher->m_d_boundary_cell_data,
		boundary_particles->m_device_data.m_d_mass,
		boundary_particles->m_device_data.m_d_volume,
		num_particles
	);

	// Unmap CUDA buffer object
	cudaGraphicsUnmapResources(1, vbo_resource, 0);
}

void Simulation::InitializeTemperature(std::vector<float>& target, float temperature)
{
	std::fill(target.begin(), target.end(), temperature);
}

void Simulation::GenerateParticleCube(glm::vec3 half_extends, glm::vec3 origin, int opt, bool use_jitter=false)
{
	const float jitter_strength = 0.01f;
	std::srand(time(NULL));
	// diameter of particle
	const float diameter = 2.f * m_sim_params.particle_radius;
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
	
	particles->m_color = (opt == 0) ? glm::vec3(0.7f, 0.7f, 1.f): glm::vec3(0.85f, 0.85f, 0.85f);
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
					float x_jitter = jitter_strength * diameter * static_cast<float>(rand() % 3);
					float y_jitter = jitter_strength * diameter * static_cast<float>(rand() % 3);
					float z_jitter = jitter_strength * diameter * static_cast<float>(rand() % 3);
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

void Simulation::GenerateParticleCube2(glm::vec3 half_extends, glm::vec3 origin, int opt, bool use_jitter = false)
{
	const float jitter_strength = 0.01f;
	std::srand(time(NULL));
	// diameter of particle
	const float diameter = 2.f * m_sim_params.particle_radius;
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
		particles = m_particle_system->AllocateDEMParticles(3 * n_particles, m_dem_particle_mass); // generate 3 cubes
	else
		return;

	particles->m_color = (opt == 0) ? glm::vec3(0.7f, 0.7f, 1.f) : glm::vec3(0.85f, 0.85f, 0.85f);
	std::cout << ((opt == 0) ? "SPH" : "DEM") << " particles: " << n_particles << std::endl;
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
					float x_jitter = jitter_strength * diameter * static_cast<float>(rand() % 3);
					float y_jitter = jitter_strength * diameter * static_cast<float>(rand() % 3);
					float z_jitter = jitter_strength * diameter * static_cast<float>(rand() % 3);
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

	for (int i = -nx; i < nx; ++i)
	{
		for (int j = -ny; j < ny; ++j)
		{
			for (int k = -nz; k < nz; ++k)
			{
				x = origin.x + diameter * static_cast<float>(i) + 0.75f * half_extends.x;
				y = origin.y + diameter * static_cast<float>(j) + 2.f * half_extends.y + 0.1f;
				z = origin.z + diameter * static_cast<float>(k);

				if (use_jitter)
				{
					float x_jitter = jitter_strength * diameter * static_cast<float>(rand() % 3);
					float y_jitter = jitter_strength * diameter * static_cast<float>(rand() % 3);
					float z_jitter = jitter_strength * diameter * static_cast<float>(rand() % 3);
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

	for (int i = -nx; i < nx; ++i)
	{
		for (int j = -ny; j < ny; ++j)
		{
			for (int k = -nz; k < nz; ++k)
			{
				x = origin.x + diameter * static_cast<float>(i) - 0.75f * half_extends.x;
				y = origin.y + diameter * static_cast<float>(j) + 4.f * half_extends.y + 0.2f;
				z = origin.z + diameter * static_cast<float>(k);

				if (use_jitter)
				{
					float x_jitter = jitter_strength * diameter * static_cast<float>(rand() % 3);
					float y_jitter = jitter_strength * diameter * static_cast<float>(rand() % 3);
					float z_jitter = jitter_strength * diameter * static_cast<float>(rand() % 3);
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


void Simulation::AppendParticleSets()
{
	auto sph_particles = m_particle_system->getSPHParticles();
	auto dem_particles = m_particle_system->getDEMParticles();
#ifdef _DEBUG
	assert(sph_particles != nullptr && dem_particles != nullptr);
#endif

	sph_particles->AppendExtraMemory(dem_particles);
	dem_particles->AppendExtraMemory(sph_particles);
}
