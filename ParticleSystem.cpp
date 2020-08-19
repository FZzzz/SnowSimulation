#include "ParticleSystem.h"
#include <cuda_runtime.h>
#include <chrono>
#include "imgui/imgui.h"
#include <iostream>

ParticleSystem::ParticleSystem() :
	m_particles(nullptr),
	m_dem_particles(nullptr),
	m_boundary_particles(nullptr),
	//m_particles(nullptr),
	m_particle_radius(5.f),
	m_vao(-1),
	m_vbo(-1),
	m_ebo(-1)
{
}

ParticleSystem::~ParticleSystem()
{
	//m_particles.clear();
	m_particle_indices.clear();
}

void ParticleSystem::Initialize()
{
	GenerateGLBuffers();
	UpdateGLBUfferData();
}

void ParticleSystem::InitializeCUDA()
{
	GenerateGLBuffers();
	SetupCUDAMemory();
	UpdateGLBUfferData();
	RegisterCUDAVBO();
}

void ParticleSystem::Update()
{
#ifndef _USE_CUDA_
	std::chrono::steady_clock::time_point t1, t2;
	t1 = std::chrono::high_resolution_clock::now();

	UpdateGLBUfferData();

	t2 = std::chrono::high_resolution_clock::now();
	m_update_elased_time = (t2 - t1).count() / 1000000.0;
#endif
}

void ParticleSystem::UpdateCUDA()
{
	std::chrono::steady_clock::time_point t1, t2;
	t1 = std::chrono::high_resolution_clock::now();
	t2 = std::chrono::high_resolution_clock::now();
	m_update_elased_time = (t2 - t1).count() / 1000000.0;
}

void ParticleSystem::Release()
{
	if (m_particles != nullptr)
	{

		// Release fluid particle cuda memory
		//cudaFree(m_particles->m_d_prev_positions);
		//cudaFree(m_particles->m_d_positions);
		cudaFree(m_particles->m_d_predict_positions);
		cudaFree(m_particles->m_d_new_positions);
		cudaFree(m_particles->m_d_prev_velocity);
		cudaFree(m_particles->m_d_velocity);
		cudaFree(m_particles->m_d_new_velocity);
		cudaFree(m_particles->m_d_correction);

		cudaFree(m_particles->m_d_force);
		cudaFree(m_particles->m_d_mass);
		cudaFree(m_particles->m_d_massInv);
		cudaFree(m_particles->m_d_density);
		cudaFree(m_particles->m_d_C);
		cudaFree(m_particles->m_d_lambda);

		cudaGraphicsUnregisterResource(m_cuda_vbo_resource);
		delete m_particles;
	}

	if (m_dem_particles != nullptr)
	{

		// Release fluid particle cuda memory
		//cudaFree(m_particles->m_d_prev_positions);
		//cudaFree(m_particles->m_d_positions);
		cudaFree(m_dem_particles->m_d_predict_positions);
		cudaFree(m_dem_particles->m_d_new_positions);
		cudaFree(m_dem_particles->m_d_prev_velocity);
		cudaFree(m_dem_particles->m_d_velocity);
		cudaFree(m_dem_particles->m_d_new_velocity);
		cudaFree(m_dem_particles->m_d_correction);

		cudaFree(m_dem_particles->m_d_force);
		cudaFree(m_dem_particles->m_d_mass);
		cudaFree(m_dem_particles->m_d_massInv);
		cudaFree(m_dem_particles->m_d_density);
		cudaFree(m_dem_particles->m_d_C);
		cudaFree(m_dem_particles->m_d_lambda);

		cudaGraphicsUnregisterResource(m_dem_cuda_vbo_resource);
		delete m_dem_particles;
	}

	if (m_boundary_particles != nullptr)
	{
		// Release boundary particle cuda memory
		cudaFree(m_boundary_particles->m_d_mass);
		cudaFree(m_boundary_particles->m_d_massInv);
		cudaFree(m_boundary_particles->m_d_density);
		cudaFree(m_boundary_particles->m_d_C);
		cudaFree(m_boundary_particles->m_d_lambda);
		cudaGraphicsUnregisterResource(m_boundary_cuda_vbo_resource);
		delete m_boundary_particles;
	}

}

ParticleSet* ParticleSystem::AllocateParticles(size_t n, float particle_mass)
{
	m_particles = new ParticleSet(n, particle_mass);
	return m_particles;
}

ParticleSet* ParticleSystem::AllocateDEMParticles()
{
	m_dem_particles = new ParticleSet();
	return m_dem_particles;
}

ParticleSet* ParticleSystem::AllocateBoundaryParticles()
{
	m_boundary_particles = new ParticleSet();
	return m_boundary_particles;
}

void ParticleSystem::setParticleRadius(float point_radius)
{
	m_particle_radius = point_radius;
}

void ParticleSystem::SetupCUDAMemory()
{
	// Fluid paritcles
	{
		//glm::vec3* prev_positions = m_particles->m_prev_positions.data();
		glm::vec3* positions = m_particles->m_positions.data();
		glm::vec3* predict_positions = m_particles->m_predict_positions.data();
		glm::vec3* new_positions = m_particles->m_new_positions.data();

		glm::vec3* velocity = m_particles->m_velocity.data();
		glm::vec3* force = m_particles->m_force.data();

		float* mass = m_particles->m_mass.data();
		float* massInv = m_particles->m_massInv.data();
		float* density = m_particles->m_density.data();
		float* C = m_particles->m_C.data();
		float* lambda = m_particles->m_lambda.data();

		size_t n = m_particles->m_size;

		// m_positions is the map/unmap target. we don't setup right here
		// Allocate memory spaces
		cudaMalloc(
		(void**)&(m_particles->m_d_predict_positions),
			n * sizeof(float3)
			);
		cudaMalloc(
		(void**)&(m_particles->m_d_new_positions),
			n * sizeof(float3)
			);
		cudaMalloc(
		(void**)&(m_particles->m_d_prev_velocity),
			n * sizeof(float3)
			);
		cudaMalloc(
		(void**)&(m_particles->m_d_velocity),
			n * sizeof(float3)
			);
		cudaMalloc(
		(void**)&(m_particles->m_d_new_velocity),
			n * sizeof(float3)
			);
		cudaMalloc(
		(void**)&(m_particles->m_d_force),
			n * sizeof(float3)
			);
		cudaMalloc(
		(void**)&(m_particles->m_d_mass),
			n * sizeof(float)
			);
		cudaMalloc(
		(void**)&(m_particles->m_d_massInv),
			n * sizeof(float)
			);
		cudaMalloc(
		(void**)&(m_particles->m_d_density),
			n * sizeof(float)
			);
		cudaMalloc(
		(void**)&(m_particles->m_d_C),
			n * sizeof(float)
			);
		cudaMalloc(
		(void**)&(m_particles->m_d_lambda),
			n * sizeof(float)
			);

		cudaMalloc(
			(void**)&(m_particles->m_d_correction),
			n * sizeof(float3)
		);

		// Set value
		cudaMemcpy(
		(void*)m_particles->m_d_predict_positions,
			(void*)predict_positions,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_particles->m_d_new_positions,
			(void*)new_positions,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_particles->m_d_velocity,
			(void*)velocity,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_particles->m_d_force,
			(void*)force,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_particles->m_d_mass,
			(void*)mass,
			n * sizeof(float),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_particles->m_d_massInv,
			(void*)massInv,
			n * sizeof(float),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_particles->m_d_density,
			(void*)density,
			n * sizeof(float),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_particles->m_d_C,
			(void*)C,
			n * sizeof(float),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_particles->m_d_lambda,
			(void*)lambda,
			n * sizeof(float),
			cudaMemcpyHostToDevice
			);

		cudaMemcpy(
		(void*)m_particles->m_d_prev_velocity,
			(void*)velocity,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_particles->m_d_new_velocity,
			(void*)velocity,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
			);

		cudaMemcpy(
			(void*)m_particles->m_d_correction,
			(void*)velocity,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
		);
	}// end of fluid particle settings

	// DEM particles
	if(m_dem_particles != nullptr)
	{
		glm::vec3* positions = m_dem_particles->m_positions.data();
		glm::vec3* predict_positions = m_dem_particles->m_predict_positions.data();
		glm::vec3* new_positions = m_dem_particles->m_new_positions.data();

		glm::vec3* velocity = m_dem_particles->m_velocity.data();
		glm::vec3* force = m_dem_particles->m_force.data();

		float* mass = m_dem_particles->m_mass.data();
		float* massInv = m_dem_particles->m_massInv.data();
		float* density = m_dem_particles->m_density.data();
		float* C = m_dem_particles->m_C.data();
		float* lambda = m_dem_particles->m_lambda.data();

		size_t n = m_dem_particles->m_size;

		// m_positions is the map/unmap target. we don't setup right here
		// Allocate memory spaces
		cudaMalloc(
		(void**)&(m_dem_particles->m_d_predict_positions),
			n * sizeof(float3)
			);
		cudaMalloc(
		(void**)&(m_dem_particles->m_d_new_positions),
			n * sizeof(float3)
			);
		cudaMalloc(
		(void**)&(m_dem_particles->m_d_prev_velocity),
			n * sizeof(float3)
			);
		cudaMalloc(
		(void**)&(m_dem_particles->m_d_velocity),
			n * sizeof(float3)
			);
		cudaMalloc(
		(void**)&(m_dem_particles->m_d_new_velocity),
			n * sizeof(float3)
			);
		cudaMalloc(
		(void**)&(m_dem_particles->m_d_force),
			n * sizeof(float3)
			);
		cudaMalloc(
		(void**)&(m_dem_particles->m_d_mass),
			n * sizeof(float)
			);
		cudaMalloc(
		(void**)&(m_dem_particles->m_d_massInv),
			n * sizeof(float)
			);
		cudaMalloc(
		(void**)&(m_dem_particles->m_d_density),
			n * sizeof(float)
			);
		cudaMalloc(
		(void**)&(m_dem_particles->m_d_C),
			n * sizeof(float)
			);
		cudaMalloc(
		(void**)&(m_dem_particles->m_d_lambda),
			n * sizeof(float)
			);

		cudaMalloc(
		(void**)&(m_dem_particles->m_d_correction),
			n * sizeof(float3)
			);

		// Set value
		cudaMemcpy(
		(void*)m_dem_particles->m_d_predict_positions,
			(void*)predict_positions,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_dem_particles->m_d_new_positions,
			(void*)new_positions,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_dem_particles->m_d_velocity,
			(void*)velocity,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_dem_particles->m_d_force,
			(void*)force,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_dem_particles->m_d_mass,
			(void*)mass,
			n * sizeof(float),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_dem_particles->m_d_massInv,
			(void*)massInv,
			n * sizeof(float),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_dem_particles->m_d_density,
			(void*)density,
			n * sizeof(float),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_dem_particles->m_d_C,
			(void*)C,
			n * sizeof(float),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_dem_particles->m_d_lambda,
			(void*)lambda,
			n * sizeof(float),
			cudaMemcpyHostToDevice
			);

		cudaMemcpy(
		(void*)m_dem_particles->m_d_prev_velocity,
			(void*)velocity,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_dem_particles->m_d_new_velocity,
			(void*)velocity,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
			);

		cudaMemcpy(
		(void*)m_dem_particles->m_d_correction,
			(void*)velocity,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
			);
	}// end of DEM particle setting

	// Boundary particless
	{
		if (!m_boundary_particles)
		{
			std::cout << "No boundary particles\n";
			return;
		}
		
		float* mass = m_boundary_particles->m_mass.data();
		float* massInv = m_boundary_particles->m_massInv.data();
		float* density = m_boundary_particles->m_density.data();
		float* C = m_boundary_particles->m_C.data();
		float* lambda = m_boundary_particles->m_lambda.data();
		float* volume = m_boundary_particles->m_volume.data();

		size_t n = m_boundary_particles->m_size;

		// Positions will be updated with map/unmap 

		cudaMalloc(
			(void**)&(m_boundary_particles->m_d_mass),
			n * sizeof(float)
		);
		cudaMalloc(
			(void**)&(m_boundary_particles->m_d_massInv),
			n * sizeof(float)
		);
		cudaMalloc(
			(void**)&(m_boundary_particles->m_d_density),
			n * sizeof(float)
		);
		cudaMalloc(
			(void**)&(m_boundary_particles->m_d_C),
			n * sizeof(float)
		);
		cudaMalloc(
			(void**)&(m_boundary_particles->m_d_lambda),
			n * sizeof(float)
		);
		cudaMalloc(
			(void**)&(m_boundary_particles->m_d_volume),
			n * sizeof(float)
		);

		// Copy data
		cudaMemcpy(
			(void*)m_boundary_particles->m_d_mass,
			(void*)mass,
			n * sizeof(float),
			cudaMemcpyHostToDevice
		);
		cudaMemcpy(
			(void*)m_boundary_particles->m_d_massInv,
			(void*)massInv,
			n * sizeof(float),
			cudaMemcpyHostToDevice
		);
		cudaMemcpy(
			(void*)m_boundary_particles->m_d_density,
			(void*)density,
			n * sizeof(float),
			cudaMemcpyHostToDevice
		);
		cudaMemcpy(
			(void*)m_boundary_particles->m_d_C,
			(void*)C,
			n * sizeof(float),
			cudaMemcpyHostToDevice
		);
		cudaMemcpy(
			(void*)m_boundary_particles->m_d_lambda,
			(void*)lambda,
			n * sizeof(float),
			cudaMemcpyHostToDevice
		);
		cudaMemcpy(
			(void*)m_boundary_particles->m_d_volume,
			(void*)volume,
			n * sizeof(float),
			cudaMemcpyHostToDevice
		);
	}// end of boundary particle settings

}

void ParticleSystem::RegisterCUDAVBO()
{
	if(m_particles)
		cudaGraphicsGLRegisterBuffer(&m_cuda_vbo_resource, m_vbo, cudaGraphicsMapFlagsNone);
	if(m_dem_particles)
		cudaGraphicsGLRegisterBuffer(&m_dem_cuda_vbo_resource, m_dem_vbo, cudaGraphicsMapFlagsNone);
	if(m_boundary_particles)
		cudaGraphicsGLRegisterBuffer(&m_boundary_cuda_vbo_resource, m_boundary_vbo, cudaGraphicsMapFlagsNone);
}

void ParticleSystem::GenerateGLBuffers()
{
	// fluid particles
	glGenVertexArrays(1, &m_vao);
	glGenBuffers(1, &m_vbo);
	glGenBuffers(1, &m_ebo); // NOTICE: not using

	glGenVertexArrays(1, &m_dem_vao);
	glGenBuffers(1, &m_dem_vbo);
	glGenBuffers(1, &m_dem_ebo); // NOTICE: not using

	// boundary paritcles
	glGenVertexArrays(1, &m_boundary_vao);
	glGenBuffers(1, &m_boundary_vbo);
	glGenBuffers(1, &m_boundary_ebo); // NOTICE: not using

}

void ParticleSystem::UpdateGLBUfferData()
{
	if (m_particles->m_size == 0)
		return;

	// Fluid particle GL buffer data
	glBindVertexArray(m_vao);

	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * m_particles->m_positions.size(),
		m_particles->m_positions.data(), GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	if (m_dem_particles != nullptr && m_dem_particles->m_size != 0)
	{
		glBindVertexArray(m_dem_vao);

		glBindBuffer(GL_ARRAY_BUFFER, m_dem_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * m_dem_particles->m_positions.size(),
			m_dem_particles->m_positions.data(), GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

	if (!m_boundary_particles || m_boundary_particles->m_size == 0)
		return;

	// Boundary particle GL buffer data
	glBindVertexArray(m_boundary_vao);
	glBindBuffer(GL_ARRAY_BUFFER, m_boundary_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * m_boundary_particles->m_positions.size(),
		m_boundary_particles->m_positions.data(), GL_DYNAMIC_DRAW);
	
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	RegisterCUDAVBO();
}
