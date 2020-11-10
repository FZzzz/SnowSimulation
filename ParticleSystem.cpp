#include "ParticleSystem.h"
#include <cuda_runtime.h>
#include <chrono>
#include "imgui/imgui.h"
#include "cuda_tool.cuh"
#include <iostream>

ParticleSystem::ParticleSystem() :
	m_sph_particles(nullptr),
	m_dem_particles(nullptr),
	m_boundary_particles(nullptr),
	//m_particles(nullptr),
	m_particle_radius(5.f),
	m_hottest_temperature(10.f),
	m_coolest_temperature(-10.f)
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
	m_sph_particles->m_maximum_connection = m_maximum_connection;
	m_dem_particles->m_maximum_connection = m_maximum_connection;
	GenerateGLBuffers();
	SetupCUDAMemory();
	UpdateGLBUfferData();
	//RegisterCUDAVBO();
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
	if (m_sph_particles != nullptr)
	{
		m_sph_particles->ReleaseDeviceData();
		cudaGraphicsUnregisterResource(m_sph_cuda_vbo_resource[0]);
		cudaGraphicsUnregisterResource(m_sph_cuda_vbo_resource[1]);
		delete m_sph_particles;
	}

	if (m_dem_particles != nullptr)
	{
		m_dem_particles->ReleaseDeviceData();
		cudaGraphicsUnregisterResource(m_dem_cuda_vbo_resource[0]);
		cudaGraphicsUnregisterResource(m_dem_cuda_vbo_resource[1]);
		delete m_dem_particles;
	}

	if (m_boundary_particles != nullptr)
	{
		m_boundary_particles->ReleaseDeviceData();
		cudaGraphicsUnregisterResource(m_boundary_cuda_vbo_resource);
		delete m_boundary_particles;
	}

	if (m_buffer_device_data != nullptr)
	{
		cudaFree(m_buffer_device_data->m_d_positions);
		cudaFree(m_buffer_device_data->m_d_predict_positions);
		cudaFree(m_buffer_device_data->m_d_new_positions);
		cudaFree(m_buffer_device_data->m_d_velocity);
		cudaFree(m_buffer_device_data->m_d_force);
		cudaFree(m_buffer_device_data->m_d_correction);
		cudaFree(m_buffer_device_data->m_d_mass);
		cudaFree(m_buffer_device_data->m_d_massInv);
		cudaFree(m_buffer_device_data->m_d_density);
		cudaFree(m_buffer_device_data->m_d_C);
		cudaFree(m_buffer_device_data->m_d_lambda);
		cudaFree(m_buffer_device_data->m_d_T);
		cudaFree(m_buffer_device_data->m_d_new_T);
		cudaFree(m_buffer_device_data->m_d_predicate);
		cudaFree(m_buffer_device_data->m_d_scan_index);
		cudaFree(m_buffer_device_data->m_d_new_end);
		cudaFree(m_buffer_device_data->m_d_contrib);
		cudaFree(m_buffer_device_data->m_d_connect_record);
		cudaFree(m_buffer_device_data->m_d_iter_end);
		cudaFree(m_buffer_device_data->m_d_connect_length);

		delete m_buffer_device_data;
	}

}

ParticleSet* ParticleSystem::AllocateSPHParticles(size_t n, float particle_mass)
{
	m_sph_particles = new ParticleSet(n, particle_mass);
	return m_sph_particles;
}

ParticleSet* ParticleSystem::AllocateDEMParticles(size_t n, float particle_mass)
{
	m_dem_particles = new ParticleSet(n, particle_mass);
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
	uint id_count = 1;
	// Fluid paritcles
	{
		size_t n = m_sph_particles->m_full_size;

		std::vector<float> contrib(n, 1.0f);
		std::vector<glm::vec3> vec3_zeros;
		vec3_zeros.resize(n, glm::vec3(0, 0, 0));
		
		glm::vec3* positions = m_sph_particles->m_positions.data();
		glm::vec3* predict_positions = m_sph_particles->m_predict_positions.data();
		glm::vec3* new_positions = m_sph_particles->m_new_positions.data();

		glm::vec3* velocity = m_sph_particles->m_velocity.data();
		glm::vec3* force = m_sph_particles->m_force.data();

		float* mass = m_sph_particles->m_mass.data();
		float* massInv = m_sph_particles->m_massInv.data();
		float* density = m_sph_particles->m_density.data();
		float* C = m_sph_particles->m_C.data();
		float* lambda = m_sph_particles->m_lambda.data();
				
		// m_positions is the map/unmap target. we don't setup right here
		// Allocate memory spaces
		cudaMalloc(
		(void**)&(m_sph_particles->m_device_data.m_d_predict_positions),
			n * sizeof(float3)
			);
		cudaMalloc(
		(void**)&(m_sph_particles->m_device_data.m_d_new_positions),
			n * sizeof(float3)
			);
		cudaMalloc(
		(void**)&(m_sph_particles->m_device_data.m_d_velocity),
			n * sizeof(float3)
			);
		cudaMalloc(
		(void**)&(m_sph_particles->m_device_data.m_d_force),
			n * sizeof(float3)
			);

		cudaMalloc(
			(void**)&(m_sph_particles->m_device_data.m_d_correction),
			n * sizeof(float3)
		);

		cudaMalloc(
		(void**)&(m_sph_particles->m_device_data.m_d_mass),
			n * sizeof(float)
			);
		cudaMalloc(
		(void**)&(m_sph_particles->m_device_data.m_d_massInv),
			n * sizeof(float)
			);
		cudaMalloc(
		(void**)&(m_sph_particles->m_device_data.m_d_density),
			n * sizeof(float)
			);
		cudaMalloc(
		(void**)&(m_sph_particles->m_device_data.m_d_C),
			n * sizeof(float)
			);
		cudaMalloc(
		(void**)&(m_sph_particles->m_device_data.m_d_lambda),
			n * sizeof(float)
			);

		cudaMalloc(
			(void**)&(m_sph_particles->m_device_data.m_d_new_T),
			n * sizeof(float)
		);

		cudaMalloc(
			(void**)&(m_sph_particles->m_device_data.m_d_predicate),
			n * sizeof(uint)
		);

		cudaMalloc(
			(void**)&(m_sph_particles->m_device_data.m_d_scan_index),
			n * sizeof(uint)
		);
		
		cudaMalloc(
			(void**)&(m_sph_particles->m_device_data.m_d_new_end),
			sizeof(uint)
		);

		cudaMalloc(
			(void**)&(m_sph_particles->m_device_data.m_d_contrib),
			n * sizeof(float)
		);

		cudaMalloc(
			(void**)&(m_sph_particles->m_device_data.m_d_mass_scale),
			n * sizeof(float)
		);

		cudaMalloc(
			(void**)&(m_sph_particles->m_device_data.m_d_trackId),
			n * sizeof(uint)
		);

		cudaMalloc(
			(void**)&(m_sph_particles->m_device_data.m_d_connect_record),
			m_maximum_connection* n * sizeof(uint)
		);
		cudaMalloc(
			(void**)&(m_sph_particles->m_device_data.m_d_iter_end),
			n * sizeof(uint)
		);
		cudaMalloc(
			(void**)&(m_sph_particles->m_device_data.m_d_connect_length),
			m_maximum_connection* n * sizeof(float)
		);
		cudaMalloc(
			(void**)&(m_sph_particles->m_device_data.m_d_new_index),
			n * sizeof(uint)
		);


		// Set value
		cudaMemcpy(
		(void*)m_sph_particles->m_device_data.m_d_predict_positions,
			(void*)predict_positions,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_sph_particles->m_device_data.m_d_new_positions,
			(void*)new_positions,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_sph_particles->m_device_data.m_d_velocity,
			(void*)velocity,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_sph_particles->m_device_data.m_d_force,
			(void*)force,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
			(void*)m_sph_particles->m_device_data.m_d_correction,
			(void*)vec3_zeros.data(),
			n * sizeof(float3),
			cudaMemcpyHostToDevice
		);

		cudaMemcpy(
		(void*)m_sph_particles->m_device_data.m_d_mass,
			(void*)mass,
			n * sizeof(float),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_sph_particles->m_device_data.m_d_massInv,
			(void*)massInv,
			n * sizeof(float),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_sph_particles->m_device_data.m_d_density,
			(void*)density,
			n * sizeof(float),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_sph_particles->m_device_data.m_d_C,
			(void*)C,
			n * sizeof(float),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_sph_particles->m_device_data.m_d_lambda,
			(void*)lambda,
			n * sizeof(float),
			cudaMemcpyHostToDevice
			);
		
		cudaMemcpy(
			(void*)m_sph_particles->m_device_data.m_d_new_T,
			(void*)m_sph_particles->m_temperature.data(),
			n * sizeof(float),
			cudaMemcpyHostToDevice
		);
		
		cudaMemcpy(
			(void*)m_sph_particles->m_device_data.m_d_contrib,
			(void*)contrib.data(),
			m_sph_particles->m_size * sizeof(float),
			cudaMemcpyHostToDevice
		);

		cudaMemcpy(
		(void*)m_sph_particles->m_device_data.m_d_mass_scale,
			(void*)contrib.data(),
			n * sizeof(float),
			cudaMemcpyHostToDevice
		);

		std::vector<uint> id_vec(n, 0u);
		for (uint i = 0; i < m_sph_particles->m_size; ++i) id_vec[i] = id_count, id_count++;

		std::vector<uint> iter_end(n);
		for (uint i = 0; i < n; ++i) iter_end[i] = m_maximum_connection * i;

		cudaMemcpy(
			(void*)m_sph_particles->m_device_data.m_d_trackId,
			(void*)id_vec.data(),
			n * sizeof(uint),
			cudaMemcpyHostToDevice
		);

		cudaMemcpy(
			(void*)m_sph_particles->m_device_data.m_d_new_end,
			(void*)&m_sph_particles->m_size,
			sizeof(uint),
			cudaMemcpyHostToDevice
		);
				
		//cudaMemset((void*)m_sph_particles->m_device_data.m_d_trackId, 0, n);

		cuda_tool_fill_uint(m_sph_particles->m_device_data.m_d_predicate, 0, m_sph_particles->m_size, 1u);
		cuda_tool_fill_uint(m_sph_particles->m_device_data.m_d_predicate, m_sph_particles->m_size, m_sph_particles->m_full_size, 0u);
		cuda_tool_fill_uint(m_sph_particles->m_device_data.m_d_scan_index, 0, m_sph_particles->m_full_size, 0u);

		// set initial value of conect record to MAXIMUM of uint
		// There's is no need to use in SPH, but we still give it to prevent potential bugs caused by compact_and_clean()
		cuda_tool_fill_uint(m_sph_particles->m_device_data.m_d_connect_record, 0, m_maximum_connection * n, UINT_MAX);
		cuda_tool_fill_float(m_sph_particles->m_device_data.m_d_connect_length, 0, m_maximum_connection * n, 0.f);

		cudaMemcpy(
			(void*)m_sph_particles->m_device_data.m_d_iter_end,
			(void*)iter_end.data(),
			n * sizeof(uint),
			cudaMemcpyHostToDevice
		);


	}// end of fluid particle settings

	// DEM particles
	if(m_dem_particles != nullptr)
	{
		
		size_t n = m_dem_particles->m_full_size;

		std::vector<float> contrib(n, 1.0f);
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

		// m_positions is the map/unmap target. we don't setup right here
		// Allocate memory spaces
		cudaMalloc(
		(void**)&(m_dem_particles->m_device_data.m_d_predict_positions),
			n * sizeof(float3)
			);
		cudaMalloc(
		(void**)&(m_dem_particles->m_device_data.m_d_new_positions),
			n * sizeof(float3)
			);
		cudaMalloc(
		(void**)&(m_dem_particles->m_device_data.m_d_velocity),
			n * sizeof(float3)
			);
		cudaMalloc(
		(void**)&(m_dem_particles->m_device_data.m_d_force),
			n * sizeof(float3)
			);
		cudaMalloc(
			(void**)&(m_dem_particles->m_device_data.m_d_correction),
			n * sizeof(float3)
		);
		cudaMalloc(
		(void**)&(m_dem_particles->m_device_data.m_d_mass),
			n * sizeof(float)
			);
		cudaMalloc(
		(void**)&(m_dem_particles->m_device_data.m_d_massInv),
			n * sizeof(float)
			);
		cudaMalloc(
		(void**)&(m_dem_particles->m_device_data.m_d_density),
			n * sizeof(float)
			);
		cudaMalloc(
		(void**)&(m_dem_particles->m_device_data.m_d_C),
			n * sizeof(float)
			);
		cudaMalloc(
		(void**)&(m_dem_particles->m_device_data.m_d_lambda),
			n * sizeof(float)
			);
		cudaMalloc(
			(void**)&(m_dem_particles->m_device_data.m_d_new_T),
			n * sizeof(float)
		);

		cudaMalloc(
			(void**)&(m_dem_particles->m_device_data.m_d_predicate),
			n * sizeof(uint)
		);

		cudaMalloc(
			(void**)&(m_dem_particles->m_device_data.m_d_scan_index),
			n * sizeof(uint)
		);

		cudaMalloc(
			(void**)&(m_dem_particles->m_device_data.m_d_new_end),
			sizeof(uint)
		);

		cudaMalloc(
			(void**)&(m_dem_particles->m_device_data.m_d_contrib),
			n * sizeof(float)
		);

		cudaMalloc(
			(void**)&(m_dem_particles->m_device_data.m_d_mass_scale),
			n * sizeof(float)
		);

		cudaMalloc(
			(void**)&(m_dem_particles->m_device_data.m_d_trackId),
			n * sizeof(float)
		);

		// refreezing parameters
		cudaMalloc(
			(void**)&(m_dem_particles->m_device_data.m_d_connect_record),
			m_maximum_connection* n * sizeof(uint)
		);
		cudaMalloc(
			(void**)&(m_dem_particles->m_device_data.m_d_iter_end),
			n * sizeof(uint)
		);
		cudaMalloc(
			(void**)&(m_dem_particles->m_device_data.m_d_connect_length),
			m_maximum_connection* n * sizeof(float)
		);

		cudaMalloc(
			(void**)&(m_dem_particles->m_device_data.m_d_new_index),
			n * sizeof(uint)
		);

		// Set value
		cudaMemcpy(
		(void*)m_dem_particles->m_device_data.m_d_predict_positions,
			(void*)predict_positions,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_dem_particles->m_device_data.m_d_new_positions,
			(void*)new_positions,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_dem_particles->m_device_data.m_d_velocity,
			(void*)velocity,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_dem_particles->m_device_data.m_d_force,
			(void*)force,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_dem_particles->m_device_data.m_d_mass,
			(void*)mass,
			n * sizeof(float),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_dem_particles->m_device_data.m_d_massInv,
			(void*)massInv,
			n * sizeof(float),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_dem_particles->m_device_data.m_d_density,
			(void*)density,
			n * sizeof(float),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_dem_particles->m_device_data.m_d_C,
			(void*)C,
			n * sizeof(float),
			cudaMemcpyHostToDevice
			);
		cudaMemcpy(
		(void*)m_dem_particles->m_device_data.m_d_lambda,
			(void*)lambda,
			n * sizeof(float),
			cudaMemcpyHostToDevice
			);

		cudaMemcpy(
		(void*)m_dem_particles->m_device_data.m_d_correction,
			(void*)velocity,
			n * sizeof(float3),
			cudaMemcpyHostToDevice
			);
		
		cudaMemcpy(
			(void*)m_dem_particles->m_device_data.m_d_new_T,
			(void*)m_dem_particles->m_temperature.data(),
			n * sizeof(float),
			cudaMemcpyHostToDevice
		);

		cudaMemcpy(
			(void*)m_dem_particles->m_device_data.m_d_contrib,
			(void*)contrib.data(),
			m_dem_particles->m_size * sizeof(float),
			cudaMemcpyHostToDevice
		);

		cudaMemcpy(
			(void*)m_dem_particles->m_device_data.m_d_mass_scale,
			(void*)contrib.data(),
			n * sizeof(float),
			cudaMemcpyHostToDevice
			);

		cudaMemcpy(
			(void*)m_dem_particles->m_device_data.m_d_new_end,
			(void*)&m_dem_particles->m_size,
			sizeof(uint),
			cudaMemcpyHostToDevice
		);


		std::vector<uint> id_vec(n, 0u);
		for (uint i = 0; i < m_dem_particles->m_size; ++i) id_vec[i] = id_count, id_count++;

		std::vector<uint> iter_end(n);
		for (uint i = 0; i < n; ++i) iter_end[i] = m_maximum_connection * i;

		cudaMemcpy(
			(void*)m_dem_particles->m_device_data.m_d_trackId,
			(void*)id_vec.data(),
			n * sizeof(uint),
			cudaMemcpyHostToDevice
		);

		cuda_tool_fill_uint(m_dem_particles->m_device_data.m_d_predicate, 0, m_dem_particles->m_size, 1u);
		cuda_tool_fill_uint(m_dem_particles->m_device_data.m_d_predicate, m_dem_particles->m_size, m_dem_particles->m_full_size, 0u);
		cuda_tool_fill_uint(m_dem_particles->m_device_data.m_d_scan_index, 0, m_dem_particles->m_full_size, 0u);

		// set initial value of conect record to MAXIMUM of uint
		// There's is no need to use in SPH, but we still give it to prevent potential bugs caused by compact_and_clean()
		cuda_tool_fill_uint(m_dem_particles->m_device_data.m_d_connect_record, 0, m_maximum_connection * n, UINT_MAX);
		cuda_tool_fill_float(m_dem_particles->m_device_data.m_d_connect_length, 0, m_maximum_connection * n, 0.f);

		cudaMemcpy(
			(void*)m_dem_particles->m_device_data.m_d_iter_end,
			(void*)iter_end.data(),
			n * sizeof(uint),
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
		
		std::vector<float> contrib(m_boundary_particles->m_size, 1.0f);
		float* mass = m_boundary_particles->m_mass.data();
		float* massInv = m_boundary_particles->m_massInv.data();
		float* density = m_boundary_particles->m_density.data();
		float* C = m_boundary_particles->m_C.data();
		float* lambda = m_boundary_particles->m_lambda.data();
		float* volume = m_boundary_particles->m_volume.data();

		size_t n = m_boundary_particles->m_size;

		// Positions will be updated with map/unmap 

		cudaMalloc(
			(void**)&(m_boundary_particles->m_device_data.m_d_mass),
			n * sizeof(float)
		);
		cudaMalloc(
			(void**)&(m_boundary_particles->m_device_data.m_d_massInv),
			n * sizeof(float)
		);
		cudaMalloc(
			(void**)&(m_boundary_particles->m_device_data.m_d_density),
			n * sizeof(float)
		);
		cudaMalloc(
			(void**)&(m_boundary_particles->m_device_data.m_d_C),
			n * sizeof(float)
		);
		cudaMalloc(
			(void**)&(m_boundary_particles->m_device_data.m_d_lambda),
			n * sizeof(float)
		);
		cudaMalloc(
			(void**)&(m_boundary_particles->m_device_data.m_d_volume),
			n * sizeof(float)
		);

		cudaMalloc(
			(void**)&(m_boundary_particles->m_device_data.m_d_contrib),
			n * sizeof(float)
		);

		cudaMalloc(
		(void**)&(m_boundary_particles->m_device_data.m_d_mass_scale),
			n * sizeof(float)
			);

		// Copy data
		cudaMemcpy(
			(void*)m_boundary_particles->m_device_data.m_d_mass,
			(void*)mass,
			n * sizeof(float),
			cudaMemcpyHostToDevice
		);
		cudaMemcpy(
			(void*)m_boundary_particles->m_device_data.m_d_massInv,
			(void*)massInv,
			n * sizeof(float),
			cudaMemcpyHostToDevice
		);
		cudaMemcpy(
			(void*)m_boundary_particles->m_device_data.m_d_density,
			(void*)density,
			n * sizeof(float),
			cudaMemcpyHostToDevice
		);
		cudaMemcpy(
			(void*)m_boundary_particles->m_device_data.m_d_C,
			(void*)C,
			n * sizeof(float),
			cudaMemcpyHostToDevice
		);
		cudaMemcpy(
			(void*)m_boundary_particles->m_device_data.m_d_lambda,
			(void*)lambda,
			n * sizeof(float),
			cudaMemcpyHostToDevice
		);
		cudaMemcpy(
			(void*)m_boundary_particles->m_device_data.m_d_volume,
			(void*)volume,
			n * sizeof(float),
			cudaMemcpyHostToDevice
		);

		cudaMemcpy(
			(void*)m_boundary_particles->m_device_data.m_d_contrib,
			(void*)contrib.data(),
			n * sizeof(float),
			cudaMemcpyHostToDevice
		);

		cudaMemcpy(
		(void*)m_boundary_particles->m_device_data.m_d_mass_scale,
			(void*)contrib.data(),
			n * sizeof(float),
			cudaMemcpyHostToDevice
			);


	}// end of boundary particle settings

	// tmp buffer allocation
	{
		uint n = m_sph_particles->m_full_size;
		m_buffer_device_data = new ParticleDeviceData();
		
		cudaMalloc(
			(void**)&(m_buffer_device_data->m_d_positions),
			n * sizeof(float3)
		);

		cudaMalloc(
			(void**)&(m_buffer_device_data->m_d_predict_positions),
			n * sizeof(float3)
		);
		cudaMalloc(
			(void**)&(m_buffer_device_data->m_d_new_positions),
			n * sizeof(float3)
		);
		cudaMalloc(
			(void**)&(m_buffer_device_data->m_d_velocity),
			n * sizeof(float3)
		);
		cudaMalloc(
			(void**)&(m_buffer_device_data->m_d_force),
			n * sizeof(float3)
		);

		cudaMalloc(
			(void**)&(m_buffer_device_data->m_d_correction),
			n * sizeof(float3)
		);

		cudaMalloc(
			(void**)&(m_buffer_device_data->m_d_mass),
			n * sizeof(float)
		);
		cudaMalloc(
			(void**)&(m_buffer_device_data->m_d_massInv),
			n * sizeof(float)
		);
		cudaMalloc(
			(void**)&(m_buffer_device_data->m_d_density),
			n * sizeof(float)
		);
		cudaMalloc(
			(void**)&(m_buffer_device_data->m_d_C),
			n * sizeof(float)
		);
		cudaMalloc(
			(void**)&(m_buffer_device_data->m_d_lambda),
			n * sizeof(float)
		);

		cudaMalloc(
			(void**)&(m_buffer_device_data->m_d_T),
			n * sizeof(float)
		);

		cudaMalloc(
			(void**)&(m_buffer_device_data->m_d_new_T),
			n * sizeof(float)
		);

		cudaMalloc(
			(void**)&(m_buffer_device_data->m_d_predicate),
			n * sizeof(uint)
		);

		cudaMalloc(
			(void**)&(m_buffer_device_data->m_d_scan_index),
			n * sizeof(uint)
		);

		cudaMalloc(
			(void**)&(m_buffer_device_data->m_d_new_end),
			sizeof(uint)
		);

		cudaMalloc(
			(void**)&(m_buffer_device_data->m_d_contrib),
			n * sizeof(float)
		);

		cudaMalloc(
			(void**)&(m_buffer_device_data->m_d_mass_scale),
			n * sizeof(float)
			);
		
		cudaMalloc(
			(void**)&(m_buffer_device_data->m_d_trackId),
			n * sizeof(uint)
		);

		// refreezing parameters
		cudaMalloc(
			(void**)&(m_buffer_device_data->m_d_connect_record),
			(size_t)m_maximum_connection* n * sizeof(uint)
		);
		cudaMalloc(
		(void**)&(m_buffer_device_data->m_d_iter_end),
			n * sizeof(uint)
		);
		cudaMalloc(
			(void**)&(m_buffer_device_data->m_d_connect_length),
			(size_t)m_maximum_connection* n * sizeof(float)
		);

		cudaMalloc(
			(void**)&(m_buffer_device_data->m_d_new_index),
			n * sizeof(uint)
		);

	}// end of tmp buffer allocation

}

void ParticleSystem::RegisterCUDAVBO()
{
	if (m_sph_particles)
	{
		cudaGraphicsGLRegisterBuffer(&m_sph_cuda_vbo_resource[0], m_sph_vbo[0], cudaGraphicsMapFlagsNone);
		cudaGraphicsGLRegisterBuffer(&m_sph_cuda_vbo_resource[1], m_sph_vbo[1], cudaGraphicsMapFlagsNone);
	}
	if (m_dem_particles)
	{
		cudaGraphicsGLRegisterBuffer(&m_dem_cuda_vbo_resource[0], m_dem_vbo[0], cudaGraphicsMapFlagsNone);
		cudaGraphicsGLRegisterBuffer(&m_dem_cuda_vbo_resource[1], m_dem_vbo[1], cudaGraphicsMapFlagsNone);
	}
	if(m_boundary_particles)
		cudaGraphicsGLRegisterBuffer(&m_boundary_cuda_vbo_resource, m_boundary_vbo, cudaGraphicsMapFlagsNone);
}

void ParticleSystem::GenerateGLBuffers()
{
	// fluid particles
	glGenVertexArrays(1, &m_sph_vao);
	glGenBuffers(2, m_sph_vbo);
	glGenBuffers(1, &m_sph_ebo); // NOTICE: not using

	glGenVertexArrays(1, &m_dem_vao);
	glGenBuffers(2, m_dem_vbo);
	glGenBuffers(1, &m_dem_ebo); // NOTICE: not using

	// boundary paritcles
	glGenVertexArrays(1, &m_boundary_vao);
	glGenBuffers(1, &m_boundary_vbo);
	glGenBuffers(1, &m_boundary_ebo); // NOTICE: not using

}

void ParticleSystem::UpdateGLBUfferData()
{
	if (m_sph_particles->m_size == 0)
		return;

	// Fluid particle GL buffer data
	glBindVertexArray(m_sph_vao);

	glBindBuffer(GL_ARRAY_BUFFER, m_sph_vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * m_sph_particles->m_full_size,
		m_sph_particles->m_positions.data(), GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// bind temperature buffer
	glBindBuffer(GL_ARRAY_BUFFER, m_sph_vbo[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * m_sph_particles->m_full_size,
		m_sph_particles->m_temperature.data(), GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	if (m_dem_particles != nullptr && m_dem_particles->m_size != 0)
	{
		size_t n = m_dem_particles->m_full_size;

		glBindVertexArray(m_dem_vao);

		glBindBuffer(GL_ARRAY_BUFFER, m_dem_vbo[0]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * n,
			m_dem_particles->m_positions.data(), GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, 0); // unbind vbo[0]
		
		// bind temperature buffer
		glBindBuffer(GL_ARRAY_BUFFER, m_dem_vbo[1]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * n,
			m_dem_particles->m_temperature.data(), GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, 0); // unbind vbo[0]
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
