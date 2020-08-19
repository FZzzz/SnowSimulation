#ifndef _PARTICLE_SYSTEM_H_
#define _PARTICLE_SYSTEM_H_

#include <GL/glew.h>
#include <vector>
#include <memory>
#include "common.h"
#include "Particle.h"
#include <cuda_gl_interop.h>

class ParticleSystem
{
public:

	ParticleSystem();
	~ParticleSystem();

	void Initialize();
	void InitializeCUDA();
	void Update();
	void UpdateCUDA();
	void Release();

	ParticleSet* AllocateParticles(size_t n, float particle_mass);
	ParticleSet* AllocateDEMParticles();
	ParticleSet* AllocateBoundaryParticles();

	//void setParticles(std::vector<std::shared_ptr<Particle>> particles);
	void setParticleRadius(float particle_radius);

	inline ParticleSet* getParticles() { return m_particles; };
	inline ParticleSet* getDEMParticles() { return m_dem_particles; };
	inline ParticleSet* getBoundaryParticles() { return m_boundary_particles; };
//	inline std::vector<std::shared_ptr<Particle>>& getParticles() { return m_particles; };
	inline GLuint getVAO() { return m_vao; };
	inline GLuint getVBO() { return m_vbo; };
	inline GLuint getEBO() { return m_ebo; };

	inline GLuint getDEMVAO() { return m_dem_vao; };
	inline GLuint getDEMVBO() { return m_dem_vbo; };
	inline GLuint getDEMEBO() { return m_dem_ebo; };

	inline GLuint getBoundaryVAO() { return m_boundary_vao; };
	inline GLuint getBoundaryVBO() { return m_boundary_vbo; };
	inline GLuint getBoundaryEBO() { return m_boundary_ebo; };

	inline cudaGraphicsResource** getCUDAGraphicsResource() { return &m_cuda_vbo_resource; };
	inline cudaGraphicsResource** getBoundaryCUDAGraphicsResource() { return &m_boundary_cuda_vbo_resource; };
	inline cudaGraphicsResource** getDEMCUDAGraphicsResource() { return &m_dem_cuda_vbo_resource; };

	inline double& getUpdateTime() { return m_update_elased_time; };

	inline float getParticleRadius() { return m_particle_radius; };

private:
	
	//inner function
	void SetupCUDAMemory();
	void RegisterCUDAVBO();
	void GenerateGLBuffers();

	void UpdateGLBUfferData();

	ParticleSet* m_particles;
	ParticleSet* m_dem_particles;
	ParticleSet* m_boundary_particles;

	/* OpenGL members */
	//std::vector<std::shared_ptr<Particle>> m_particles;
	std::vector<unsigned int> m_particle_indices;
	std::vector<unsigned int> m_dem_particle_indices;
	std::vector<unsigned int> m_boundary_particle_indices;
	
	float m_particle_radius;

	GLuint m_vao;
	GLuint m_vbo;
	GLuint m_ebo;

	GLuint m_dem_vao;
	GLuint m_dem_vbo;
	GLuint m_dem_ebo;

	GLuint m_boundary_vao;
	GLuint m_boundary_vbo;
	GLuint m_boundary_ebo;
		
	GLuint m_cuda_vbo;
	GLuint m_dem_cuda_vbo;
	GLuint m_boundary_cuda_vbo;
	
	struct cudaGraphicsResource* m_cuda_vbo_resource;
	struct cudaGraphicsResource* m_dem_cuda_vbo_resource;
	struct cudaGraphicsResource* m_boundary_cuda_vbo_resource;
	
	double m_update_elased_time;
};

#endif
