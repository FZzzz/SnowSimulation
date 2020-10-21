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

	ParticleSet* AllocateSPHParticles(size_t n, float particle_mass);
	ParticleSet* AllocateDEMParticles(size_t n, float particle_mass);
	ParticleSet* AllocateBoundaryParticles();

	//void setParticles(std::vector<std::shared_ptr<Particle>> particles);
	void setParticleRadius(float particle_radius);

	inline ParticleSet* getSPHParticles() { return m_sph_particles; };
	inline ParticleSet* getDEMParticles() { return m_dem_particles; };
	inline ParticleSet* getBoundaryParticles() { return m_boundary_particles; };
//	inline std::vector<std::shared_ptr<Particle>>& getParticles() { return m_particles; };
	inline GLuint getSPH_VAO() { return m_sph_vao; };
	inline GLuint getSPH_VBO_0() { return m_sph_vbo[0]; };
	inline GLuint getSPH_VBO_1() { return m_sph_vbo[1]; };
	inline GLuint getSPH_EBO() { return m_sph_ebo; };

	inline GLuint getDEMVAO() { return m_dem_vao; };
	inline GLuint getDEM_VBO_0() { return m_dem_vbo[0]; };
	inline GLuint getDEM_VBO_1() { return m_dem_vbo[1]; };
	inline GLuint getDEMEBO() { return m_dem_ebo; };

	inline GLuint getBoundaryVAO() { return m_boundary_vao; };
	inline GLuint getBoundaryVBO() { return m_boundary_vbo; };
	inline GLuint getBoundaryEBO() { return m_boundary_ebo; };

	inline cudaGraphicsResource** getSPHCUDAGraphicsResource() { return m_sph_cuda_vbo_resource; };
	inline cudaGraphicsResource** getDEMCUDAGraphicsResource() { return m_dem_cuda_vbo_resource; };
	inline cudaGraphicsResource** getBoundaryCUDAGraphicsResource() { return &m_boundary_cuda_vbo_resource; };
	

	inline double& getUpdateTime() { return m_update_elased_time; };

	inline float getParticleRadius() { return m_particle_radius; };

private:
	
	//inner function
	void SetupCUDAMemory();
	void RegisterCUDAVBO();
	void GenerateGLBuffers();

	void UpdateGLBUfferData();

	ParticleSet* m_sph_particles;
	ParticleSet* m_dem_particles;
	ParticleSet* m_boundary_particles;

	/* OpenGL members */
	//std::vector<std::shared_ptr<Particle>> m_particles;
	std::vector<unsigned int> m_particle_indices;
	std::vector<unsigned int> m_dem_particle_indices;
	std::vector<unsigned int> m_boundary_particle_indices;
	
	float m_particle_radius;

	GLuint m_sph_vao;
	GLuint m_sph_vbo[2];
	GLuint m_sph_ebo;

	GLuint m_dem_vao;
	GLuint m_dem_vbo[2];
	//GLuint m_dem_vbo;
	GLuint m_dem_ebo;

	GLuint m_boundary_vao;
	GLuint m_boundary_vbo;
	GLuint m_boundary_ebo;
		
	GLuint m_sph_cuda_vbo[2];
	GLuint m_dem_cuda_vbo[2];
	//GLuint m_dem_cuda_vbo;
	GLuint m_boundary_cuda_vbo;
	
	struct cudaGraphicsResource* m_sph_cuda_vbo_resource[2];
	struct cudaGraphicsResource* m_dem_cuda_vbo_resource[2];
	struct cudaGraphicsResource* m_boundary_cuda_vbo_resource;

	//struct cudaGraphicsResource* m_dem_color_vbo_res;
	
	double m_update_elased_time;
};

#endif
