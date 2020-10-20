#ifndef _PARTICLE_H_
#define _PARTICLE_H_

#include <memory>
#include <vector>
#include <glm/common.hpp>
#include <cuda_runtime.h>
#include "Collider.h"

class Particle;
using Particle_Ptr = std::shared_ptr<Particle>;

struct ParticleDeviceData
{
	float3* m_d_prev_positions;
	float3* m_d_positions;
	float3* m_d_predict_positions;
	float3* m_d_new_positions;
	float3* m_d_prev_velocity;
	float3* m_d_velocity;
	float3* m_d_new_velocity;
	float3* m_d_force;
	float3* m_d_correction;

	float* m_d_mass;
	float* m_d_massInv;
	float* m_d_density;
	float* m_d_C;
	float* m_d_lambda;
	float* m_d_volume;
};

class ParticleSet
{
public:
	ParticleSet();
	ParticleSet(size_t n, float particle_mass);
	~ParticleSet();

	void Update(float dt);

	bool TestCollision(size_t i,Collider* other);
	void OnCollision(size_t i, Collider* other, float dt);

	void ResetPositions(std::vector<glm::vec3> positions, float particle_mass);
	void EraseTail(size_t start);

	void ReleaseDeviceData();

	size_t m_size;

	//std::vector<glm::vec3>	m_prev_positions;
	std::vector<glm::vec3>	m_positions;
	std::vector<glm::vec3>	m_predict_positions;
	std::vector<glm::vec3>	m_new_positions;

	std::vector<glm::vec3>	m_velocity;
	std::vector<glm::vec3>	m_force;
	
	std::vector<float>		m_mass;
	std::vector<float>		m_massInv;
	std::vector<float>		m_density;
	std::vector<float>		m_C;
	std::vector<float>		m_lambda;
	std::vector<float>		m_volume;

	ParticleDeviceData m_device_data;
};

/* deprecated("Using ParticleSet class instead\n") */
class Particle
{
public:
	Particle(glm::vec3 pos, float mass);
	~Particle();

	void Update(float dt);
	void UpdateCollider();

	bool TestCollision(Collider* other);
	void OnCollision(Collider* other, const float& dt);
	
	// Attributes
	
	glm::vec3	m_position;
	glm::vec3	m_velocity;
	glm::vec3	m_force;

	glm::vec3	m_prev_position;

	glm::vec3   m_new_position;
	glm::vec3   m_new_velocity;
	glm::vec3	m_new_force;

	float		m_mass;
	float		m_massInv;

	float		m_density;

	float		m_C;
	float		m_lambda;

private:
	PointCollider* m_collider;
};




#endif

