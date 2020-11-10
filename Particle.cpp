#include "Particle.h"
#include "imgui/imgui.h"
#include "CollisionDetection.h"

Particle::Particle(glm::vec3 pos, float m): 
	m_position(pos),
	m_velocity(glm::vec3(0)),
	m_force(glm::vec3(0)),
	//m_prev_position(glm::vec3(0)),
	m_new_position(glm::vec3(0)),
	m_mass(m),
	m_massInv(1.f / m_mass), 
	m_density(0.f),
	m_C(0.f),
	m_lambda(0.f),
	m_collider(new PointCollider(pos))
{
}

Particle::~Particle()
{
	if(m_collider) delete m_collider;
	m_collider = nullptr;
}

void Particle::Update(float dt)
{
	//m_prev_position = m_position;
	
	m_force = m_new_force;
	
	/* This is why it is called "Position Based Dynamics" */
	m_velocity = (m_new_position - m_position) / dt;
	m_position = m_new_position;

	UpdateCollider();
}

void Particle::UpdateCollider()
{
	m_collider->m_position = m_position;
}

bool Particle::TestCollision(Collider* other)
{
	bool result = false;
	switch (other->getColliderTypes())
	{
	case Collider::ColliderTypes::SPHERE:
		result = CollisionDetection::PointSphereIntersection(
			m_new_position, dynamic_cast<SphereCollider*>(other));
		break;

	case Collider::ColliderTypes::AABB:
		result = CollisionDetection::PointAABBIntersection(
			m_new_position, dynamic_cast<AABB*>(other)
		);
		break;

	case Collider::ColliderTypes::OBB:
		result = CollisionDetection::PointOBBIntersection(
			m_new_position, dynamic_cast<OBB*>(other)
		);
		break;

	case Collider::ColliderTypes::PLANE:
		result = CollisionDetection::PointPlaneIntersection(
			m_new_position, dynamic_cast<PlaneCollider*>(other)
		);
		break;
	}
	return result;
}

void Particle::OnCollision(Collider* other, const float& dt)
{
	/* Collision response variables */
	glm::vec3 normal;
	glm::vec3 v_r;
	
	/**/
	switch (other->getColliderTypes())
	{
	case Collider::ColliderTypes::SPHERE: 
	{
		SphereCollider* sphere = dynamic_cast<SphereCollider*>(other);
		normal = m_new_position - sphere->m_center;

		if (glm::dot(normal, normal) == 0)
			v_r = -m_velocity;
		else
		{
			normal = glm::normalize(normal);
			v_r = m_velocity + 2.f * glm::dot(v_r, normal) * normal;
		}
		break;
	}

	case Collider::ColliderTypes::AABB:
	{
		AABB* aabb = dynamic_cast<AABB*>(other);

		glm::vec3 diff2max = aabb->m_max - m_new_position;
		glm::vec3 diff2min = m_new_position - aabb->m_min;

		glm::vec3 normal_preset[6] =
		{
			glm::vec3(1,0,0),
			glm::vec3(0,1,0),
			glm::vec3(0,0,1),
			glm::vec3(-1,0,0),
			glm::vec3(0,-1,0),
			glm::vec3(0,0,-1),
		};

		float min_dist = diff2max.x;
		normal = normal_preset[0];

		// test 6 distance
		for (int i = 1; i < 3; ++i)
		{
			if(min_dist < diff2max[i])
				min_dist = diff2max[i], normal = normal_preset[i];
		}

		for (int i = 0; i < 3; ++i)
		{
			if (min_dist < diff2min[i])
				min_dist = diff2min[i], normal = normal_preset[3 + i];
		}
		glm::vec3 tmp = glm::dot(m_velocity, normal) * normal;
		
		v_r = m_velocity - 2.f * glm::dot(m_velocity, normal) * normal;
		m_velocity = v_r;

		break;
	}
	
	case Collider::ColliderTypes::OBB:
	{
		/*
			TODO: Implement this
		*/
		break;
	}

	case Collider::ColliderTypes::PLANE:
	{
		PlaneCollider* plane = dynamic_cast<PlaneCollider*>(other);
		normal = plane->m_normal;
		glm::vec3 tmp = glm::dot(m_velocity, normal) * normal;
		v_r = m_velocity - 2.f * glm::dot(m_velocity, normal) * normal;
		m_velocity = v_r;

		break;
	}
	}


	/* Restitution */
	m_velocity = 0.87f * m_velocity;

	/*Re-prediction*/
	m_new_position = m_position + dt * m_velocity;
}

ParticleSet::ParticleSet()
	: m_size(0)
{
}

ParticleSet::ParticleSet(size_t n, float particle_mass)
	: m_size(n), m_full_size(n)
{
	//m_prev_positions.resize(n, glm::vec3(0,0,0));
	m_positions.resize(n, glm::vec3(0, 0, 0));
	m_predict_positions.resize(n, glm::vec3(0, 0, 0));
	m_new_positions.resize(n, glm::vec3(0, 0, 0));
	
	m_velocity.resize(n, glm::vec3(0.f, 0.f, 0.f));
	m_force.resize(n, glm::vec3(0, 0, 0));

	m_mass.resize(n, particle_mass);
	m_massInv.resize(n, 1.f/particle_mass);
	m_density.resize(n, 0.f);
	m_C.resize(n, 0.f);
	m_lambda.resize(n, 0.f);

	m_temperature.resize(n, 0.f);
}

ParticleSet::~ParticleSet()
{
	//m_prev_positions.clear();
	m_positions.clear();
	m_new_positions.clear();
	m_new_positions.clear();

	m_velocity.clear();
	m_force.clear();

	m_mass.clear();
	m_massInv.clear();
	m_density.clear();
	m_C.clear();

	m_lambda.clear();
	m_volume.clear();
	m_temperature.clear();
}

void ParticleSet::Update(float dt)
{
	for (size_t i = 0; i < m_size; ++i)
	{
		//m_prev_positions[i] = m_positions[i];
		m_velocity[i] = (m_new_positions[i] - m_positions[i]) / dt;
		m_positions[i] = m_new_positions[i];
	}
}

bool ParticleSet::TestCollision(size_t i, Collider* other)
{
	bool result = false;
	switch (other->getColliderTypes())
	{
	case Collider::ColliderTypes::SPHERE:
		result = CollisionDetection::PointSphereIntersection(
			m_predict_positions[i], dynamic_cast<SphereCollider*>(other));
		break;

	case Collider::ColliderTypes::AABB:
		result = CollisionDetection::PointAABBIntersection(
			m_predict_positions[i], dynamic_cast<AABB*>(other)
		);
		break;

	case Collider::ColliderTypes::OBB:
		result = CollisionDetection::PointOBBIntersection(
			m_predict_positions[i], dynamic_cast<OBB*>(other)
		);
		break;

	case Collider::ColliderTypes::PLANE:
		result = CollisionDetection::PointPlaneIntersection(
			m_predict_positions[i], dynamic_cast<PlaneCollider*>(other)
		);
		break;
	}
	return result;
}

void ParticleSet::OnCollision(size_t i, Collider* other, float dt)
{
	/* Collision response variables */
	glm::vec3 normal;
	glm::vec3 v_r;

	/**/
	switch (other->getColliderTypes())
	{
	case Collider::ColliderTypes::SPHERE:
	{
		SphereCollider* sphere = dynamic_cast<SphereCollider*>(other);
		normal = m_predict_positions[i] - sphere->m_center;

		if (glm::dot(normal, normal) == 0)
			v_r = -m_velocity[i];
		else
		{
			normal = glm::normalize(normal);
			v_r = m_velocity[i] + 2.f * glm::dot(v_r, normal) * normal;
		}
		break;
	}

	case Collider::ColliderTypes::AABB:
	{
		AABB* aabb = dynamic_cast<AABB*>(other);

		glm::vec3 diff2max = aabb->m_max - m_predict_positions[i];
		glm::vec3 diff2min = m_predict_positions[i] - aabb->m_min;

		glm::vec3 normal_preset[6] =
		{
			glm::vec3(1,0,0),
			glm::vec3(0,1,0),
			glm::vec3(0,0,1),
			glm::vec3(-1,0,0),
			glm::vec3(0,-1,0),
			glm::vec3(0,0,-1),
		};

		float min_dist = diff2max.x;
		normal = normal_preset[0];

		// test 6 distance
		for (int i = 1; i < 3; ++i)
		{
			if (min_dist < diff2max[i])
				min_dist = diff2max[i], normal = normal_preset[i];
		}

		for (int i = 0; i < 3; ++i)
		{
			if (min_dist < diff2min[i])
				min_dist = diff2min[i], normal = normal_preset[3 + i];
		}
		//glm::vec3 tmp = glm::dot(m_velocity[i], normal) * normal;

		v_r = m_velocity[i] - 2.f * glm::dot(m_velocity[i], normal) * normal;
		m_velocity[i] = v_r;

		break;
	}

	case Collider::ColliderTypes::OBB:
	{
		/*
			TODO: Implement this
		*/
		break;
	}

	case Collider::ColliderTypes::PLANE:
	{
		PlaneCollider* plane = dynamic_cast<PlaneCollider*>(other);
		normal = plane->m_normal;
		glm::vec3 tmp = glm::dot(m_velocity[i], normal) * normal;
		v_r = m_velocity[i] - 2.f * glm::dot(m_velocity[i], normal) * normal;
		m_velocity[i] = v_r;

		break;
	}
	}

	/* Restitution */
	m_velocity[i] = 0.99f * m_velocity[i];

	/*Re-prediction*/
	m_predict_positions[i] = m_positions[i] + dt * m_velocity[i];
	m_new_positions[i] = m_predict_positions[i];
}

/* 
    This is the dirty function for setting boundary particles in convenience
	Fix this if possible
*/
void ParticleSet::ResetPositions(std::vector<glm::vec3> positions, float particle_mass)
{
	m_positions.clear();

	m_positions = positions;
	m_size = m_positions.size();

	m_predict_positions.resize(m_size, glm::vec3(0, 0, 0));
	m_new_positions.resize(m_size, glm::vec3(0, 0, 0));

	m_velocity.resize(m_size, glm::vec3(0.f, 0, 0.f));
	m_force.resize(m_size, glm::vec3(0, 0, 0));

	m_mass.resize(m_size, particle_mass);
	m_massInv.resize(m_size, 1.f / particle_mass);
	m_density.resize(m_size, 0.f);
	m_C.resize(m_size, 0.f);
	m_lambda.resize(m_size, 0.f);
	m_volume.resize(m_size, 0.f);
	m_temperature.resize(m_size, 0.f);
}

void ParticleSet::ReleaseDeviceData()
{
	// Release cuda memory
	if (m_device_data.m_d_predict_positions != nullptr) cudaFree(m_device_data.m_d_predict_positions);
	if (m_device_data.m_d_new_positions != nullptr) cudaFree(m_device_data.m_d_new_positions);
	if (m_device_data.m_d_velocity != nullptr) cudaFree(m_device_data.m_d_velocity);
	if (m_device_data.m_d_force!= nullptr) cudaFree(m_device_data.m_d_force);
	if (m_device_data.m_d_correction != nullptr) cudaFree(m_device_data.m_d_correction);
	if (m_device_data.m_d_mass != nullptr) cudaFree(m_device_data.m_d_mass);
	if (m_device_data.m_d_massInv != nullptr) cudaFree(m_device_data.m_d_massInv);
	if (m_device_data.m_d_density != nullptr) cudaFree(m_device_data.m_d_density);
	if (m_device_data.m_d_C != nullptr) cudaFree(m_device_data.m_d_C);
	if (m_device_data.m_d_lambda != nullptr) cudaFree(m_device_data.m_d_lambda);
	if (m_device_data.m_d_volume != nullptr) cudaFree(m_device_data.m_d_volume);

	if (m_device_data.m_d_T != nullptr) cudaFree(m_device_data.m_d_T);
	if (m_device_data.m_d_new_T != nullptr) cudaFree(m_device_data.m_d_new_T);

	if (m_device_data.m_d_predicate != nullptr) cudaFree(m_device_data.m_d_predicate);
	if (m_device_data.m_d_scan_index != nullptr) cudaFree(m_device_data.m_d_scan_index);
	if (m_device_data.m_d_new_end != nullptr) cudaFree(m_device_data.m_d_new_end);

	if (m_device_data.m_d_contrib != nullptr) cudaFree(m_device_data.m_d_contrib);
 
	if (m_device_data.m_d_connect_record != nullptr) cudaFree(m_device_data.m_d_connect_record);
	if (m_device_data.m_d_connect_length != nullptr) cudaFree(m_device_data.m_d_connect_length);
	if (m_device_data.m_d_iter_end != nullptr) cudaFree(m_device_data.m_d_iter_end);

	if (m_device_data.m_d_new_end != nullptr) cudaFree(m_device_data.m_d_new_end);
	

}

void ParticleSet::AppendExtraMemory(ParticleSet* other)
{
	if (m_size == 0)
		return;
	// a.insert(a.end(), b.begin(), b.end());
	std::vector<glm::vec3> float3_inf_vec(other->m_size, glm::vec3(INFINITY));
	std::vector<glm::vec3> float3_zero_vec(other->m_size, glm::vec3(0));
	std::vector<float> float_inf_vec(other->m_size, INFINITY);
	std::vector<float> float_zero_vec(other->m_size, 0);

	m_positions.insert(m_positions.end(), float3_inf_vec.begin(), float3_inf_vec.end());
	m_predict_positions.insert(m_predict_positions.end(), float3_inf_vec.begin(), float3_inf_vec.end());
	m_new_positions.insert(m_new_positions.end(), float3_inf_vec.begin(), float3_inf_vec.end());
	m_velocity.insert(m_velocity.end(), float3_zero_vec.begin(), float3_zero_vec.end());
	m_force.insert(m_force.end(), float3_zero_vec.begin(), float3_zero_vec.end());
	
	m_mass.insert(m_mass.end(), other->m_size, m_mass[0]);
	m_massInv.insert(m_massInv.end(), other->m_size, m_massInv[0]);
	m_density.insert(m_density.end(), float_zero_vec.begin(), float_zero_vec.end());
	m_C.insert(m_C.end(), float_zero_vec.begin(), float_zero_vec.end());
	m_lambda.insert(m_lambda.end(), float_zero_vec.begin(), float_zero_vec.end());
	//m_volume.insert(m_C.end(), float_zero_vec.begin(), float_zero_vec.end());
	m_temperature.insert(m_temperature.end(), float_zero_vec.begin(), float_zero_vec.end());

	m_full_size = m_size + other->m_size;
}
