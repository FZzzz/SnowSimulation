#include <glm/glm.hpp>
#include <assert.h>
#include "Constraints.h"

#define EPSILON 0.000000000000001f 

Constraint::Constraint(size_t numOfRigidbodies)
	: m_lambda(0.f), m_stiffness(1.f), m_compliance(0.f), m_compliance_tmp(0.f)
{
	m_particles.resize(numOfRigidbodies, nullptr);
}

Constraint::~Constraint()
{
	for (size_t i = 0; i < m_particles.size(); ++i)
		m_particles[i] = nullptr;
	m_particles.clear();
}

void Constraint::ComputeCompliance(const float &dt)
{
	m_compliance_tmp = m_compliance / (dt * dt);
}

void Constraint::setStiffness(float stiffness)
{
	if (stiffness > 1.f)
		m_stiffness = 1.f;
	else if (stiffness < 0.f)
		m_stiffness = 0.f;
	else
		m_stiffness = stiffness;
}

void Constraint::setCompliance(float compliance)
{
	m_compliance = compliance;
}

BendConstraint::BendConstraint(Particle* p1, Particle* p2, float d)
	: Constraint(2)
{

}

BendConstraint::~BendConstraint()
{
}

DistanceConstraint::DistanceConstraint(Particle_Ptr p0, Particle_Ptr p1, float rest_length)
	: Constraint(2), m_rest_length(rest_length)
{
	m_particles[0] = p0;
	m_particles[1] = p1;
}

DistanceConstraint::~DistanceConstraint()
{
}

bool DistanceConstraint::SolvePBDConstraint()
{
	glm::vec3 correction;

	Particle_Ptr p0 = m_particles[0];
	Particle_Ptr p1 = m_particles[1];

	const float& w0 = p0->m_massInv;
	const float& w1 = p1->m_massInv;

	float w_sum = w0 + w1;
	float distance = glm::distance(p0->m_new_position, p1->m_new_position);
	float C = distance - m_rest_length;
	glm::vec3 v = p0->m_new_position - p1->m_new_position;

	//assert(distance < EPSILON);
	if (distance < EPSILON)
		distance = EPSILON;
	
	glm::vec3 n = v / distance;
	
	correction = (1.f / w_sum) * C * n;

	// Correction 
	p0->m_new_position += m_stiffness * -w0 * correction;
	p1->m_new_position += m_stiffness *  w1 * correction;

	return true;
}

bool DistanceConstraint::SolveXPBDConstraint()
{
	glm::vec3 correction;

	Particle_Ptr p0_data = m_particles[0];
	Particle_Ptr p1_data = m_particles[1];

	const float& w0 = p0_data->m_massInv;
	const float& w1 = p1_data->m_massInv;

	float w_sum = w0 + w1;
	float distance = glm::distance(p0_data->m_new_position, p1_data->m_new_position);
	float C = distance - m_rest_length;
	glm::vec3 v = p0_data->m_new_position - p1_data->m_new_position;

	//assert(distance < EPSILON);
	if (distance < EPSILON)
		distance = EPSILON;

	float delta_lambda = (-C - m_compliance_tmp * m_lambda) / (w_sum + m_compliance_tmp);
	correction = (delta_lambda * v) / distance;	
	m_lambda += delta_lambda;
	

	// Correction 
	p0_data->m_new_position +=  w0 * correction;
	p1_data->m_new_position += -w1 * correction;

	return true;
}

float DistanceConstraint::ConstraintFunction()
{
	float constraint_value = 0.f;
	const glm::vec3& p0 = m_particles[0]->m_position;
	const glm::vec3& p1 = m_particles[1]->m_position;

	constraint_value = glm::distance(p0, p1) - m_rest_length;	

	return constraint_value;
}

std::vector<std::vector<float>> DistanceConstraint::GradientFunction()
{
	std::vector<std::vector<float>> jacobian;

	jacobian.resize(1, std::vector<float>(3, 0));
	
	const glm::vec3& p0 = m_particles[0]->m_position;
	const glm::vec3& p1 = m_particles[1]->m_position;

	glm::vec3 n = p0 - p1;
	jacobian[0][0] = n.x;
	jacobian[0][1] = n.y;
	jacobian[0][2] = n.z;


	return jacobian;
}

bool BendConstraint::SolvePBDConstraint()
{
	return true;
}

bool BendConstraint::SolveXPBDConstraint()
{
	return true;
}

float BendConstraint::ConstraintFunction()
{
	return 0.0f;
}

std::vector<std::vector<float>> BendConstraint::GradientFunction()
{
	return std::vector<std::vector<float>>();
}

