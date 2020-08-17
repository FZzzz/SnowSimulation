#include "Rigidbody.h"

Rigidbody::Rigidbody(glm::vec3 pos, float mass, float inertia=1.f)
	: m_position(pos),
	m_velocity(glm::vec3()),
	m_mass(mass),
	m_force(mass),
	m_massInv( (mass==0)? 100000.f: 1.f / mass),
	m_inertia(inertia),
	m_isStatic(false),
	m_collider(nullptr)
{

}

Rigidbody::~Rigidbody()
{
}

void Rigidbody::Update()
{
	if (m_isStatic)
		return;
	m_prev_position = m_position;
	m_prev_velocity = m_velocity;

	m_position = m_new_position;
	m_velocity = m_new_velocity;
}

void Rigidbody::setCollider(Collider* collider)
{
	m_collider = collider;
}
