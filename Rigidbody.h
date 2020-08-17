#ifndef _RIGIDBODY_H_
#define _RIGIDBODY_H_

#include "common.h"
#include "Collider.h"

/* Provide Rigidbody class in this project but not going to use this for jelly */

class Rigidbody
{
public:
	Rigidbody() = delete;
	Rigidbody(glm::vec3 pos, float mass, float inertia);
	~Rigidbody();
	
	void Update();

	glm::vec3	m_position;
	glm::vec3	m_velocity;
	glm::vec3	m_force;

	glm::vec3	m_new_position;
	glm::vec3	m_new_velocity;

	glm::vec3	m_prev_position;
	glm::vec3	m_prev_velocity;
	
	bool		m_isStatic;

	// setters
	void setCollider(Collider* collider);

	// getters for members unchange members 
	inline float getMass()			{ return m_mass; };
	inline float getMassInv()		{ return m_massInv; };
	inline float getInertia()		{ return m_inertia; };
	
	inline Collider* getCollider()	{ return m_collider; };


private:
	
	float		m_mass;
	float		m_massInv;
	float		m_inertia;

	Collider*	m_collider;

};

#endif
