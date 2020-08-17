#ifndef _COLLIDER_H_
#define _COLLIDER_H_

#include <glm/glm.hpp>
#include <memory>

class Collider
{
public:
	enum class ColliderTypes
	{
		POINT = 0,
		SPHERE,
		AABB,
		OBB,
		PLANE
	};

	Collider() = delete;
	Collider(ColliderTypes type);
	~Collider();
	
	virtual bool TestCollision(Collider* other) = 0;

	// getters
	__forceinline ColliderTypes getColliderTypes() { return m_type; };

private:
	ColliderTypes m_type;

};

/* 
 * Performance issue: Extra allocation needed
 */
class PointCollider final : public Collider
{
public:
	PointCollider(glm::vec3 pos);
	~PointCollider();

	virtual bool TestCollision(Collider* other);
	
	glm::vec3 m_position;
};

class SphereCollider final : public Collider
{
public:
	SphereCollider() = delete;
	SphereCollider(glm::vec3 center, float radius);
	~SphereCollider();

	virtual bool TestCollision(Collider* other);

	glm::vec3 m_center;
	float	  m_radius;

};

class AABB final : public Collider
{
public:
	AABB() = delete;
	AABB(glm::vec3 min, glm::vec3 max);
	~AABB();

	virtual bool TestCollision(Collider* other);

	//getters
	inline glm::vec3 getCenter() { return (m_max + m_min) * 0.5f; };
	inline glm::vec3 getExtends() { return (m_max - m_min) * 0.5f; };

	glm::vec3 m_min;
	glm::vec3 m_max;
};

class OBB final : public Collider
{
public:
	OBB() = delete;
	OBB(glm::vec3 center,
		glm::vec3 local_x_axis,
		glm::vec3 local_y_axis,
		glm::vec3 local_z_axis,
		glm::vec3 extend);
	~OBB();

	virtual bool TestCollision(Collider* other);

	glm::vec3 m_center;
	glm::vec3 m_local_axis[3];
	glm::vec3 m_extend;

};

class PlaneCollider final : public Collider
{
public:
	PlaneCollider() = delete;
	PlaneCollider(glm::vec3 plane_normal, float distance_from_orgin);
	~PlaneCollider();

	virtual bool TestCollision(Collider* other);

	glm::vec3 m_normal;
	float m_d;

};


#endif 
