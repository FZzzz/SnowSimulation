#include "Collider.h"
#include "CollisionDetection.h"


Collider::Collider(ColliderTypes type)
	: m_type(type)
{
}

Collider::~Collider()
{
}

PointCollider::PointCollider(glm::vec3 pos)
	: Collider(ColliderTypes::POINT), m_position(pos)
{
}

PointCollider::~PointCollider()
{
}

bool PointCollider::TestCollision(Collider* other)
{
	bool result = false;

	switch (other->getColliderTypes())
	{
	case ColliderTypes::SPHERE:
		result = CollisionDetection::PointSphereIntersection(this, dynamic_cast<SphereCollider*>(other));
		break;
	case ColliderTypes::AABB:
		result = CollisionDetection::PointAABBIntersection(this, dynamic_cast<AABB*>(other));
		break;
	case ColliderTypes::OBB:
		result = CollisionDetection::PointOBBIntersection(this, dynamic_cast<OBB*>(other));
		break;
	case ColliderTypes::PLANE:
		result = CollisionDetection::PointSphereIntersection(this, dynamic_cast<SphereCollider*>(other));
		break;
	default:
		break;
	}
	
	return result;
}

SphereCollider::SphereCollider(glm::vec3 center, float radius)
	: Collider(ColliderTypes::SPHERE), m_center(center), m_radius(radius)
{
}

SphereCollider::~SphereCollider()
{
}

bool SphereCollider::TestCollision(Collider* other)
{
	bool result = false;

	switch (other->getColliderTypes())
	{
	case ColliderTypes::POINT:
		result = CollisionDetection::PointSphereIntersection(dynamic_cast<PointCollider*>(other), this);
		break;
	case ColliderTypes::SPHERE:
		result = CollisionDetection::SphereSphereIntersection(dynamic_cast<SphereCollider*>(other), this);
		break;
	case ColliderTypes::AABB:
		result = CollisionDetection::SphereAABBIntersection(this, dynamic_cast<AABB*>(other));
		break;
	case ColliderTypes::OBB:
		result = CollisionDetection::SphereOBBIntersection(this, dynamic_cast<OBB*>(other));
		break;
	case ColliderTypes::PLANE:
		result = CollisionDetection::SpherePlaneIntersection(this, dynamic_cast<PlaneCollider*>(other));
		break;
	default:
		break;
	}

	return result;
}

AABB::AABB(glm::vec3 min, glm::vec3 max)
	: Collider(ColliderTypes::AABB), 
	m_min(min), m_max(max)
{
}

AABB::~AABB()
{
}

bool AABB::TestCollision(Collider* other)
{
	bool result = false;
	
	switch (other->getColliderTypes())
	{
	case ColliderTypes::POINT:
		result = CollisionDetection::PointAABBIntersection(dynamic_cast<PointCollider*>(other), this);
		break;
	case ColliderTypes::SPHERE:
		result = CollisionDetection::SphereAABBIntersection(dynamic_cast<SphereCollider*>(other), this);
		break;
	case ColliderTypes::AABB:
		result = CollisionDetection::AABBAABBIntersection(dynamic_cast<AABB*>(other), this);
		break;
	case ColliderTypes::OBB:
		result = CollisionDetection::AABBOBBIntersection(this ,dynamic_cast<OBB*>(other));
		break;
	case ColliderTypes::PLANE:
		result = CollisionDetection::AABBPlaneIntersection(this, dynamic_cast<PlaneCollider*>(other));
		break;
	default:
		break;
	}

	return result;
}

OBB::OBB(glm::vec3 center, glm::vec3 local_x_axis, glm::vec3 local_y_axis, glm::vec3 local_z_axis, glm::vec3 extend)
	: Collider(ColliderTypes::OBB),
	m_local_axis{local_x_axis, local_y_axis, local_z_axis},
	m_extend(extend)
{
}

OBB::~OBB()
{
}

bool OBB::TestCollision(Collider* other)
{
	bool result = false;

	switch (other->getColliderTypes())
	{
	case ColliderTypes::POINT:
		result = CollisionDetection::PointOBBIntersection(dynamic_cast<PointCollider*>(other), this);
		break;
	case ColliderTypes::SPHERE:
		result = CollisionDetection::SphereOBBIntersection(dynamic_cast<SphereCollider*>(other), this);
		break;
	case ColliderTypes::AABB:
		result = CollisionDetection::AABBOBBIntersection(dynamic_cast<AABB*>(other), this);
		break;
	case ColliderTypes::OBB:
		result = CollisionDetection::OBBOBBIntersection(this, dynamic_cast<OBB*>(other));
		break;
	case ColliderTypes::PLANE:
		result = CollisionDetection::OBBPlaneIntersection(this, dynamic_cast<PlaneCollider*>(other));
		break;
	default:
		break;
	}

	return result;
}

PlaneCollider::PlaneCollider(glm::vec3 plane_normal, float d)
	: Collider(ColliderTypes::PLANE), m_normal(plane_normal), m_d(d)
{
}

PlaneCollider::~PlaneCollider()
{
}

bool PlaneCollider::TestCollision(Collider* other)
{
	bool result = false;

	switch (other->getColliderTypes())
	{
	case ColliderTypes::POINT:
		result = CollisionDetection::PointPlaneIntersection(dynamic_cast<PointCollider*>(other), this);
		break;
	case ColliderTypes::SPHERE:
		result = CollisionDetection::SpherePlaneIntersection(dynamic_cast<SphereCollider*>(other), this);
		break;
	case ColliderTypes::AABB:
		result = CollisionDetection::AABBPlaneIntersection(dynamic_cast<AABB*>(other), this);
		break;
	case ColliderTypes::OBB:
		result = CollisionDetection::OBBPlaneIntersection(dynamic_cast<OBB*>(other), this);
		break;	
	default:
		break;
	}

	return result;
}
