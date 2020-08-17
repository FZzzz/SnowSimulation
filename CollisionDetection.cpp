#include "CollisionDetection.h"

bool CollisionDetection::PointSphereIntersection(PointCollider* point, SphereCollider* sphere)
{
	const glm::vec3& point_pos = point->m_position;
	const glm::vec3  v = point_pos - sphere->m_center;
	const float		 radius2 = sphere->m_radius * sphere->m_radius;

	return (glm::dot(v,v) <= radius2);
}

bool CollisionDetection::PointSphereIntersection(const glm::vec3& point, SphereCollider* sphere)
{
	const glm::vec3  v = point - sphere->m_center;
	const float		 radius2 = sphere->m_radius * sphere->m_radius;

	return (glm::dot(v, v) <= radius2);
}

bool CollisionDetection::PointAABBIntersection(PointCollider* point, AABB* aabb)
{
	const glm::vec3& point_pos = point->m_position;

	return (point_pos.x < aabb->m_max.x && point_pos.x > aabb->m_min.x &&
			point_pos.y < aabb->m_max.y && point_pos.y > aabb->m_min.y &&
			point_pos.z < aabb->m_max.z && point_pos.z > aabb->m_min.z);
}

bool CollisionDetection::PointAABBIntersection(const glm::vec3& point_pos, AABB* aabb)
{
	return (point_pos.x < aabb->m_max.x && point_pos.x > aabb->m_min.x &&
		point_pos.y < aabb->m_max.y && point_pos.y > aabb->m_min.y &&
		point_pos.z < aabb->m_max.z && point_pos.z > aabb->m_min.z);
}

bool CollisionDetection::PointOBBIntersection(PointCollider* point, OBB* obb)
{
	const glm::vec3& point_pos = point->m_position;
	const glm::vec3 v = point_pos - obb->m_center;
	   
	return (glm::abs(glm::dot(v, obb->m_local_axis[0])) <= obb->m_extend.x &&
			glm::abs(glm::dot(v, obb->m_local_axis[1])) <= obb->m_extend.y &&
			glm::abs(glm::dot(v, obb->m_local_axis[2])) <= obb->m_extend.z);
}

bool CollisionDetection::PointOBBIntersection(const glm::vec3& point, OBB* obb)
{
	const glm::vec3 v = point - obb->m_center;

	return (glm::abs(glm::dot(v, obb->m_local_axis[0])) <= obb->m_extend.x &&
		glm::abs(glm::dot(v, obb->m_local_axis[1])) <= obb->m_extend.y &&
		glm::abs(glm::dot(v, obb->m_local_axis[2])) <= obb->m_extend.z);
}

bool CollisionDetection::PointPlaneIntersection(PointCollider* point, PlaneCollider* plane)
{
	const glm::vec3& point_pos = point->m_position;

	return (glm::dot(plane->m_normal, point_pos)) <= plane->m_d;
}

bool CollisionDetection::PointPlaneIntersection(const glm::vec3& p, PlaneCollider* plane)
{
	float dot_val = glm::dot(plane->m_normal, p);

	return dot_val <= plane->m_d;
}

bool CollisionDetection::SphereSphereIntersection(SphereCollider* sphere0, SphereCollider* sphere1)
{
	const glm::vec3& center0 = sphere0->m_center;
	const glm::vec3& center1 = sphere1->m_center;
	const float&	 radius0 = sphere0->m_radius;
	const float&	 radius1 = sphere1->m_radius;
	const float		 dist2 = glm::dot(center0 - center1, center0 - center1);
	const float		 r_sum = radius0 + radius1;

	return dist2 <= (r_sum * r_sum);
}

bool CollisionDetection::SphereAABBIntersection(SphereCollider* sphere, AABB* aabb)
{
	const glm::vec3& center = sphere->m_center;
	const float& r = sphere->m_radius;

	/*Get closest AABB point to sphere center*/
	float x = glm::max(aabb->m_min.x, glm::min(center.x, aabb->m_max.x));
	float y = glm::max(aabb->m_min.y, glm::min(center.y, aabb->m_max.y));
	float z = glm::max(aabb->m_min.z, glm::min(center.z, aabb->m_max.z));

	glm::vec3 v = center - glm::vec3(x, y, z);

	return (glm::dot(v, v) <= r);
}

bool CollisionDetection::SphereOBBIntersection(SphereCollider* sphere, OBB* obb)
{
	
	return false;
}

bool CollisionDetection::SpherePlaneIntersection(SphereCollider* sphere, PlaneCollider* plane)
{
	const glm::vec3& center = sphere->m_center;
	float d_sum = sphere->m_radius + plane->m_d;

	return glm::dot(center, plane->m_normal) <= (d_sum*d_sum);
}

bool CollisionDetection::AABBAABBIntersection(AABB* box0, AABB* box1)
{
	if (box0->m_max.x < box1->m_min.x || box0->m_min.x > box1->m_max.x)  return false;
	if (box0->m_max.y < box1->m_min.y || box0->m_min.y > box1->m_max.y)  return false;
	if (box0->m_max.z < box1->m_min.z || box0->m_min.z > box1->m_max.z)  return false;
	
	return false;
}

bool CollisionDetection::AABBOBBIntersection(AABB* aabb, OBB* obb)
{
	return false;
}

bool CollisionDetection::AABBPlaneIntersection(AABB* aabb, PlaneCollider* plane)
{
	const glm::vec3& center = 0.5f * (aabb->m_min + aabb->m_max);
	const glm::vec3& extend = aabb->m_max - center;
		
	float r = extend.x * plane->m_normal.x + extend.y * plane->m_normal.y + extend.z * plane->m_normal.z;
	float s = glm::dot(plane->m_normal, center) - plane->m_d;

	return glm::abs(s) <= r;
}

bool CollisionDetection::OBBOBBIntersection(OBB* box0, OBB* box1)
{
	return false;
}

bool CollisionDetection::OBBPlaneIntersection(OBB* obb, PlaneCollider* plane)
{
	const glm::vec3& center = obb->m_center;
	const glm::vec3& extend = obb->m_extend;
	
	float r = extend.x * glm::abs(glm::dot(plane->m_normal, obb->m_local_axis[0])) +
			  extend.y * glm::abs(glm::dot(plane->m_normal, obb->m_local_axis[1])) +
			  extend.z * glm::abs(glm::dot(plane->m_normal, obb->m_local_axis[2]));

	float s = glm::dot(plane->m_normal, center) - plane->m_d;
	   
	return glm::abs(s) <= r;
}
