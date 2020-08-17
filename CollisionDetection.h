#ifndef _COLLISION_DETECTION_H_
#define _COLLISION_DETECTION_H_

#include <memory>
#include "Collider.h"

// forward declaration
class Collider;
class PointCollider;
class SphereCollider;
class AABB;
class OBB;
class PlaneCollider;

namespace CollisionDetection
{
	bool PointSphereIntersection(PointCollider* point, SphereCollider* sphere);
	bool PointSphereIntersection(const glm::vec3& point, SphereCollider* sphere);
	bool PointAABBIntersection(PointCollider* point, AABB* aabb);
	bool PointAABBIntersection(const glm::vec3& point, AABB* aabb);
	bool PointOBBIntersection(PointCollider* point, OBB* obb);
	bool PointOBBIntersection(const glm::vec3& point, OBB* obb);
	bool PointPlaneIntersection(PointCollider* point, PlaneCollider* plane);
	bool PointPlaneIntersection(const glm::vec3& p, PlaneCollider* plane);
	
	bool SphereSphereIntersection(SphereCollider* sphere0, SphereCollider* sphere1);
	bool SphereAABBIntersection(SphereCollider* sphere, AABB* aabb);
	bool SphereOBBIntersection(SphereCollider* sphere, OBB* obb);
	bool SpherePlaneIntersection(SphereCollider* sphere, PlaneCollider* plane);
	
	bool AABBAABBIntersection(AABB* box0, AABB* box1);
	bool AABBOBBIntersection(AABB* aabb, OBB* obb);
	bool AABBPlaneIntersection(AABB* aabb, PlaneCollider* plane);
	
	bool OBBOBBIntersection(OBB* box0, OBB* box1);
	bool OBBPlaneIntersection(OBB* obb, PlaneCollider* plane);
};

#endif