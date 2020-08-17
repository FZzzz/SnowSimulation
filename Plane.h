#ifndef _PLANE_H_
#define _PLANE_H_

#include "GameObject.h"
#include "Shader.h"
#include "Collider.h"

class Plane :
	public GameObject
{
public:
	Plane();
	~Plane();

	void Initialize(const glm::vec3& init_pos, const std::shared_ptr<Shader>& shader);

	// getters
	inline PlaneCollider* getCollider() { return m_collider; };

private:
	PlaneCollider* m_collider;
	
};

#endif