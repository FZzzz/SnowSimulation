#ifndef _GAMEOBJECT_H_
#define _GAMEOBJECT_H_

#include <vector>
#include <memory>
#include <string>
#include "Mesh.h"
#include "Transform.h"

/*
*  A GameObject is an object in the scene with the following properties
*  - Named object
*  - Has Transform (scale ,position, rotation)
*  - Either updatable or static
*  - May have mesh
*  - May have hierarchical structure
*/

// Unchangable after setting
enum OBJECT_FLAG_ENUM
{
	OBJECT_STATIC,
	OBJECT_INSTANCED,
	OBJECT_NORMAL,
	OBJECT_INVISIBLE
};

class GameObject;

class GameObject
{
public:
	GameObject();
	~GameObject();

	virtual void Initialize();
	virtual void Initialize(const glm::vec3& init_pos);
	virtual void Initialize(Transform& trans);
	virtual void Update();
	virtual void Release();

	void AddChild(std::shared_ptr<GameObject> child);
	void setParent(std::shared_ptr<GameObject> parent);
	void setMesh(std::shared_ptr<Mesh> obj);
	void setObjectType(OBJECT_FLAG_ENUM type);
	void setName(std::string name);

	inline bool hasMesh() { return (m_mesh != nullptr) ? true : false; };
	inline const std::shared_ptr<Mesh>& getMesh() { return m_mesh; };
	inline const std::vector<std::shared_ptr<GameObject>>& getChilds() { return m_childs; };
	inline const std::shared_ptr<GameObject> getParent() { return m_parent; }
	inline const OBJECT_FLAG_ENUM& getObjectType() { return m_object_type; };
	inline std::string getName() { return m_name; };
	
	Transform m_transform;
	
protected:	
	std::string m_name;
	std::vector<std::shared_ptr<GameObject>> m_childs;
	std::shared_ptr<GameObject> m_parent;
	std::shared_ptr<Mesh> m_mesh;
	
private:
	OBJECT_FLAG_ENUM m_object_type;
	
	
};

#endif
