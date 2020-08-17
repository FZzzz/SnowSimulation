#include "GameObject.h"
#include "GLFWApp.h"

GameObject::GameObject() : m_parent(nullptr), m_mesh(nullptr), m_object_type(OBJECT_NORMAL)//, m_animator(nullptr)//, m_collider(nullptr)
{
}

GameObject::~GameObject()
{
#ifdef _DEBUG
	std::cout << "Obj Dtor\n";
#endif
	Release();
}

void GameObject::Initialize()
{
	m_transform.Update();
}

void GameObject::Initialize(const glm::vec3& init_pos)
{
	m_transform.m_translation = init_pos;
	m_transform.Update();
}

void GameObject::Initialize(Transform & trans)
{
	m_transform = trans;
	m_transform.Update();             
}

void GameObject::Update()
{
	if (m_object_type == OBJECT_FLAG_ENUM::OBJECT_STATIC)
		return;

	m_transform.Update();
}

void GameObject::Release()
{	
	m_mesh.reset();
	//m_animator.reset();
	//delete m_transform;	
}

void GameObject::AddChild(std::shared_ptr<GameObject> child)
{
	m_childs.push_back(child);
}

void GameObject::setParent(std::shared_ptr<GameObject> parent)
{
	m_parent = parent;
	m_transform.setParent(&(parent->m_transform));
}

void GameObject::setMesh(std::shared_ptr<Mesh> obj)
{
	m_mesh = obj;
}
void GameObject::setObjectType(OBJECT_FLAG_ENUM type)
{
	m_object_type = type;
}
void GameObject::setName(std::string name)
{
	m_name = name;
}
/*
void GameObject::setAnimator(std::shared_ptr<Animator> animator)
{
	//m_animator = animator;
}
*/
