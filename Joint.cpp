#include "Joint.h"
#include "GLFWApp.h"

// Joint
Joint::Joint(std::string name) : m_offset_from_parent(glm::vec3(0,0,0)), m_num_channels(0), m_parent(nullptr)
{
	GameObject::setName(name);
	for (int i = 0; i < 6; ++i)
		m_joint_limit[i] = { 0,0 };
}

Joint::~Joint()
{
}

void Joint::Initialize()
{
	GameObject::Initialize();
}

void Joint::Initialize(Transform& trans)
{
	auto resource_manager = GLFWApp::getInstance()->getResourceManager();
	const auto mesh_list = resource_manager->getMeshes();
	// TODO: remember to replace this with findMesh(name) / setMesh()
	auto mesh = mesh_list[2];

	GameObject::Initialize(trans.m_translation);
}

void Joint::setParent(std::shared_ptr<Joint> parent)
{
	m_parent = parent;
	m_transform.setParent(&(parent->m_transform));
}

void Joint::AddChild(std::shared_ptr<Joint> child)
{
	m_childs.push_back(child);
}
