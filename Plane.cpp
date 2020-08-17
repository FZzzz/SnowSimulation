#include "Plane.h"
#include "GLFWApp.h"

Plane::Plane()
	:m_collider(nullptr)
{
}

Plane::~Plane()
{
	if (m_collider) delete m_collider;
	m_collider = nullptr;
}

void Plane::Initialize(const glm::vec3& init_pos, const std::shared_ptr<Shader>& shader)
{

	//std::shared_ptr<Shader> shader = std::make_shared<Shader>();
	//shader->SetupShader("resources/shader/shadow_mapping_vs.glsl",
	//					"resources/shader/shadow_mapping_fs.glsl");

	std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>();
	std::vector<glm::vec3> vertex_pos;
	std::vector<unsigned int> indices;

	mesh->Initialize(shader);

	vertex_pos.reserve(4);
	vertex_pos.push_back(glm::vec3(  100.0f, 0.0f, -100.0f));
	vertex_pos.push_back(glm::vec3(  100.0f, 0.0f,  100.0f));
	vertex_pos.push_back(glm::vec3( -100.0f, 0.0f,  100.0f));
	vertex_pos.push_back(glm::vec3( -100.0f, 0.0f, -100.0f));

	vertex_pos.reserve(6);
	indices.push_back(0);
	indices.push_back(1);
	indices.push_back(2);
	indices.push_back(0);
	indices.push_back(2);
	indices.push_back(3);
	
	mesh->setPositions(vertex_pos);
	mesh->setIndices(indices);
	mesh->SetupGLBuffers();

	GameObject::setMesh(mesh);
	GameObject::Initialize(init_pos);

	m_collider = new PlaneCollider(glm::vec3(0,1,0), 0);

}

