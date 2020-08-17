#include "Renderer.h"


Renderer::Renderer() :
	m_mainCamera(nullptr),
	m_resource_manager(nullptr),
	m_viewport_width(1600),
	m_viewport_height(900),
	m_clear_color(0, 0, 0.35f, 0),
	m_ubo(-1)
{
}

Renderer::~Renderer()
{
}

void Renderer::Initialize(
	std::shared_ptr<ResourceManager> resource_manager,
	std::shared_ptr<ParticleSystem> particle_system,
	std::shared_ptr<Camera> camera, 
	int viewport_width, int viewport_height)
{
	m_resource_manager = resource_manager;
	m_particle_system = particle_system;
	m_mainCamera = camera;
	m_viewport_width = viewport_width;
	m_viewport_height = viewport_height;
}

void Renderer::ClearBuffer()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(m_clear_color.r, m_clear_color.g, m_clear_color.b, m_clear_color.a);
}

void Renderer::Render()
{
	ClearBuffer();
	glViewport(0, 0, m_viewport_width, m_viewport_height);
	//RenderObjects();
	RenderParticles();
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


void Renderer::setMainCamera(std::shared_ptr<Camera> camera)
{
	m_mainCamera = camera;
}

void Renderer::setClearColor(glm::vec4 clear_color)
{
	m_clear_color = clear_color;
}

void Renderer::SetupUniformBufferOject()
{
	if (m_ubo == -1)
		return;
	/*
	 * Setup UBO here
	 */
	glBindBuffer(GL_UNIFORM_BUFFER, m_ubo);
	glBufferSubData(
		GL_UNIFORM_BUFFER,
		0,
		sizeof(glm::mat4),
		glm::value_ptr(m_mainCamera->m_projection));
	glBufferSubData(
		GL_UNIFORM_BUFFER,
		sizeof(glm::mat4),
		sizeof(glm::mat4),
		glm::value_ptr(m_mainCamera->m_lookAt));
	/* light matrix from light system (light map?)
	glBufferSubData(
		GL_UNIFORM_BUFFER,
		2 * sizeof(glm::mat4),
		sizeof(glm::mat4),
		glm::value_ptr(m_renderer->getLightMat()));
	*/
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void Renderer::RenderObjects()
{
	const Shader_Object_Map& shader_object_map = m_resource_manager->getShaderObjectMap();
	// For each shader program we call object->render()
	// Our render will automatically arrange the shader->objects map
	for (auto it = shader_object_map.cbegin(); it != shader_object_map.cend(); ++it)
	{
		auto shader = it->first;
		const auto& obj_vec = it->second;

		shader->Use();
		for (auto obj_it = obj_vec.cbegin(); obj_it != obj_vec.cend(); ++obj_it)
		{
			RenderGameObject(shader, (*obj_it)->getMesh(), (*obj_it)->m_transform);
		}
	}
}

void Renderer::RenderParticles()
{
	// fluid particles
	const ParticleSet* const particles = m_particle_system->getParticles();
	const std::shared_ptr<Shader> shader = m_resource_manager->FindShaderByName("PointSprite");
	
#ifdef _DEBUG
	assert(shader);
#endif
	shader->Use();

	const glm::mat4 pvm = m_mainCamera->m_cameraMat * glm::mat4(1);
	shader->SetUniformMat4("pvm", pvm);
	shader->SetUniformFloat("point_size", 30.f);
	shader->SetUniformVec3("light_pos", m_mainCamera->m_position);
	shader->SetUniformVec3("camera_pos", m_mainCamera->m_position);
	shader->SetUniformMat4("view", m_mainCamera->m_lookAt);
	shader->SetUniformVec3("point_color", glm::vec3(0.7f, 0.7f, 1.f));
	glBindVertexArray(m_particle_system->getVAO());
	glBindBuffer(GL_ARRAY_BUFFER, m_particle_system->getVBO());
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_particle_system->getEBO());

	glDrawArrays(GL_POINTS, 0, m_particle_system->getParticles()->m_size);

	glBindVertexArray(0);

#ifdef _RENDER_BOUNDARY_
	// render boundary particles
	const ParticleSet* const boundary_particles = m_particle_system->getBoundaryParticles();

	shader->SetUniformVec3("point_color", glm::vec3(1.f, 1.f, 1.f));

	glBindVertexArray(m_particle_system->getBoundaryVAO());
	glBindBuffer(GL_ARRAY_BUFFER, m_particle_system->getBoundaryVBO());
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glDrawArrays(GL_POINTS, 0, m_particle_system->getBoundaryParticles()->m_size);

	glBindVertexArray(0);
#endif
}

void Renderer::RenderGameObject(const std::shared_ptr<Shader>& shader, const std::shared_ptr<Mesh>& mesh, const Transform& transform)
{
	if (!m_mainCamera)
		return;

	if (!mesh)
	{
		std::cout << "No Mesh to Render\n";
		return;
	}
	//mesh->Render();

	//shadow mapping shader
	const glm::mat4 pvm = m_mainCamera->m_cameraMat * transform.getModelMatWorld();
	//shader->SetUniformMat4("pvm", pvm);
	shader->SetUniformMat4("modelMat", transform.getModelMatWorld());
	//shader->SetUniformMat4("lightSpaceMatrix", light_mat);

	// shadow map configuration
	shader->SetUniformInt("shadowMap", 0);
	shader->SetUniformVec3("lightPos", m_mainCamera->m_position);
	shader->SetUniformVec3("viewPos", m_mainCamera->m_position);

	//glBindTexture(GL_TEXTURE_2D, m_depthMap);
	glActiveTexture(GL_TEXTURE0);
	//glBindTexture(GL_TEXTURE_2D, m_depthMap);

	glBindVertexArray(mesh->getVAO());
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->getEBO());
	glDrawElements(
		GL_TRIANGLES,
		static_cast<unsigned int>(mesh->getNumberOfIndices()),
		GL_UNSIGNED_INT,
		0);
	glBindVertexArray(0);
}
