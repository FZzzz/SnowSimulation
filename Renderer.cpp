#include "Renderer.h"


Renderer::Renderer() :
	m_mainCamera(nullptr),
	m_resource_manager(nullptr),
	m_viewport_width(1600),
	m_viewport_height(900),
	m_clear_color(0.15f, 0.15f, 0.15f, 0),
	m_ubo(-1),
	m_b_sph_visibility(true),
	m_b_dem_visibility(true),
	m_b_boundary_visibility(false),
	m_b_smooth_depth(false),
	m_use_temperature_shader(false)
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

	m_screen_size = glm::vec2(viewport_width, viewport_height);

	m_blur_dirX = glm::vec2(1.f / m_screen_size.x, 1.0f);
	m_blur_dirY = glm::vec2(1.f, 1.f / m_screen_size.y);

	InitializeDepthBuffers();
}

void Renderer::InitializeDepthBuffers()
{
	glGenFramebuffers(1, &m_depth_map_fbo);
	glGenTextures(1, &m_depth_map);
	glBindTexture(GL_TEXTURE_2D, m_depth_map);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
		m_viewport_width, m_viewport_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	float borderColor[] = { 1.0, 1.0, 1.0, 1.0 };
	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
	// attach depth texture as FBO's depth buffer
	glBindFramebuffer(GL_FRAMEBUFFER, m_depth_map_fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_depth_map, 0);
	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);


	float quadVertices[] = {
		// positions        // texture Coords
		-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
		 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
		 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
	};
	// setup plane VAO
	glGenVertexArrays(1, &m_screen_vao);
	glGenBuffers(1, &m_screen_vbo);
	glBindVertexArray(m_screen_vao);
	glBindBuffer(GL_ARRAY_BUFFER, m_screen_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	glBindVertexArray(0); //unbind screen_vao
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
	RenderFluidDepth();
	SmoothDepth();
	//RenderParticles();
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Renderer::SwitchSphVisibility()
{
	m_b_sph_visibility = !m_b_sph_visibility;
}

void Renderer::SwitchDEMVisibility()
{
	m_b_dem_visibility = !m_b_dem_visibility;
}

void Renderer::SwitchBoundaryVisibility()
{
	m_b_boundary_visibility = !m_b_boundary_visibility;
}

void Renderer::SwitchTemperatureShader()
{
	m_use_temperature_shader = !m_use_temperature_shader;
}

void Renderer::SwitchDepthSmooth()
{
	m_b_smooth_depth = !m_b_smooth_depth;
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
	const std::shared_ptr<Shader> point_shader = m_resource_manager->FindShaderByName("PointSprite");
	const glm::mat4 pvm = m_mainCamera->m_cameraMat * glm::mat4(1);
	point_shader->SetUniformMat4("pvm", pvm);
	point_shader->SetUniformFloat("point_size", 30.f);
	point_shader->SetUniformVec3("light_pos", m_mainCamera->m_position);
	point_shader->SetUniformVec3("camera_pos", m_mainCamera->m_position);
	point_shader->SetUniformMat4("view", m_mainCamera->m_lookAt);
	
	const std::shared_ptr<Shader> temp_shader = m_resource_manager->FindShaderByName("PointSpriteTemperature");
	temp_shader->SetUniformMat4("pvm", pvm);
	temp_shader->SetUniformFloat("point_size", 30.f);
	temp_shader->SetUniformVec3("light_pos", m_mainCamera->m_position);
	temp_shader->SetUniformVec3("camera_pos", m_mainCamera->m_position);
	temp_shader->SetUniformMat4("view", m_mainCamera->m_lookAt);
	temp_shader->SetUniformVec3("hottest_color", glm::vec3(1, 0, 0));
	temp_shader->SetUniformVec3("coolest_color", glm::vec3(0, 0, 1));
	temp_shader->SetUniformFloat("hottest_temperature", m_particle_system->getHottestTemperature());
	temp_shader->SetUniformFloat("coolest_temperature", m_particle_system->getCoolestTemperature());

	const std::shared_ptr<Shader> shader = (m_use_temperature_shader) ? temp_shader: point_shader;

	if (m_b_sph_visibility)
	{
		// fluid particles
		const ParticleSet* const particles = m_particle_system->getSPHParticles();

#ifdef _DEBUG
		assert(shader);
#endif
		shader->Use();
		point_shader->SetUniformVec3("point_color", glm::vec3(0.7f, 0.7f, 1.f));
		glBindVertexArray(m_particle_system->getSPH_VAO());
		glBindBuffer(GL_ARRAY_BUFFER, m_particle_system->getSPH_VBO_0());
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
		//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_particle_system->getEBO());

		glBindBuffer(GL_ARRAY_BUFFER, m_particle_system->getSPH_VBO_1());
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);

		glDrawArrays(GL_POINTS, 0, m_particle_system->getSPHParticles()->m_size);

		glBindVertexArray(0);
	}
	if (m_b_dem_visibility)
	{
		/*
		const std::shared_ptr<Shader> shader = m_resource_manager->FindShaderByName("WetPointSprite");
		const glm::mat4 pvm = m_mainCamera->m_cameraMat * glm::mat4(1);
		shader->SetUniformMat4("pvm", pvm);
		shader->SetUniformFloat("point_size", 30.f);
		shader->SetUniformVec3("light_pos", m_mainCamera->m_position);
		shader->SetUniformVec3("camera_pos", m_mainCamera->m_position);
		shader->SetUniformMat4("view", m_mainCamera->m_lookAt);
		*/
		// dem particles
		const ParticleSet* const dem_particles = m_particle_system->getDEMParticles();
#ifdef _DEBUG
		assert(shader);
#endif
		shader->Use();

		point_shader->SetUniformVec3("point_color", glm::vec3(0.0f, 0.7f, 0.35f));
		glBindVertexArray(m_particle_system->getDEMVAO());
		glBindBuffer(GL_ARRAY_BUFFER, m_particle_system->getDEM_VBO_0());
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
		//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_particle_system->getEBO());
		
		glBindBuffer(GL_ARRAY_BUFFER, m_particle_system->getDEM_VBO_1());
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
		
		glDrawArrays(GL_POINTS, 0, m_particle_system->getDEMParticles()->m_size);
		
		glBindVertexArray(0);
	}

	if (m_b_boundary_visibility)
	{
		// render boundary particles
		const ParticleSet* const boundary_particles = m_particle_system->getBoundaryParticles();

		point_shader->SetUniformVec3("point_color", glm::vec3(1.f, 1.f, 1.f));

		glBindVertexArray(m_particle_system->getBoundaryVAO());
		glBindBuffer(GL_ARRAY_BUFFER, m_particle_system->getBoundaryVBO());
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

		glDrawArrays(GL_POINTS, 0, m_particle_system->getBoundaryParticles()->m_size);

		glBindVertexArray(0);
	}
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

void Renderer::RenderFluidDepth()
{
	const std::shared_ptr<Shader> shader = m_resource_manager->FindShaderByName("PointDepth");
	
	const glm::mat4 pvm = m_mainCamera->m_cameraMat * glm::mat4(1);
	glm::mat4 model_view = m_mainCamera->m_lookAt * glm::mat4(1);
	
	shader->SetUniformMat4("pvm", pvm);
	shader->SetUniformFloat("point_size", 30.f);
	shader->SetUniformVec3("light_pos", m_mainCamera->m_position);
	shader->SetUniformVec3("camera_pos", m_mainCamera->m_position);
	shader->SetUniformMat4("view", m_mainCamera->m_lookAt);
	shader->SetUniformVec3("point_color", glm::vec3(0.7f, 0.7f, 1.f));
	shader->SetUniformMat4("model_view", model_view);
	shader->SetUniformMat4("projection", m_mainCamera->m_projection);
	shader->SetUniformFloat("sphere_radius", m_particle_system->getParticleRadius());
	
	//shader->SetUniformVec3("light_pos", m_mainCamera->m_position);
	//shader->SetUniformVec3("camera_pos", m_mainCamera->m_position);
	//shader->SetUniformMat4("view", m_mainCamera->m_lookAt);


	if (m_b_sph_visibility)
	{
		// fluid particles
		const ParticleSet* const particles = m_particle_system->getSPHParticles();

#ifdef _DEBUG
		assert(shader);
#endif
		shader->Use();
		//point_shader->SetUniformVec3("point_color", glm::vec3(0.7f, 0.7f, 1.f));

		// render to depth map fbo
		glBindFramebuffer(GL_FRAMEBUFFER, m_depth_map_fbo);
		
		glBindVertexArray(m_particle_system->getSPH_VAO());
		glDrawArrays(GL_POINTS, 0, m_particle_system->getSPHParticles()->m_size);
		glBindVertexArray(0);
	}

	

}

void Renderer::SmoothDepth()
{
	if (!m_b_smooth_depth)
		return;

	const std::shared_ptr<Shader> shader = m_resource_manager->FindShaderByName("DepthSmooth");
	shader->Use();

	// reset
	// unbind to make sure not rendering wrong framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, m_viewport_width, m_viewport_height);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// activate texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_depth_map);

	// set uniforms
	shader->SetUniformInt("depth_map", m_depth_map);
	shader->SetUniformFloat("filter_radius", 3);
	shader->SetUniformFloat("blur_scale", 0.01f);
	
	// vertical blur
	shader->SetUniformVec2("blur_dir", m_blur_dirY);

	//glBindFramebuffer(GL_FRAMEBUFFER, m_depth_map_fbo);


	glBindVertexArray(m_screen_vao);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
	//glBindFramebuffer(GL_FRAMEBUFFER, 0);

	/*
	// Horizontal blur
	shader->SetUniformVec2("blur_dir", m_blur_dirX);

	glBindFramebuffer(GL_FRAMEBUFFER, m_depth_map_fbo);

	glBindVertexArray(m_screen_vao);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
	*/

}
