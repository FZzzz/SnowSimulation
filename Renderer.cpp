#include "Renderer.h"
#include "GLFunctions.h"
#include <iostream>

Renderer::Renderer() :
	m_mainCamera(nullptr),
	m_resource_manager(nullptr),
	m_viewport_width(1600),
	m_viewport_height(900),
	m_clear_color(0.15f, 0.15f, 0.15f, 0),
	m_point_size(30.f),
	m_ubo(-1),
	m_b_sph_visibility(true),
	m_b_dem_visibility(true),
	m_b_boundary_visibility(false),
	//m_b_smooth_depth(false),
	m_b_render_fluid(true),
	m_b_use_temperature_shader(false),
	m_b_use_mc_mesh(false)
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

	InitializeSSFRBuffers();
}

void Renderer::InitializeSSFRBuffers()
{
	m_rtt_scene.SetupColorAttachment(m_viewport_width, m_viewport_height, true);
	m_rtt_depth.SetupDepthAttachment(m_viewport_width, m_viewport_height);
	m_rtt_scene_depth.SetupDepthAttachment(m_viewport_width, m_viewport_height);
	m_rtt_blurX.SetupDepthAttachment(m_viewport_width, m_viewport_height);
	m_rtt_blurY.SetupDepthAttachment(m_viewport_width, m_viewport_height);
	m_rtt_thickness.SetupColorAttachment(m_viewport_width, m_viewport_height);

	/*
	string path = "capture.JPG";
	debug_texture = GLFunctions::LoadTexture(path.c_str(), false);
	*/

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

void Renderer::DisableGLFunctions()
{
	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
	//glDisable(GL_POINT_SPRITE);
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

	if (m_b_use_mc_mesh)
	{

	}
	else
	{
		//RenderObjects();
		RenderScene();

		if (m_b_sph_visibility && m_b_render_fluid)
		{
			RenderSceneDepth();
			RenderFluidDepth();
			SmoothDepth();
			RenderThickness();
			RenderFluid();
		}
	}
	
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
	m_b_use_temperature_shader = !m_b_use_temperature_shader;
}

void Renderer::SwitchRenderFluid()
{
	m_b_render_fluid = !m_b_render_fluid;
	std::cout << "Status: " << ((m_b_render_fluid) ? "On" : "Off") << std::endl;
}

void Renderer::SwitchMCMeshRender()
{
	m_b_use_mc_mesh = !m_b_use_mc_mesh;
}

void Renderer::SetUpFluidMeshInfo(const std::vector<glm::vec3>& vert_pos, const std::vector<glm::vec3>& vert_normal, const std::vector<unsigned int>& indices)
{
}

/*
void Renderer::SwitchDepthSmooth()
{
	m_b_smooth_depth = !m_b_smooth_depth;
	std::cout << "Status: " << ((m_b_smooth_depth) ? "On" : "Off") << std::endl;
}
*/


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
	//glBindBuffer(GL_UNIFORM_BUFFER, 0);
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
	//glActiveTexture(GL_TEXTURE0);
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

void Renderer::RenderGameObjectDepth(const std::shared_ptr<Shader>& shader, const std::shared_ptr<Mesh>& mesh, const Transform& transform)
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
	shader->SetUniformMat4("pvm", pvm);

	// render
	glBindVertexArray(mesh->getVAO());
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->getEBO());
	glDrawElements(
		GL_TRIANGLES,
		static_cast<unsigned int>(mesh->getNumberOfIndices()),
		GL_UNSIGNED_INT,
		0);
	glBindVertexArray(0);
}

void Renderer::RenderObjectsDepth()
{
	const auto obj_vec = m_resource_manager->getObjects();
	const auto static_obj_vec = m_resource_manager->getStaticObjects();
	const auto shader = m_resource_manager->FindShaderByName("SimpleDepth");
	
	shader->Use();
	
	for (auto obj_it = obj_vec.cbegin(); obj_it != obj_vec.cend(); ++obj_it)
	{
		RenderGameObjectDepth(shader, (*obj_it)->getMesh(), (*obj_it)->m_transform);
	}

	for (auto obj_it = static_obj_vec.cbegin(); obj_it != static_obj_vec.cend(); ++obj_it)
	{
		RenderGameObjectDepth(shader, (*obj_it)->getMesh(), (*obj_it)->m_transform);
	}
}

void Renderer::RenderScene()
{
	// Good habbit to reset :>
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, m_viewport_width, m_viewport_height);
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(m_clear_color.r, m_clear_color.g, m_clear_color.b, m_clear_color.a);

	const glm::mat4 pvm = m_mainCamera->m_cameraMat * glm::mat4(1);

	// Get shaders
	const std::shared_ptr<Shader> point_shader = m_resource_manager->FindShaderByName("PointSprite");
	const std::shared_ptr<Shader> temp_shader = m_resource_manager->FindShaderByName("PointSpriteTemperature");

	// select shader
	const std::shared_ptr<Shader> shader = (m_b_use_temperature_shader) ? temp_shader : point_shader;
	shader->Use();

	// set uniforms
	shader->SetUniformMat4("pvm", pvm);
	shader->SetUniformFloat("point_size", m_point_size);
	shader->SetUniformVec3("light_pos", m_mainCamera->m_position);
	shader->SetUniformVec3("camera_pos", m_mainCamera->m_position);
	shader->SetUniformMat4("view", m_mainCamera->m_lookAt);

	if (m_b_use_temperature_shader)
	{
		shader->SetUniformVec3("hottest_color", glm::vec3(1, 0, 0));
		shader->SetUniformVec3("coolest_color", glm::vec3(0, 0, 1));
		shader->SetUniformFloat("hottest_temperature", m_particle_system->getHottestTemperature());
		shader->SetUniformFloat("coolest_temperature", m_particle_system->getCoolestTemperature());
	}

	// enable opengl functions
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	//glEnable(GL_POINT_SPRITE);
	glEnable(GL_DEPTH_TEST);
	//glEnable(GL_BLEND);
	//glDepthMask(GL_TRUE);

	if (m_b_sph_visibility && !m_b_render_fluid)
	{
		// fluid particles
		const ParticleSet* const particles = m_particle_system->getSPHParticles();

#ifdef _DEBUG
		assert(shader);
#endif
		// set point color
		if (!m_b_use_temperature_shader)
			shader->SetUniformVec3("point_color", glm::vec3(0.7f, 0.7f, 1.f));

		glBindVertexArray(m_particle_system->getSPH_VAO());
		glDrawArrays(GL_POINTS, 0, m_particle_system->getSPHParticles()->m_size);
		glBindVertexArray(0);
	}

	if (m_b_dem_visibility)
	{
		// dem particles
		const ParticleSet* const dem_particles = m_particle_system->getDEMParticles();
#ifdef _DEBUG
		assert(shader);
#endif
		// set point color
		if (!m_b_use_temperature_shader)
			shader->SetUniformVec3("point_color", glm::vec3(0.75f, 0.75f, 0.75f));

		// if we are rendering the fluid, render to scene fbo
		if (m_b_render_fluid)
		{
			glBindFramebuffer(GL_FRAMEBUFFER, m_rtt_scene.m_fbo);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		}

		// render
		glBindVertexArray(m_particle_system->getDEMVAO());
		glDrawArrays(GL_POINTS, 0, m_particle_system->getDEMParticles()->m_size);
		glBindVertexArray(0);
	}
	else
	{
		// if the dem visibility is set to false but still rendering fluid, render nothing to scene fbo
		if (m_b_render_fluid)
		{
			glBindFramebuffer(GL_FRAMEBUFFER, m_rtt_scene.m_fbo);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			
			//render nothing
			glBindVertexArray(m_particle_system->getDEMVAO());
			glDrawArrays(GL_POINTS, 0, 0);
			glBindVertexArray(0);
		}
	}

	if (m_b_boundary_visibility)
	{
		// render boundary particles
		const ParticleSet* const boundary_particles = m_particle_system->getBoundaryParticles();
		point_shader->Use();
		shader->SetUniformMat4("pvm", pvm);
		shader->SetUniformFloat("point_size", m_point_size);
		shader->SetUniformVec3("light_pos", m_mainCamera->m_position);
		shader->SetUniformVec3("camera_pos", m_mainCamera->m_position);
		shader->SetUniformMat4("view", m_mainCamera->m_lookAt);
		point_shader->SetUniformVec3("point_color", glm::vec3(1.f, 1.f, 1.f));

		glBindVertexArray(m_particle_system->getBoundaryVAO());
		glDrawArrays(GL_POINTS, 0, m_particle_system->getBoundaryParticles()->m_size);
		glBindVertexArray(0);
	}

	// disable when not using 
	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);

	RenderObjects();
}

void Renderer::RenderSceneDepth()
{
	// Good habbit to reset :>
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, m_viewport_width, m_viewport_height);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(m_clear_color.r, m_clear_color.g, m_clear_color.b, m_clear_color.a);

	const glm::mat4 pvm = m_mainCamera->m_cameraMat * glm::mat4(1);
	const glm::mat4 model_view = m_mainCamera->m_lookAt * glm::mat4(1);

	// Get shaders
	const std::shared_ptr<Shader> shader = m_resource_manager->FindShaderByName("PointDepth");

	shader->Use();

	// set uniforms
	shader->SetUniformMat4("pvm", pvm);
	shader->SetUniformFloat("point_size", m_point_size);
	shader->SetUniformVec3("camera_pos", m_mainCamera->m_position);
	shader->SetUniformMat4("view", m_mainCamera->m_lookAt);
	shader->SetUniformMat4("model_view", model_view);

	// enable opengl functions
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	// bind scene depth map fbo
	glBindFramebuffer(GL_FRAMEBUFFER, m_rtt_scene_depth.m_fbo);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	if (m_b_dem_visibility)
	{
		// dem particles
		const ParticleSet* const dem_particles = m_particle_system->getDEMParticles();
#ifdef _DEBUG
		assert(shader);
#endif
		// set point color
		if (!m_b_use_temperature_shader)
			shader->SetUniformVec3("point_color", glm::vec3(0.75f, 0.75f, 0.75f));


		// render
		glBindVertexArray(m_particle_system->getDEMVAO());
		glDrawArrays(GL_POINTS, 0, m_particle_system->getDEMParticles()->m_size);
		glBindVertexArray(0);
	}
	else
	{
		// if the dem visibility is set to false but still rendering fluid, render nothing to scene fbo
		if (m_b_render_fluid)
		{
			glBindFramebuffer(GL_FRAMEBUFFER, m_rtt_scene.m_fbo);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			//render nothing
			glBindVertexArray(m_particle_system->getDEMVAO());
			glDrawArrays(GL_POINTS, 0, 0);
			glBindVertexArray(0);
		}
	}

	if (m_b_boundary_visibility)
	{
		// render boundary particles
		const ParticleSet* const boundary_particles = m_particle_system->getBoundaryParticles();
		
		glBindVertexArray(m_particle_system->getBoundaryVAO());
		glDrawArrays(GL_POINTS, 0, m_particle_system->getBoundaryParticles()->m_size);
		glBindVertexArray(0);
	}
	// disable when not using 
	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);

	RenderObjectsDepth();
}

void Renderer::RenderFluidDepth()
{
	const std::shared_ptr<Shader> shader = m_resource_manager->FindShaderByName("PointDepth");
	
	const glm::mat4 pvm = m_mainCamera->m_cameraMat * glm::mat4(1);
	glm::mat4 model_view = m_mainCamera->m_lookAt * glm::mat4(1);
	
	// reset
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, m_viewport_width, m_viewport_height);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(m_clear_color.r, m_clear_color.g, m_clear_color.b, m_clear_color.a);

	shader->Use();

	shader->SetUniformMat4("pvm", pvm);
	shader->SetUniformFloat("point_size", m_point_size);
	shader->SetUniformVec3("light_pos", m_mainCamera->m_position);
	shader->SetUniformVec3("camera_pos", m_mainCamera->m_position);
	shader->SetUniformMat4("view", m_mainCamera->m_lookAt);
	shader->SetUniformVec3("point_color", glm::vec3(0.7f, 0.7f, 1.f));
	shader->SetUniformMat4("model_view", model_view);
	shader->SetUniformMat4("projection", m_mainCamera->m_projection);
	shader->SetUniformFloat("sphere_radius", m_particle_system->getParticleRadius());
	shader->SetUniformFloat("near_plane", 0.01f);
	shader->SetUniformFloat("far_plane", 15.f);
	
	if (m_b_sph_visibility)
	{
		// fluid particles
		const ParticleSet* const particles = m_particle_system->getSPHParticles();

#ifdef _DEBUG
		assert(shader);
#endif

		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
		glDepthMask(GL_TRUE); // Allow to write depth buffer

		// write to fbo
		glBindFramebuffer(GL_FRAMEBUFFER, m_rtt_depth.m_fbo);
		glClear(GL_DEPTH_BUFFER_BIT);

		//render
		glBindVertexArray(m_particle_system->getSPH_VAO());
		glDrawArrays(GL_POINTS, 0, m_particle_system->getSPHParticles()->m_size);
		glBindVertexArray(0);

		// unbind
		glBindFramebuffer(GL_FRAMEBUFFER, 0);


		glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
	}
}

void Renderer::SmoothDepth()
{
	const std::shared_ptr<Shader> shader = m_resource_manager->FindShaderByName("DepthSmooth");
	shader->Use();
	
	// reset
	// unbind to make sure not rendering wrong framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, m_viewport_width, m_viewport_height);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(m_clear_color.r, m_clear_color.g, m_clear_color.b, m_clear_color.a);

	// -------------------vertical blur---------------------

	// activate texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_rtt_depth.m_texture);
	
	// set uniforms
	shader->SetUniformInt("depth_map", 0);
	shader->SetUniformFloat("filter_radius", 3);
	shader->SetUniformFloat("blur_scale", 0.01f);
	shader->SetUniformVec2("blur_dir", m_blur_dirY);
	shader->SetUniformFloat("near_plane", 0.01f);
	shader->SetUniformFloat("far_plane", 15.f);

	glBindFramebuffer(GL_FRAMEBUFFER, m_rtt_blurY.m_fbo);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	glBindVertexArray(m_screen_vao);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);


	// -------------------Horizontal blur---------------------
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, m_viewport_width, m_viewport_height);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(m_clear_color.r, m_clear_color.g, m_clear_color.b, m_clear_color.a);

	// active & bind texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_rtt_blurY.m_texture);

	// set uniforms
	shader->SetUniformInt("depth_map", 0);
	shader->SetUniformVec2("blur_dir", m_blur_dirX);

	// bind framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, m_rtt_blurX.m_fbo);
	glClear(GL_DEPTH_BUFFER_BIT);

	glBindVertexArray(m_screen_vao);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);

}

void Renderer::RenderThickness()
{	
	const std::shared_ptr<Shader> shader = m_resource_manager->FindShaderByName("Thickness");
	shader->Use();

	// reset
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, m_viewport_width, m_viewport_height);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(m_clear_color.r, m_clear_color.g, m_clear_color.b, m_clear_color.a);

	// set uniforms
	const glm::mat4 pvm = m_mainCamera->m_cameraMat * glm::mat4(1);
	glm::mat4 model_view = m_mainCamera->m_lookAt * glm::mat4(1);
	shader->SetUniformMat4("model_view", model_view);
	shader->SetUniformMat4("pvm", pvm);
	shader->SetUniformFloat("point_size", m_point_size);
	shader->SetUniformFloat("thickness_unit", 0.1f);

	// Enable functions
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE); // C_{final} = C_a * 1 + C_b * 1
	glBlendEquation(GL_FUNC_ADD); // add to get the thickness

	glDepthMask(GL_FALSE); // forbbit shader writting to depth buffer

	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	////glEnable(GL_POINT_SPRITE);

	// bind framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, m_rtt_thickness.m_fbo);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// render particle thickness
	glBindVertexArray(m_particle_system->getSPH_VAO());
	glDrawArrays(GL_POINTS, 0, m_particle_system->getSPHParticles()->m_size);
	glBindVertexArray(0);
	//glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
	//glDisable(GL_POINT_SPRITE);
	glDisable(GL_BLEND); // disable blend for safety
	glDepthMask(GL_TRUE);
}

void Renderer::RenderFluid()
{
	const std::shared_ptr<Shader> shader = m_resource_manager->FindShaderByName("SSFR_Fluid");
	shader->Use();

	// reset
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, m_viewport_width, m_viewport_height);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(m_clear_color.r, m_clear_color.g, m_clear_color.b, m_clear_color.a);

	// active & bind texture
	// depth texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_rtt_blurX.m_texture);

	// thickness texture
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, m_rtt_thickness.m_texture);

	// scene texture
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, m_rtt_scene.m_texture);

	// scene depth texture
	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, m_rtt_scene_depth.m_texture);

	// set uniforms
	const glm::mat4 pvm = m_mainCamera->m_cameraMat * glm::mat4(1);
	glm::mat4 model_view = m_mainCamera->m_lookAt * glm::mat4(1);
	shader->SetUniformInt("depth_map", 0);
	shader->SetUniformInt("thickness_map", 1);
	shader->SetUniformInt("scene_map", 2);
	shader->SetUniformInt("scene_depth_map", 3);
	shader->SetUniformVec4("light_color", glm::vec4(1,1,1,0.8f));
	shader->SetUniformVec3("light_pos", m_mainCamera->m_position);
	shader->SetUniformMat4("projection", m_mainCamera->m_projection);
	shader->SetUniformMat4("model_view", model_view);
	shader->SetUniformVec2("inv_tex_scale", glm::vec2(1.f / (float)m_viewport_width, 1.f / (float)m_viewport_height));
	
	// render
	glBindVertexArray(m_screen_vao);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
}

void Renderer::RenderSceneWithMCMesh()
{
}
