#ifndef _RENDERER_H_
#define _RENDERER_H_

#include <memory>
#include "ResourceManager.h"
#include "Camera.h"
#include "ParticleSystem.h"

class Renderer;

class Renderer
{
public:
	Renderer();
	~Renderer();

	void Initialize(
		std::shared_ptr<ResourceManager> resource_manager,
		std::shared_ptr<ParticleSystem> particle_system,
		std::shared_ptr<Camera> camera,
		int viewport_width, int viewport_height
	);

	void Render();

	void SwitchSphVisibility();
	void SwitchDEMVisibility();
	void SwitchBoundaryVisibility();
	void SwitchTemperatureShader();

	// setters
	void setMainCamera(std::shared_ptr<Camera> camera);
	void setClearColor(glm::vec4 clear_color);

private:

	void InitializeDepthBuffers();

	void ClearBuffer();

	void SetupUniformBufferOject();

	void RenderObjects();
	// Use instancing
	void RenderParticles();

	void RenderGameObject(
		const std::shared_ptr<Shader>& shader,
		const std::shared_ptr<Mesh>& mesh,
		const Transform& transform);

	void RenderFluidDepth();
	void SmoothDepth();

	std::shared_ptr<Camera> m_mainCamera;
	std::shared_ptr<ResourceManager> m_resource_manager;
	std::shared_ptr<ParticleSystem> m_particle_system;

	GLsizei m_viewport_width;
	GLsizei m_viewport_height;

	GLuint m_ubo;

	GLuint m_depth_map;
	GLuint m_depth_map_fbo;

	glm::vec4 m_clear_color;

	bool m_sph_visibility;
	bool m_dem_visibility;
	bool m_boundary_visibility;

	bool m_use_temperature_shader;

};

#endif
