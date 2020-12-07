#ifndef _RENDERER_H_
#define _RENDERER_H_

#include <memory>
#include "ResourceManager.h"
#include "Camera.h"
#include "ParticleSystem.h"
#include "RenderTargetTexture.h"

class Renderer;

/*
class RenderTargetTexture
{
	GLuint m_texture;
	GLuint m_fbo;
};
*/

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
	//void SwitchDepthSmooth();
	void SwtichRenderFluid();

	// setters
	void setMainCamera(std::shared_ptr<Camera> camera);
	void setClearColor(glm::vec4 clear_color);

private:

	void InitializeSSFRBuffers();

	void DisableGLFunctions();

	void ClearBuffer();

	void SetupUniformBufferOject();

	void RenderObjects();
	// Use instancing
	void RenderParticles();

	void RenderGameObject(
		const std::shared_ptr<Shader>& shader,
		const std::shared_ptr<Mesh>& mesh,
		const Transform& transform);

	// SSFR
	void RenderFluidDepth();
	void SmoothDepth();
	void RenderThickness();
	void RenderFluid();


	std::shared_ptr<Camera> m_mainCamera;
	std::shared_ptr<ResourceManager> m_resource_manager;
	std::shared_ptr<ParticleSystem> m_particle_system;

	GLsizei m_viewport_width;
	GLsizei m_viewport_height;

	GLuint m_ubo;

	//GLuint m_depth_map;
	//GLuint m_depth_map_fbo;
	//GLuint m_depth_smooth_fbo;
	RenderTargetTexture m_rtt_scene;
	RenderTargetTexture m_rtt_depth;
	RenderTargetTexture m_rtt_blurX;
	RenderTargetTexture m_rtt_blurY;
	RenderTargetTexture m_rtt_thickness;


	GLuint m_screen_vao;
	GLuint m_screen_vbo;

	glm::vec2 m_screen_size;
	glm::vec2 m_blur_dirX;
	glm::vec2 m_blur_dirY;

	glm::vec4 m_clear_color;

	float m_point_size;

	bool m_b_sph_visibility;
	bool m_b_dem_visibility;
	bool m_b_boundary_visibility;
	//bool m_b_smooth_depth;
	bool m_b_render_fluid;

	bool m_b_use_temperature_shader;


	GLuint debug_texture;
};

#endif
