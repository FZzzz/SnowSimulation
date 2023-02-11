#ifndef _GLFWAPP_H
#define _GLFWAPP_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>
#include <glm/glm.hpp>
#include <memory>
#include <chrono>
#include <unordered_map>
#include "imgui/imgui.h"
#include "GameObject.h"
#include "ResourceManager.h"
#include "Camera.h"
#include "Renderer.h"
#include "AssetImporter.h"
#include "GUIManager.h"
#include "Simulation.h"
#include "ParticleSystem.h"

class GLFWApp
{
public:
	
	~GLFWApp();
	
	bool Initialize(int width , int height , const std::string &title);
	void Run();
	void ReleaseResources();
	
	void ExportBgeoFile();

	static GLFWApp* getInstance() {
		if (!appInstance)
			appInstance = new GLFWApp();
		return appInstance; 
	};

	inline GLFWwindow* getGLFWwindow() { return m_window; };
	inline std::shared_ptr<ResourceManager> getResourceManager() { return m_resource_manager; }
	inline std::shared_ptr<GUIManager> getGUIManager() { return m_gui_manager; }
	inline const std::shared_ptr<Renderer> getRenderer() { return m_renderer; }
	inline const std::shared_ptr<Simulation> getSimulator() { return m_simulator; }
	inline const std::shared_ptr<Camera> getMainCamera() { return m_mainCamera; };
	inline bool GetAppStatus() { return m_app_status; };
	
	/*virtual functions*/
	virtual float getElapsedTime();

	/* Mouse controls */
	float m_mouse_last_x;
	float m_mouse_last_y;
	bool m_mouse_pressed;
	bool m_b_record;

	unsigned int m_frame_count;
	unsigned int m_record_clip_index;

	

private:

	GLFWApp();
	
	void Render();
	void Update();
	void SignalFail();
	void SetUpImGui();

	static GLFWApp* appInstance;

	GLFWwindow* m_window;
	GLFWcursor* m_mouseCursors[ImGuiMouseCursor_COUNT];

	double m_previousTime, m_currentTime, m_deltaTime;
	float m_time_step;
	
	/*Importer*/
	std::shared_ptr<AssetImporter> m_importer;

	/*Manager*/
	std::shared_ptr<ResourceManager> m_resource_manager;
	std::shared_ptr<GUIManager> m_gui_manager;
	
	/*Resources*/
	std::shared_ptr<Camera> m_mainCamera;
	std::shared_ptr<Renderer> m_renderer;
	
	/* Timer */
	std::chrono::high_resolution_clock::time_point t0, t1, t2, t3, t4;

	/*Simulation*/
	std::shared_ptr<Simulation> m_simulator;
	std::shared_ptr<ParticleSystem> m_particle_system;

	

	/*GUIs*/
	//void Frame_Status_GUI();
	//void Object_Viewer_GUI();

	bool m_app_status;
};

#endif
