#include "GLFWApp.h"
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include "imgui/imgui_impl_glfw_gl3.h"
#ifdef _WIN32
#undef APIENTRY
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#include <GLFW/glfw3native.h>
#include <chrono>
#endif

#include "GameObject.h"
#include "Plane.h"

void Error_callback(int error, const char* description);
void Key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void Mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void Mouse_cursor_callback(GLFWwindow* window, double xpos, double ypos);
void Mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);


void Frame_Status_GUI();
void Object_Viewer_GUI();
void Mouse_Position_GUI();

GLFWApp* GLFWApp::appInstance;

GLFWApp::GLFWApp() : 
	m_previousTime(0), 
	m_currentTime(0), 
	m_app_status(true),
	m_resource_manager(nullptr),
	m_gui_manager(nullptr),
	m_mainCamera(nullptr),
	m_renderer(nullptr),
	m_mouse_pressed(false),
	m_frame_count(0)
{
}

GLFWApp::~GLFWApp()
{
	m_resource_manager.reset();
	m_mainCamera.reset();
	m_renderer.reset();
}

bool GLFWApp::Initialize(int width , int height , const std::string &title)
{
	
	srand(time(NULL));

	if (!glfwInit())
	{
		return false;
	}

	const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

	const float f_width = 0.8f * static_cast<float>(mode->width);
	const float f_height = 0.8f * static_cast<float>(mode->height);

	glfwSetErrorCallback(Error_callback);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_SAMPLES, 4);

	/* Setting GLFW window */
	m_window = glfwCreateWindow((int)f_width, (int)f_height, title.c_str() , NULL , NULL);
	
	if (!m_window)
	{
		std::cout << "Window Creation Failed!" << std::endl;
		return false;
	}

	glfwSetWindowPos(m_window, 100, 100);
	glfwMakeContextCurrent(m_window);
	glfwSetKeyCallback(m_window, Key_callback);
	glfwSetMouseButtonCallback(m_window, Mouse_button_callback);
	glfwSetCursorPosCallback(m_window, Mouse_cursor_callback);
	glfwSetScrollCallback(m_window, Mouse_scroll_callback);
	glfwSwapInterval(0);

	// Initialize glew
	glewExperimental = true;
	GLenum status = glewInit();
	if (status != GLEW_OK)
	{
		std::cout << "GLEW INIT Failed! " << std::endl;
		return false;
	}
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_MULTISAMPLE);
	//glEnable(GL_BLEND);
	glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_POINT_SPRITE); 
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//glDepthFunc(GL_LESS);

	/* ResourceManager Creation */
	m_resource_manager = std::make_shared<ResourceManager>();


	// Initialize asset importer
	m_importer = std::make_shared<AssetImporter>();

	// Initialize particle system
	m_particle_system = std::make_shared<ParticleSystem>();

	/* Simulator creation */
	m_simulator = std::make_shared<Simulation>();
	
#ifdef _DEBUG
	assert(m_resource_manager != nullptr && 
		m_importer != nullptr && 
		m_particle_system != nullptr 
		&& m_simulator != nullptr);
#endif

	// ShowdowMapping shader settings
	std::shared_ptr<Shader> shadow_mapping_shader = std::make_shared<Shader>("ShadowMapping");
	shadow_mapping_shader->SetupShader("resources/shader/shadow_mapping_vs_adv.glsl",
		"resources/shader/shadow_mapping_fs.glsl");
	
	std::shared_ptr<Shader> point_shader = std::make_shared<Shader>("PointSprite");
	point_shader->SetupShader("resources/shader/point_sprite_vs.glsl", 
		"resources/shader/point_sprite_fs.glsl");

	std::shared_ptr<Shader> point_temp_shader = std::make_shared<Shader>("PointSpriteTemperature");
	point_temp_shader->SetupShader("resources/shader/point_sprite_temperature_vs.glsl",
		"resources/shader/point_sprite_fs.glsl");

	
	std::shared_ptr<Shader> point_depth_shader = std::make_shared<Shader>("PointDepth");
	point_depth_shader->SetupShader("resources/shader/point_depth_vs.glsl",
		"resources/shader/point_depth_fs.glsl");
	
	std::shared_ptr<Shader> depth_smooth_shader = std::make_shared<Shader>("DepthSmooth");
	depth_smooth_shader->SetupShader("resources/shader/bilateral_vs.glsl",
		"resources/shader/bilateral_fs.glsl");

	std::shared_ptr<Shader> thickness_shader = std::make_shared<Shader>("Thickness");
	thickness_shader->SetupShader("resources/shader/ssfr_thickness_vs.glsl",
		"resources/shader/ssfr_thickness_fs.glsl");


	auto mat_uniform = glGetUniformBlockIndex(shadow_mapping_shader->getProgram(), "Matrices");
	GLuint ubo;
	glGenBuffers(1, &ubo);
	glBindBuffer(GL_UNIFORM_BUFFER, ubo);
	glBufferData(GL_UNIFORM_BUFFER, 3 * sizeof(glm::mat4), NULL, GL_STATIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
	// define the range of the buffer that links to a uniform binding point
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, ubo, 0, 3 * sizeof(glm::mat4));

	// Camera Setting
	{
		CameraDesc camera_desc;
		camera_desc.fov = 45.0f;
		camera_desc.screen_width = f_width;
		camera_desc.screen_height = f_height;
		camera_desc.near_plane = 0.001f;
		camera_desc.far_plane = 1000.0f;
		camera_desc.position = glm::vec3(0.0f, 0.0f, 5.0f);
		camera_desc.target_position = glm::vec3(0, 0, 0);
		camera_desc.ubo = ubo;

		//Main Camera setting
		m_mainCamera = std::make_shared<Camera>(camera_desc);
		m_resource_manager->SetMainCamera(m_mainCamera);

		//Renderer setting
		m_renderer = std::make_shared<Renderer>();
		m_renderer->Initialize(m_resource_manager, m_particle_system, m_mainCamera, (int)f_width, (int)f_height);
	}
	
	// Load Meshes
	//std::shared_ptr<GameObject> obj = std::make_shared<GameObject>();	
	{
		//default mesh
		auto load_status = IMPORT_STATUS::IMPORT_FAILED;
		auto mesh = m_importer->LoadMesh("resources/models/monkey.obj", shadow_mapping_shader, load_status);
		mesh->Initialize(shadow_mapping_shader);
		if (load_status == IMPORT_STATUS::IMPORT_FAILED)
		{
			std::cout << "Load Failed\n";
			SignalFail();
		}
	}
	
		
	// Terrain Initilization
	{
		
		std::shared_ptr<Plane> plane_terrain = std::make_shared<Plane>();
		plane_terrain->Initialize(glm::vec3(0, 0, 0), shadow_mapping_shader);
		m_resource_manager->AddGameObject(static_pointer_cast<GameObject>(plane_terrain));
		m_simulator->AddCollider(plane_terrain->getCollider());
		
		
		auto collider = new PlaneCollider(glm::vec3(1, 0, 0), -10);
		m_simulator->AddCollider(collider);
		
		collider = new PlaneCollider(glm::vec3(-1, 0, 0), -10);
		m_simulator->AddCollider(collider);
		collider = new PlaneCollider(glm::vec3(0, 0, 1), -15 );
		m_simulator->AddCollider(collider);
		collider = new PlaneCollider(glm::vec3(0, 0, -1), 1);
		m_simulator->AddCollider(collider);
		
	}

	//GenerateRadomParticles();
	//GenerateFluidParticles();
	/*
	 *	End of resource settings
	 */

	/*
	 *	UI settings
	 */

	 /* GUI Initialize */
	m_gui_manager = std::make_shared<GUIManager>();
	m_gui_manager->Initialize(m_window);
	m_gui_manager->AddGUIFunction(std::bind(Frame_Status_GUI));
	//m_gui_manager->AddGUIFunction(std::bind(Mouse_Position_GUI));
	//m_gui_manager->AddGUIFunction(std::bind(Object_Viewer_GUI));
	//m_gui_manager->AddGUIFunction(std::bind(Animated_Character_GUI));
	
	/* Managers initialization */
	m_resource_manager->ArrangeStaticObjects();
	m_simulator->Initialize(PBD_MODE::XPBD, m_particle_system);
	//m_simulator->SetSolverIteration(2);

	// Simulation control settings
	{
		uint32_t iterations = 3;
		int clip_length = 200000;
		m_simulator->SetSolverIteration(iterations);
		m_simulator->setClipLength(clip_length);
	}

	return true;
}

void GLFWApp::Run()
{

	while (!glfwWindowShouldClose(m_window))
	{
		t0 = std::chrono::high_resolution_clock::now();
		ImGui_ImplGlfwGL3_NewFrame();

		//Frame_Status_GUI();
		//Object_Viewer_GUI();

		t1 = std::chrono::high_resolution_clock::now();
		Update();
		t2 = std::chrono::high_resolution_clock::now();
		Render();
		t3 = std::chrono::high_resolution_clock::now();

		glfwSwapBuffers(m_window);
		glfwPollEvents();
		t4 = std::chrono::high_resolution_clock::now();;
	}
	
	ImGui_ImplGlfwGL3_Shutdown();
	ImGui::DestroyContext();
	glfwTerminate();
}


void GLFWApp::ReleaseResources()
{
	m_resource_manager->ShutDown();
	m_particle_system->Release();
}

void GLFWApp::SignalFail()
{
	m_app_status = false;
}

float GLFWApp::getElapsedTime()
{
	return static_cast<float>(glfwGetTime());
}

void Error_callback(int error, const char* description)
{
	fputs(description, stderr);
}

void Key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	static GLFWApp* const instance = GLFWApp::getInstance();
	static auto res_manager = instance->getResourceManager();
	static auto camera = res_manager->getMainCamera();
	if (action == GLFW_RELEASE)
	{
		switch (key)
		{
		case GLFW_KEY_ESCAPE:
		{
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		}

		case GLFW_KEY_SPACE:
		{
			auto simulator = instance->getSimulator();
			simulator->Pause();
			break;
		}
		case GLFW_KEY_T:
		{
			auto renderer = instance->getRenderer();
			renderer->SwitchTemperatureShader();
			break;
		}
		case GLFW_KEY_1:
		{
			auto renderer = instance->getRenderer();
			renderer->SwitchSphVisibility();
			break;
		}
		case GLFW_KEY_2:
		{
			auto renderer = instance->getRenderer();
			renderer->SwitchDEMVisibility();
			break;
		}
		case GLFW_KEY_3:
		{
			auto renderer = instance->getRenderer();
			renderer->SwitchBoundaryVisibility();
			break;
		}

		case GLFW_KEY_P:
		{			
			auto renderer = instance->getRenderer();
			renderer->SwitchDepthSmooth();
			break;
		}
		}
	}
	else if (action == GLFW_REPEAT)
	{
		auto camera = instance->getMainCamera();
		switch (key)
		{
		case GLFW_KEY_UP:
		{
			camera->Rotate(0.0f, 0.03f);
			break;
		}
		case GLFW_KEY_DOWN:
		{
			camera->Rotate(0.0f, -0.03f);
			break;
		}
		case GLFW_KEY_LEFT:
		{
			camera->Rotate(-0.03f, 0.0f);
			break;
		}
		case GLFW_KEY_RIGHT:
		{
			camera->Rotate(0.03f, 0.0f);
			break;
		}

		}
	}

}

void Mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	static GLFWApp* const instance = GLFWApp::getInstance();
	
	double x_pos, y_pos;

	glfwGetCursorPos(window, &x_pos, &y_pos);

	/* record last mouse position when press mouse right button */
	if (button == GLFW_MOUSE_BUTTON_RIGHT)
	{
		if (action == GLFW_PRESS)
		{
			instance->m_mouse_last_x = static_cast<float>(x_pos);
			instance->m_mouse_last_y = static_cast<float>(y_pos);
			instance->m_mouse_pressed = true;

		}
		else if (action == GLFW_RELEASE)
		{
			instance->m_mouse_pressed = false;
		}
	}

}

void Mouse_cursor_callback(GLFWwindow* window, double xpos, double ypos)
{
	auto instance = GLFWApp::getInstance();
	if (instance->m_mouse_pressed)
	{
		auto camera = instance->getMainCamera();

		float x_change = -0.0005f * static_cast<float>(xpos - instance->m_mouse_last_x);
		float y_change = 0.0005f * static_cast<float>(ypos - instance->m_mouse_last_y);

		camera->Rotate(x_change, y_change);

		instance->m_mouse_last_x = xpos;
		instance->m_mouse_last_y = ypos;
	}

	
}

void Mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	auto camera = GLFWApp::getInstance()->getMainCamera();

	//camera->Zoom(-0.1f * yoffset);
	camera->MoveForward(0.1f * yoffset);
}

void GLFWApp::Render()
{
	m_renderer->Render();
	//GUI rendering
	m_gui_manager->Render();
}

void GLFWApp::Update()
{
	auto anim_characters = m_resource_manager->getAnimCharacters();

	m_previousTime = m_currentTime;
	m_currentTime = static_cast<float>(glfwGetTime());
	
	float dt = m_currentTime - m_previousTime;

	const float time_step = 0.001f;
#ifdef _USE_CUDA_
	m_simulator->StepCUDA(time_step);
#else
	m_simulator->Step(time_step);
#endif
	for (auto it = m_resource_manager->getObjects().begin(); 
		it != m_resource_manager->getObjects().end(); ++it)
	{
		(*it)->Update();
	}

	m_mainCamera->Update();

	// GUI update
	m_gui_manager->Update();

	if(!m_simulator->isPause())
		m_frame_count++;
}

void GLFWApp::SetUpImGui()
{
	//Set up imgui binding
	ImGui::CreateContext();
	ImGuiIO &io = ImGui::GetIO(); (void)io;
	ImGui_ImplGlfwGL3_Init(m_window, false);
	ImGui::StyleColorsDark();
}
/*
 * GUI functions
 */

void Frame_Status_GUI()
{
	auto resource_manager = GLFWApp::getInstance()->getResourceManager();
	auto camera = resource_manager->getMainCamera();
	// If we don't call ImGui::Begin()/ImGui::End() the widgets automatically appears in a window called "Debug".
	ImGui::Begin("Frame Status");
	static float f = 0.0f;
	static int counter = 0;
	ImGui::Text("Frame: %u", GLFWApp::getInstance()->m_frame_count);
	ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	ImGui::NewLine();
	//ImGui::Text("Object Array Size: %u", resource_manager->getObjects().size());
	//ImGui::Text("Mesh Array Size: %u", resource_manager->getMeshes().size());
	//ImGui::Text("Shader Array Size: %u", resource_manager->getShaders().size());
	//ImGui::Text("IMGUI time: %.2lf ms", (t0-t4).count()/1000000.0);
	//ImGui::Text("Update time: %.2lf ms", (t2-t1).count()/1000000.0);
	//ImGui::Text("Render time: %.2lf ms", (t3-t2).count()/1000000.0);
	//ImGui::Text("SWAP & PollEvent time: %.2lf ms", (t4-t3).count()/1000000.0);

	//ImGui::Text("Camera Position: %.2f, %.2f, %.2f", camera->m_position.x, camera->m_position.y, camera->m_position.z);

	ImGui::End();
}

void Object_Viewer_GUI()
{
	ImGui::Begin("Object List");
	auto resource_manager = GLFWApp::getInstance()->getResourceManager();
	
	static int update_count = 0;
	static std::string name_list = "";
	if (update_count % 60 == 0)
	{
		update_count = 0;
		name_list.clear();
		for (auto it = resource_manager->getObjects().begin();
			it != resource_manager->getObjects().end(); ++it)
		{
			name_list += (*it)->getName() + "\n";
		}
	}
	ImGui::Text("%s", name_list.data());
	ImGui::NewLine();
	
	update_count++;
	ImGui::End();
}

void Mouse_Position_GUI()
{
	auto instance = GLFWApp::getInstance();
	
	double xpos, ypos;
	glfwGetCursorPos(instance->getGLFWwindow(), &xpos, &ypos);

	{
		ImGui::Begin("Mouse");
		ImGui::Text("Last:    %f, %f", instance->m_mouse_last_x, instance->m_mouse_last_y);
		ImGui::Text("Current: %f, %f", static_cast<float>(xpos), static_cast<float>(ypos));
		ImGui::End();
	}
}

