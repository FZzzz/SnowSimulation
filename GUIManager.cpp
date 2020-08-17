#include "GUIManager.h"
#include "imgui/imgui_impl_glfw_gl3.h"
#include <iostream>


GUIManager::GUIManager()
{
}


GUIManager::~GUIManager()
{
}

void GUIManager::Initialize(GLFWwindow * window)
{
	if (window == nullptr)
	{
		std::cout << "No window detected" << std::endl;
		return;
	}
	//Set up imgui binding
	ImGui::CreateContext();
	ImGuiIO &io = ImGui::GetIO(); (void)io;
	ImGui_ImplGlfwGL3_Init(window, false);
	ImGui::StyleColorsDark();
}

void GUIManager::Update()
{
	for (auto it = m_gui_functions.begin(); it != m_gui_functions.end(); ++it)
	{
		(*it).func();
	}
}

void GUIManager::Render()
{
	ImGui::Render();
	ImGui_ImplGlfwGL3_RenderDrawData(ImGui::GetDrawData());
}

void GUIManager::AddGUIFunction(std::function<void()> func, bool visiblility)
{
	m_gui_functions.push_back(GUIFunction(func,visiblility));
}
