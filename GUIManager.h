#ifndef _GUI_MANAGER_H_
#define _GUI_MANAGER_H_

#include "imgui/imgui.h"
#include <GLFW/glfw3.h>
#include <functional>
#include <vector>

struct GUIFunction
{
	GUIFunction(std::function<void()> gui_func, bool visibility) :
		func(gui_func),
		is_visible(visibility)
	{}
	std::function<void()> func;
	bool is_visible;
};

class GUIManager
{
public:
	GUIManager();
	~GUIManager();

	void Initialize(GLFWwindow* window);
	void Update();
	void Render();

	void AddGUIFunction(std::function<void()> func, bool visiblility=true);

private:
	std::vector<GUIFunction> m_gui_functions;

};

#endif
