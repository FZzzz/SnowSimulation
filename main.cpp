#include "GLFWApp.h"
#include <iostream>
#define _CRTDBG_MAP_ALLOC 
#include <crtdbg.h>  

// Replace _NORMAL_BLOCK with _CLIENT_BLOCK if you want the
// allocations to be of _CLIENT_BLOCK type

int main()
{
	std::string file_path;
	//Create GLFWApp
	std::unique_ptr<GLFWApp> demoApp(GLFWApp::getInstance());

	demoApp->Initialize(1600, 900, "SnowSimulation");

	/*
		Add text to App
	*/
	if (demoApp->GetAppStatus())
	{
		demoApp->Run();
	}
	

	//release resources
	demoApp->ReleaseResources();

	demoApp.reset();
#ifdef _DEBUG
	//system("pause");
	//_CrtDumpMemoryLeaks();
#endif
	return 0;
}
