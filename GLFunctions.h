#ifndef _GL_FUNCTIONS_H_
#define _GL_FUNCTIONS_H_

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <SOIL/SOIL.h>
#include <string>

namespace GLFunctions
{
	//void SetUniformMat4(const std::string& name, const glm::mat4& mat, const GLuint program);
	//void SetUniformVec3(const std::string& name, const glm::vec3& mat, const GLuint program);
	//void SetUniformInt(const std::string& name, const GLint value, const GLuint program);
	
	GLuint LoadTexture(const GLchar* path, GLboolean alpha);
	
};

#endif // ! _GL_FUNCTIONS_H_
