#ifndef _SHADER_H_
#define _SHADER_H_

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <GL/glew.h>
#include <memory>
#include <map>
#include <glm/common.hpp>

using namespace std;

enum ShaderEnum
{
	SHADER_MODE_FLAT,
	SHADER_MODE_PHONG,
	SHADER_MODE_TEXTURED_FLAT,
	SHADER_MODE_TEXTURED_PHONG
};

class Shader : public std::enable_shared_from_this<Shader>
{
public:
	Shader(std::string name);
	~Shader();

	bool SetupShader();
	bool SetupShader(const GLchar* vsPath, const GLchar* fsPath , const GLchar* gsPath);
	bool SetupShader(const GLchar* vsPath, const GLchar* fsPath);
	bool SetupShader(ShaderEnum);
	void SetUniformMat4(const std::string& name, const glm::mat4& mat);
	void SetUniformVec2(const std::string& name, const glm::vec2& v);
	void SetUniformVec3(const std::string& name, const glm::vec3& v);
	void SetUniformVec4(const std::string& name, const glm::vec4& v);

	void SetUniformInt(const std::string& name, const GLint value);
	void SetUniformFloat(const std::string& name, const float value);
	void Use();
	
	//getter 
	inline std::string		getName() { return m_name; };
	inline const GLuint&	getProgram() const { return m_program; };

private:
	std::string m_name;
	GLuint m_program;
	GLuint m_vertex_shader, m_geometry_shader, m_fragment_shader;

	void RegistToManager();

};

#endif
