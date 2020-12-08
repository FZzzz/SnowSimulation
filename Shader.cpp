#include "Shader.h"
#include "GLFWApp.h"
#include <iostream>

Shader::Shader(std::string name) 
	: m_vertex_shader(0), m_geometry_shader(0), m_fragment_shader(0), m_program(0), m_name(name)
{
}

Shader::~Shader()
{
#ifdef _DEBUG
	std::cout << "Shader dctor called" << std::endl;
#endif
	if (m_vertex_shader)
	{
		glDetachShader(m_program, m_vertex_shader);
		glDeleteShader(m_vertex_shader);
	}

	if (m_geometry_shader)
	{
		glDetachShader(m_program, m_geometry_shader);
		glDeleteShader(m_geometry_shader);
	}

	if (m_fragment_shader)
	{
		glDetachShader(m_program, m_fragment_shader);
		glDeleteShader(m_fragment_shader);
	}
	glDeleteProgram(m_program);
}

bool Shader::SetupShader()
{
	const GLchar* vs_path = "resources/shader/flat_vertex.glsl";
	const GLchar* fs_path = "resources/shader/flat_fragment.glsl";
	const GLchar* gs_path = "resources/shader/flat_geometry.glsl";
	string vs_code;
	string gs_code;
	string fs_code;

	try
	{
		ifstream vs_fptr(vs_path);
		ifstream fs_fptr(fs_path);
		ifstream gs_fptr(gs_path);

		stringstream vs_stream, gs_stream, fs_stream;

		vs_stream << vs_fptr.rdbuf();
		gs_stream << gs_fptr.rdbuf();
		fs_stream << fs_fptr.rdbuf();


		vs_fptr.close();
		gs_fptr.close();
		fs_fptr.close();

		vs_code = vs_stream.str();
		gs_code = gs_stream.str();
		fs_code = fs_stream.str();
	}
	catch (exception e)
	{
		cout << "ERROR: SHADER FILE FAILED!!" << endl;
		return false;
	}
	const GLchar* vs_code_input = vs_code.c_str();
	const GLchar* gs_code_input = gs_code.c_str();
	const GLchar* fs_code_input = fs_code.c_str();

	GLuint vertex, geometry, fragment;
	GLint success;
	GLchar infolog[512];

	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vs_code_input, nullptr);
	glCompileShader(vertex);

	glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);

	if (!success)
	{
		glGetShaderInfoLog(vertex, 512, nullptr, infolog);
		cout << "ERROR: VERTEX SHADER COMPILE FAILED\n" << infolog << endl;
		return false;
	}

	geometry = glCreateShader(GL_GEOMETRY_SHADER);
	glShaderSource(geometry, 1, &gs_code_input, nullptr);
	glCompileShader(geometry);

	glGetShaderiv(geometry, GL_COMPILE_STATUS, &success);

	if (!success)
	{
		glGetShaderInfoLog(geometry, 512, nullptr, infolog);
		cout << "ERROR: GEOMETRY SHADER COMPILE FAILED\n" << infolog << endl;
		return false;
	}

	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fs_code_input, nullptr);
	glCompileShader(fragment);

	glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragment, 512, nullptr, infolog);
		cout << "ERROR::SHADER::FRAGMENT::COMPILATION::FAILED\n" << infolog << endl;
		return false;
	}

	this->m_program = glCreateProgram();
	glAttachShader(this->m_program, vertex);
	glAttachShader(this->m_program, geometry);
	glAttachShader(this->m_program, fragment);
	glLinkProgram(this->m_program);

	glGetProgramiv(this->m_program, GL_LINK_STATUS, &success);
	if (!success)
	{
		glGetProgramInfoLog(this->m_program, 512, nullptr, infolog);
		cout << "ERROR::PROGRAM::LINK\n" << infolog << endl;
		return false;
	}

	cout << "Shader Creation Success!!" << endl;

	m_vertex_shader = vertex;
	m_geometry_shader = geometry;
	m_fragment_shader = fragment;

	return true;

}

/*
bool Shader::SetupShader()
{
	const GLchar* vs_path = "resources/shader/vertex.glsl";
	const GLchar* fs_path = "resources/shader/fragment.glsl";
	string vs_code;
	string fs_code;

	try
	{
		ifstream vs_fptr(vs_path);
		ifstream fs_fptr(fs_path);

		stringstream vs_stream, gs_stream, fs_stream;

		vs_stream << vs_fptr.rdbuf();
		fs_stream << fs_fptr.rdbuf();


		vs_fptr.close();
		fs_fptr.close();

		vs_code = vs_stream.str();
		fs_code = fs_stream.str();
	}
	catch (exception e)
	{
		cout << "ERROR: SHADER FILE FAILED!!" << endl;
		return false;
	}
	const GLchar* vs_code_input = vs_code.c_str();
	const GLchar* fs_code_input = fs_code.c_str();

	GLuint vertex, fragment;
	GLint success;
	GLchar infolog[512];

	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vs_code_input, nullptr);
	glCompileShader(vertex);

	glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);

	if (!success)
	{
		glGetShaderInfoLog(vertex, 512, nullptr, infolog);
		cout << "ERROR: VERTEX SHADER COMPILE FAILED\n" << infolog << endl;
		return false;
	}

	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fs_code_input, nullptr);
	glCompileShader(fragment);

	glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragment, 512, nullptr, infolog);
		cout << "ERROR::SHADER::FRAGMENT::COMPILATION::FAILED\n" << infolog << endl;
		return false;
	}

	this->m_program = glCreateProgram();
	glAttachShader(this->m_program, vertex);
	glAttachShader(this->m_program, fragment);
	glLinkProgram(this->m_program);

	glGetProgramiv(this->m_program, GL_LINK_STATUS, &success);
	if (!success)
	{
		glGetProgramInfoLog(this->m_program, 512, nullptr, infolog);
		cout << "ERROR::PROGRAM::LINK\n" << infolog << endl;
		return false;
	}

	cout << "Shader Creation Success!!" << endl;

	return true;
}
*/

bool Shader::SetupShader(const GLchar * vsPath, const GLchar * fsPath, const GLchar * gsPath)
{
	return false;
}

bool Shader::SetupShader(const GLchar* vsPath, const GLchar* fsPath)
{
	cout << "Creating shader: " << m_name << endl;
	const GLchar* vs_path = vsPath;
	const GLchar* fs_path = fsPath;
	string vs_code;
	string fs_code;

	try
	{
		ifstream vs_fptr(vs_path);
		ifstream fs_fptr(fs_path);

		stringstream vs_stream, gs_stream, fs_stream;

		vs_stream << vs_fptr.rdbuf();
		fs_stream << fs_fptr.rdbuf();


		vs_fptr.close();
		fs_fptr.close();

		vs_code = vs_stream.str();
		fs_code = fs_stream.str();
	}
	catch (exception e)
	{
		cout << "ERROR: SHADER FILE FAILED!!" << endl;
		return false;
	}
	const GLchar* vs_code_input = vs_code.c_str();
	const GLchar* fs_code_input = fs_code.c_str();

	GLuint vertex, fragment;
	GLint success;
	GLchar infolog[512];

	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vs_code_input, nullptr);
	glCompileShader(vertex);

	glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);

	if (!success)
	{
		glGetShaderInfoLog(vertex, 512, nullptr, infolog);
		cout << "ERROR: VERTEX SHADER COMPILE FAILED\n" << infolog << endl;
		return false;
	}

	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fs_code_input, nullptr);
	glCompileShader(fragment);

	glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragment, 512, nullptr, infolog);
		cout << "ERROR::SHADER::FRAGMENT::COMPILATION::FAILED\n" << infolog << endl;
		return false;
	}

	this->m_program = glCreateProgram();
	glAttachShader(this->m_program, vertex);
	glAttachShader(this->m_program, fragment);
	glLinkProgram(this->m_program);

	glGetProgramiv(this->m_program, GL_LINK_STATUS, &success);
	if (!success)
	{
		glGetProgramInfoLog(this->m_program, 512, nullptr, infolog);
		cout << "ERROR::PROGRAM::LINK\n" << infolog << endl;
		return false;
	}

	cout << "Shader Creation Success!!" << endl;

	m_vertex_shader = vertex;
	m_fragment_shader = fragment;

	RegistToManager();
	
	return true;
}

bool Shader::SetupShader(ShaderEnum mode)
{
	const GLchar* vs_path="";
	const GLchar* fs_path="";

	switch (mode)
	{
	case SHADER_MODE_FLAT:
		vs_path = "resources/shader/df_vs.glsl";
		fs_path = "resources/shader/df_fs.glsl";
		break;
	case SHADER_MODE_PHONG:
		vs_path = "resources/shader/phong_vs.glsl";
		fs_path = "resources/shader/phong_fs.glsl";
		break;
	case SHADER_MODE_TEXTURED_FLAT:
		vs_path = "resources/shader/flat_texture_vs.glsl";
		fs_path = "resources/shader/flag_texture_fs.glsl";
		break;
	case SHADER_MODE_TEXTURED_PHONG:
		vs_path = "resources/shader/phong_texture_vs.glsl";
		fs_path = "resources/shader/phong_texture_fs.glsl";
		break;
	default:
		fprintf(stderr, "Invalid operation(Shader Mode)");
		return false;
	}

	string vs_code;
	string fs_code;

	try
	{
		ifstream vs_fptr(vs_path);
		ifstream fs_fptr(fs_path);

		stringstream vs_stream, gs_stream, fs_stream;

		vs_stream << vs_fptr.rdbuf();
		fs_stream << fs_fptr.rdbuf();


		vs_fptr.close();
		fs_fptr.close();

		vs_code = vs_stream.str();
		fs_code = fs_stream.str();
	}
	catch (exception e)
	{
		cout << "ERROR: SHADER FILE FAILED!!" << endl;
		return false;
	}
	const GLchar* vs_code_input = vs_code.c_str();
	const GLchar* fs_code_input = fs_code.c_str();

	GLuint vertex, fragment;
	GLint success;
	GLchar infolog[512];

	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vs_code_input, nullptr);
	glCompileShader(vertex);

	glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);

	if (!success)
	{
		glGetShaderInfoLog(vertex, 512, nullptr, infolog);
		cout << "ERROR: VERTEX SHADER COMPILE FAILED\n" << infolog << endl;
		return false;
	}

	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fs_code_input, nullptr);
	glCompileShader(fragment);

	glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragment, 512, nullptr, infolog);
		cout << "ERROR::SHADER::FRAGMENT::COMPILATION::FAILED\n" << infolog << endl;
		return false;
	}

	this->m_program = glCreateProgram();
	glAttachShader(this->m_program, vertex);
	glAttachShader(this->m_program, fragment);
	glLinkProgram(this->m_program);

	glGetProgramiv(this->m_program, GL_LINK_STATUS, &success);
	if (!success)
	{
		glGetProgramInfoLog(this->m_program, 512, nullptr, infolog);
		cout << "ERROR::PROGRAM::LINK\n" << infolog << endl;
		return false;
	}

	cout << "Shader Creation Success!!" << endl;

	m_vertex_shader = vertex;
	m_fragment_shader = fragment;

	RegistToManager();

	return true;
}

void Shader::SetUniformMat4(const std::string & name, const glm::mat4 & mat)
{
	GLint loc = glGetUniformLocation(m_program, name.c_str());
	if (loc == -1)
		return;

	glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(mat));
}

void Shader::SetUniformVec2(const std::string& name, const glm::vec2& v)
{
	GLint loc = glGetUniformLocation(m_program, name.c_str());
	if (loc == -1)
		return;

	glUniform2fv(loc, 1, glm::value_ptr(v));
}

void Shader::SetUniformVec3(const std::string & name, const glm::vec3 & vec)
{
	GLint loc = glGetUniformLocation(m_program, name.c_str());
	if (loc == -1)
		return;

	glUniform3fv(loc, 1, glm::value_ptr(vec));
}

void Shader::SetUniformVec4(const std::string& name, const glm::vec4& v)
{
	GLint loc = glGetUniformLocation(m_program, name.c_str());
	if (loc == -1)
		return;

	glUniform4fv(loc, 1, glm::value_ptr(v));
}

void Shader::SetUniformInt(const std::string & name, const GLint value)
{
	GLint loc = glGetUniformLocation(m_program, name.c_str());
	if (loc == -1)
		return;

	glUniform1i(loc, value);
}

void Shader::SetUniformFloat(const std::string& name, const float value)
{
	GLint loc = glGetUniformLocation(m_program, name.c_str());
	if (loc == -1)
		return;

	glUniform1f(loc, value);
}

void Shader::Use()
{
	glUseProgram(this->m_program);
}


void Shader::RegistToManager()
{
	/*
	GLint count = 0;
	glGetProgramiv(m_program, GL_ACTIVE_UNIFORMS, &count);
	std::cout << "Active Uniforms: " << count << "\n";
	glm::mat4 test(1.0f);
	GLsizei length;
	GLint size;
	GLenum type;
	GLchar name[32];
	for (GLint i = 0; i < count; ++i)
	{
		glGetActiveUniform(m_program, (GLuint)i, 32, &length, &size, &type, name);
		std::cout << "\nUniform Number: " << i;
		std::cout << "\nType: " << type;
		std::cout << "\nUniform Name: " << name << std::endl;
	}
	*/
	GLFWApp::getInstance()->getResourceManager()->AddShader( shared_from_this() );
}
