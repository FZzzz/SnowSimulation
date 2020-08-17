#include "GLFunctions.h"
/* Descrepted: Moved to Shader
void GLFunctions::SetUniformMat4(const std::string & name, const glm::mat4 & mat, const GLuint program)
{
	//glUseProgram(program);
	GLint loc = glGetUniformLocation(program, name.c_str());
	if (loc == -1)
		return;

	glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(mat));
}

void GLFunctions::SetUniformVec3(const std::string & name, const glm::vec3 & vec, const GLuint program)
{
	//glUseProgram(program);
	GLint loc = glGetUniformLocation(program, name.c_str());
	if (loc == -1)
		return;

	glUniform3fv(loc, 1, glm::value_ptr(vec));
}

void GLFunctions::SetUniformInt(const std::string & name, const GLint value, const GLuint program)
{
	GLint loc = glGetUniformLocation(program, name.c_str());
	if (loc == -1)
		return;

	glUniform1i(loc, value);
}
*/

GLuint GLFunctions::LoadTexture(GLchar * path, GLboolean alpha)
{
	//Generate texture ID and load texture data 
	GLuint textureID;
	glGenTextures(1, &textureID);
	int width, height;
	unsigned char* image = SOIL_load_image(path, &width, &height, 0, alpha ? SOIL_LOAD_RGBA : SOIL_LOAD_RGB);
	// Assign texture to ID
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexImage2D(GL_TEXTURE_2D, 0, alpha ? GL_RGBA : GL_RGB, width, height, 0, alpha ? GL_RGBA : GL_RGB, GL_UNSIGNED_BYTE, image);
	glGenerateMipmap(GL_TEXTURE_2D);

	// Parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, alpha ? GL_CLAMP_TO_EDGE : GL_REPEAT);	// Use GL_CLAMP_TO_EDGE to prevent semi-transparent borders. Due to interpolation it takes value from next repeat 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, alpha ? GL_CLAMP_TO_EDGE : GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);
	SOIL_free_image_data(image);
	return textureID;

}
