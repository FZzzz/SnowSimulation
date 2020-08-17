#version 330
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec3 color;

uniform mat4 pvm;
uniform mat4 modelMat;
uniform bool hasNormal;

out vec3 pos;
//out vec3 vertexColor;

void main()
{
	
	pos = mat3(modelMat) * position;
	gl_Position = pvm * vec4(position, 1.0f);
		
	//pos = vec3(modelMat * vec4(position , 1.0f));

	//vertexColor = color;
	
}