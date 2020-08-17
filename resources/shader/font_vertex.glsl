#version 330
layout(location=0) in vec3 position;
layout(location=1) in vec3 color;
layout(location=2) in vec2 texcoord;

uniform mat4 pvm;
uniform mat4 modelMat;

out vec3 vertexColor;
out vec2 vertexTex;

void main()
{

	gl_Position = pvm * vec4(position, 1.0f);

	vertexColor = color;
	vertexTex = vec2(texcoord.x , 1-texcoord.y);
}
