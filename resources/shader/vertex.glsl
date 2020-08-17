#version 410
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;

//layout(location=2) in vec3 color;

uniform mat4 pvm;
uniform mat4 modelMat;

flat out vec3 vertexColor;

void main()
{
	/*

	vec3 obj_color = vec3(1,1,1);
	float ambient = 0.1f;
	vec3 light_color = vec3(1.0);
	vec3 light_pos = vec3(10,10,10);
	vec3 light_dir = normalize(light_pos - position);
	float diff = max(dot(normal , light_dir) , 0.0);
	vec3 diffuse = diff * light_color;

	*/
	gl_Position = pvm * vec4(position, 1.0f);

	//vertexColor = (ambient + diffuse) * obj_color;//vec3(1.0f,1.0f,1.0f);
	vertexColor = vec3(1.0,1.0,1.0);

}
