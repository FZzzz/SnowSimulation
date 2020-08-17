#version 330
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
//simple calculate normals and output fColor

//uniform vec3 light_pos;
//uniform vec3 light_color;

in vec3 pos[3];
//in vec3 vertexColor[3];

out vec3 fColor;


void main()
{

	//vec3 light_pos = vec3(5.0f , 5.0f , 5.0f);
	
	vec3 light_color = vec3(1.0f , 1.0f , 1.0f);

	vec3 object_color = vec3(1.0f , 1.0f , 1.0f);
	
	vec3 v1 = pos[1] - pos[0];
	vec3 v2 = pos[2] - pos[0];

	vec3 center = ( pos[0] + pos[1] + pos[2] )/ 3.0f;
	vec3 normal = normalize(cross(v1,v2));
	
	float ambientStrength = 0.05f;
	vec3 ambient = ambientStrength * light_color;
	
	vec3 light_dir = normalize( vec3(-1,-1,-1) );
	float diff = max(dot(normal , light_dir) , 0.0);
	vec3 diffuse = diff * light_color;

	fColor = (ambient + diffuse) * object_color;
	//fColor = vec3(1.0f,1.0f,1.0f);
	for(int i=0;i<3;i++)
	{
		gl_Position = gl_in[i].gl_Position;
		//fColor = vec3(1.0f,1.0f,1.0f);
		EmitVertex();
	}
	
	EndPrimitive();
}