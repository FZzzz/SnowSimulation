#version 330 core
out vec4 fragColor;

in vec3 fpos;
in vec3 fnormal;

void main()
{

    vec3 light_pos = vec3(5.0f , 5.0f , 5.0f);
	vec3 view_pos = vec3(5.0f,5.0f,5.0f);

	vec3 light_color = vec3(1.0f , 1.0f , 1.0f);
	vec3 object_color = vec3(1.0f , 1.0f , 1.0f);

    // ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * light_color;
  	
    // diffuse
    vec3 litDir = normalize(light_pos- fpos);
    float diff = max(dot(fnormal, litDir), 0.0);
    vec3 diffuse = diff * light_color;
    
	//spceular
	float specStrength = 0.5;
	
	vec3 view_dir = normalize(view_pos - fpos);
	vec3 reflect_dir = reflect(-litDir, fnormal);
	float spec = pow(max( dot(view_dir, reflect_dir), 0.0) , 32);
	vec3 specular = specStrength * spec * light_color;


	// (x, y, z) * (a, b, c) = (ax, by, cz)
    vec3 result = (ambient + diffuse + specular) * object_color;
    fragColor = vec4(result, 1.0);
} 