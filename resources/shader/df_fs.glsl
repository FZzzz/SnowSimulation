#version 330 core
out vec4 fragColor;

in vec3 fpos;

void main()
{

    //vec3 light_pos = vec3(5.0f , 5.0f , 5.0f);
	vec3 light_color = vec3(1.0f , 1.0f , 1.0f);
	vec3 object_color = vec3(1.0f , 1.0f , 1.0f);

    // ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * light_color;
  	
    // diffuse
	vec3 fdx = dFdx(fpos);
	vec3 fdy = dFdy(fpos);
	vec3 normal = normalize(cross(fdx , fdy));
    vec3 litDir = normalize( vec3(1,2,3));
    float diff = max(dot(normal, litDir), 0.0);
    vec3 diffuse = diff * light_color;
    
	// (x, y, z) * (a, b, c) = (ax, by, cz)
    vec3 result = (ambient + diffuse) * object_color;
    fragColor = vec4(result, 1.0);
} 