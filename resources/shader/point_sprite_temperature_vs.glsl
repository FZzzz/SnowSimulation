#version 410

layout (location = 0) in vec3 aPos;
layout (location = 1) in float aTemperature;

uniform mat4 pvm;
uniform float point_size;
uniform vec3 light_pos;
uniform vec3 camera_pos;
uniform mat4 view;
uniform vec3 point_color;

uniform vec3 hottest_color;
uniform vec3 coolest_color;

uniform float hottest_temperature;
uniform float coolest_temperature;

out vec3 light_dir;
out vec3 vertex_point_color;

void main()
{
    light_dir = normalize(vec3(light_pos - vec3(view * vec4(aPos, 1.f))));

    float temp_region = hottest_temperature - coolest_temperature;
    float alpha = (aTemperature - coolest_temperature) / temp_region;
    vertex_point_color = mix(coolest_color, hottest_color, alpha);

    vec3 v = camera_pos - aPos;
    float dist = sqrt(dot(v,v));

    gl_Position = pvm * vec4(aPos, 1.0f);
    gl_PointSize = point_size / dist;
}