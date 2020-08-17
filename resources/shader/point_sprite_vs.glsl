#version 410

layout (location = 0) in vec3 aPos;

uniform mat4 pvm;
uniform float point_size;
uniform vec3 light_pos;
uniform vec3 camera_pos;
uniform mat4 view;
uniform vec3 point_color;

out vec3 light_dir;
out vec3 vertex_point_color;

void main()
{
    light_dir = normalize(vec3(light_pos - vec3(view * vec4(aPos, 1.f))));
    vertex_point_color = point_color;

    vec3 v = camera_pos - aPos;
    float dist = sqrt(dot(v,v));

    gl_Position = pvm * vec4(aPos, 1.0f);
    gl_PointSize = point_size / dist;
}