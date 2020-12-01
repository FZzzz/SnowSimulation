#version 410

layout (location = 0) in vec3 aPos;

uniform mat4 model_view;
uniform mat4 pvm;
uniform float point_size;

out vec3 eye_space_pos;

void main()
{
    vec4 view_pos = model_view * vec4(aPos, 1.0);
    gl_Position = pvm * vec4(aPos, 1.f);

    eye_space_pos = view_pos.xyz;
    gl_PointSize = point_size / gl_Position.w;
}