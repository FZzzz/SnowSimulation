#version 410

layout (location = 0) in vec3 vertex_pos;
layout (location = 1) in vec2 tex_coord;

out vec2 coord;

void main()
{
    coord = tex_coord;
    gl_Position = vec4(vertex_pos, 1.0f);
}