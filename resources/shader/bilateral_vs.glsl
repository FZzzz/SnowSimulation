#version 410

in vec3 vertex_pos;
in vec2 tex_coord;

out vec2 coord;

void main()
{
    coord = tex_coord;
    gl_position = vec4(vertex_pos, 1.0f);
}