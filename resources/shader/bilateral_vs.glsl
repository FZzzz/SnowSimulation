#version 410

in vec2 vertex_pos;

out vec2 coord;

void main()
{
    coord = 0.5f * vertex_pos + 0.5f;
    gl_position = vec4(vertex_pos, 0.0f, 1.0f);
}