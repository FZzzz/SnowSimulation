#version 410

layout (location = 0) in vec3 aPos;

uniform mat4 pvm;
//uniform float point_size;

void main()
{
    gl_Position = pvm * vec4(aPos, 1.0f);
    //gl_PointSize = point_size / gl_Position.w;
}