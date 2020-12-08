#version 410
layout (location = 0) in vec3 aPos;

uniform mat4 pvm;

void main()
{
    gl_Position = pvm * vec4(aPos, 1.0);
}