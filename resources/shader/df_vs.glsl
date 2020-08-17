#version 330 core
layout (location = 0) in vec3 posIn;
layout (location = 1) in vec3 normalIn;

out vec3 fpos;

uniform mat4 pvm;
uniform mat4 modelMat;

void main()
{
    fpos = vec3(modelMat * vec4(posIn, 1.0));

    gl_Position = pvm * vec4(posIn, 1.0);
}
