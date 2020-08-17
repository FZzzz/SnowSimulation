#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec2 TexCoords;

out VS_OUT {
    vec3 FragPos;
    vec2 TexCoords;
    vec4 FragPosLightSpace;
} vs_out;

uniform mat4 pvm;
//uniform mat4 view;
uniform mat4 modelMat;
uniform mat4 lightSpaceMatrix;

void main()
{
    vs_out.FragPos = vec3(modelMat * vec4(aPos, 1.0));
    //vs_out.Normal = transpose(inverse(mat3(model))) * aNormal;
    //vs_out.TexCoords = aTexCoords;
    vs_out.FragPosLightSpace = lightSpaceMatrix * vec4(vs_out.FragPos, 1.0);
    gl_Position = pvm * vec4(aPos, 1.0);
}