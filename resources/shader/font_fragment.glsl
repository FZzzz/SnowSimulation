#version 330

// Default color buffer location is 0
// If you create framebuffer your own, you need to take care of it
in vec3 vertexColor;
in vec2 vertexTex;

out vec4 color;

uniform sampler2D tex1;

void main()
{
	//color = vec4(vertexColor, 1.0);
	//color = vec4( 1.0f , 1.0f , 1.0f , 1.0f );
	vec4 texColor = texture( tex1 , vertexTex);

	if(texColor.a < 0.1f)
		discard;
	color = texColor;
}
