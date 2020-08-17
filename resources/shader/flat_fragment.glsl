#version 330

// Default color buffer location is 0
// If you create framebuffer your own, you need to take care of it
in vec3 fColor;

out vec4 color;

void main()
{
	color = vec4(fColor, 1.0);
	//color = vec4( 1.0f , 1.0f , 1.0f , 1.0f );
	
}