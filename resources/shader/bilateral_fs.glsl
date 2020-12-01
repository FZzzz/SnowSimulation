#version 410

in vec2 coord;

uniform sampler2D depth_map;
uniform vec2 blur_dir;
uniform float filter_radius;
uniform float blur_scale;
uniform float far_plane;
uniform float near_plane;

const float blur_depth_falloff = 65.0f;

out vec4 frag_color;

float LinearizeDepth(float depth)
{
    //const float near_plane = 0.01f;
    //const float far_plane = 20.0f;
    float z = depth * 2.0 - 1.0; // Back to NDC 
    return (2.0 * near_plane * far_plane) / (far_plane + near_plane - z * (far_plane - near_plane));	
}


void main() 
{
    // read depth from depth map texture
    
    float depth = texture2D(depth_map, coord).x;

    if(depth <= 0.0f)
    {
        gl_FragDepth = 0;
        return;
    }
    if(depth >= 1.0f)
    {
        gl_FragDepth = depth;
        return;
    }

    float sum = 0.f;
    float wsum = 0.0f;

    for(float x = -filter_radius; x<= filter_radius; x+= 1.0f)
    {
        float s = texture(depth_map, coord + x * blur_dir).x;
        if(s >= 1.f) continue;

        float r = x * blur_scale;
        float w = exp(-r*r);

        float r2 = (s - depth) * blur_depth_falloff;
        float g = exp(-r2 * r2);

        sum += s * w * g;
        wsum += w * g;
    }

    if(wsum > 0.0f)
        sum /= wsum;
    
    gl_FragDepth = sum;


    // visualize this pass
    frag_color = vec4(vec3(LinearizeDepth(sum) / far_plane), 1.0f);

    //frag_color = texture2D(depth_map, coord);
    //frag_color = vec4(coord, 0.f, 1.f);
    //frag_color = vec4(vec3(LinearizeDepth(sum)), 1.0f);
    //
    //frag_color = vec4(vec3(1,1,1), 1.0f);
}