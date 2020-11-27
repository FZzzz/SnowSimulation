#version 410

in vec2 coord;

uniform sampler2D depth_map;
uniform vec2 blur_dir;
uniform float filter_radius;
uniform float blur_scale;

const float blur_depth_falloff = 65.0f;

out vec4 frag_color;

void main() 
{
    // read depth from depth map texture
    
    float depth = texture2D(depth_map, coord.xy).r;

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
        
    //gl_FragDepth = sum;
    */
    frag_color = vec4(vec3(1,1,1), 1.0f);
    gl_FragDepth = sum;

    //frag_color = vec4(vec3(depth), 1.0f);
    //frag_color = texture2D(depth_map, coord);
    frag_color = vec4(vec3(depth), 1.0f);

    //frag_color = vec4(vec3(LinearizeDepth(sum)), 1.0f);
    //frag_color = vec4(vec3(LinearizeDepth(depth) / 2.f), 1.0f);
    //frag_color = vec4(vec3(1,1,1), 1.0f);
}