#version 410

in vec2 coord;

uniform sampler2D depth_map;
uniform vec2 blur_dir;
uniform float filter_radius;
uniform float blur_scale;

const float blur_depth_falloff = 65.0f;

void main() 
{
    float depth = texture(depth_map, coord).x;

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

}