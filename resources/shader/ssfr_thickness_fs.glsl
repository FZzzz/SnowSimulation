#version 410

in vec3 eye_space_pos;

out float thickness;

void main()
{
    vec3 n;
    n.xy = gl_PointCoord.xy * 2.f - 1.f;
    float r2 = dot(n.xy, n.xy);

    if(r2 > 1.f) 
        discard;
    
    n.z = sqrt(1.0f - r2);

    // approximate thickness (not need to be precise)
    thickness = n.z * 0.005f;
}