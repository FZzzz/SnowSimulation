#version 410

//in vec3 light_dir;
//in vec3 vertex_point_color;
in vec3 eye_space_pos;

uniform float sphere_radius;
uniform mat4 projection;
uniform float far_plane;
uniform float near_plane;

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
    vec3 n;
    n.xy = gl_PointCoord.xy * 2.f - 1.f;
    float r2 = dot(n.xy, n.xy);
    //n.z = 1.f - dot(n.xy, n.xy);
 
    if(r2 > 1.f) 
        discard;
    
    n.z = sqrt(1.0f - r2);
    
    //transform with modelview matrix
    //vec3 m = normalize(gl_NormalMatrix * n);
    //float diffuse = dot(light_dir, n);

    vec4 pixel_pos =  vec4(eye_space_pos + n * sphere_radius, 1.0f);
    vec4 clip_space_pos = projection * pixel_pos;
   
   float depth = (clip_space_pos.z / clip_space_pos.w);

    // assign depth buffer value
    gl_FragDepth = depth;// * 0.5f + 0.5f;
    //gl_FragDepth = 1.f;

    //frag_color.r = 1.0f; //-pixel_pos.z;
    // assign color buffer value
    frag_color = vec4(vec3(LinearizeDepth(depth) / far_plane), 1.0);

    //frag_color = vec4(1.0f);

    //frag_color.xyz = diffuse * vertex_point_color;
    //frag_color.w = 1.f;

    // calculate depth
    
    //gl_FragDepth = clip_pasce_pos.z / clip_pasce_pos.w;

}