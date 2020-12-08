#version 410

// This part of code references to https://github.com/JAGJ10/PositionBasedFluids/

in vec2 coord;

uniform sampler2D depth_map;
uniform sampler2D thickness_map;
uniform sampler2D scene_map;
uniform sampler2D scene_depth_map;
uniform vec4 light_color;
//uniform vec3 light_pos;
uniform mat4 projection;
uniform mat4 model_view;
uniform vec2 inv_tex_scale;

out vec4 frag_color;

const vec3 light_dir = vec3(0, 1, 0);
const vec3 light_pos = vec3(0, 1000, 0);
const float shininess = 1000.0;
const float fres_power = 5.0f;
const float fres_scale = 0.9;
const float fres_bias = 0.1;

vec3 uv_to_eye(vec2 p, float z) {
	vec2 pos = p * 2.0f - 1.0f;
	vec4 clip_pos = vec4(pos, z, 1.0f);
	vec4 view_pos = inverse(projection) * clip_pos;
	return view_pos.xyz / view_pos.w;
}

void main()
{
    float depth = texture2D(depth_map, coord).x;
    float scene_depth = texture2D(scene_depth_map, coord).x;
    float thickness = max(texture2D(thickness_map, coord).r, 0.3f);
    vec4  scene = texture2D(scene_map, coord);


    if(depth == 0.0f)
    {
        frag_color = vec4(0);
        return;
    }
    if(depth == 1.f)
    {
        frag_color = scene;
        return;
    }
    if(depth >= scene_depth)
    {
        frag_color = scene;
        return;
    }

    // eye space pos from depth
    vec3 eye_pos = uv_to_eye(coord, depth);

    // finite difference approx for normals, can't take dFdx because
	// the one-sided difference is incorrect at shape boundaries
	vec3 zl = eye_pos 
        - uv_to_eye(coord - vec2(inv_tex_scale.x, 0.0), 
                    texture2D(depth_map, 
                            coord - vec2(inv_tex_scale.x, 0.0)
                             ).x
                    );
	vec3 zr = uv_to_eye(coord + vec2(inv_tex_scale.x, 0.0), 
                        texture2D(depth_map, 
                                coord + vec2(inv_tex_scale.x, 0.0)
                                 ).x
                        ) - eye_pos;
	vec3 zt = uv_to_eye(coord + vec2(0.0, inv_tex_scale.y), 
                        texture2D(depth_map, 
                                  coord + vec2(0.0, inv_tex_scale.y)
                                 ).x
                       ) - eye_pos;
	vec3 zb = eye_pos 
        - uv_to_eye(coord - vec2(0.0, inv_tex_scale.y), 
                    texture2D(depth_map, 
                              coord - vec2(0.0, inv_tex_scale.y)
                             ).x
                   );

    vec3 dx = zl;
	vec3 dy = zt;

	if (abs(zr.z) < abs(zl.z))
		dx = zr;

	if (abs(zb.z) < abs(zt.z))
		dy = zb;

    //vec3 dx = vec3(inverse(model_view) * vec4(uv_to_eye(dFdx(coord), depth), 1.0));
    //vec3 dy = vec3(inverse(model_view) * vec4(uv_to_eye(dFdy(coord), depth), 1.0));

    vec3 normal = normalize(cross(dx, dy));
    
	vec4 world_pos = inverse(model_view) * vec4(eye_pos, 1.0);
    //vec3 light_dir = normalize(world_pos.xyz - light_pos);

    // Phong specular
    vec3 l = (model_view * vec4(light_dir, 0.0)).xyz;
    vec3 view_dir = -normalize(eye_pos);
    vec3 half_vec = normalize(view_dir + l);
    float specular = pow(max(0.0f, dot(normal, half_vec)), shininess);	

    vec2 tex_scale = vec2(0.75f, 1.0f);
    float refract_scale = 1.33 * 0.025;
    refract_scale *= smoothstep(0.1f, 0.4f, world_pos.y);

    vec2 refract_coord = coord + normal.xy * refract_scale * tex_scale;

    // Color substraction
	vec3 transmission = exp(-(vec3(0.9804, 1.0, 0.7255) - light_color.xyz) * thickness);

    vec3 refract = texture(scene_map, refract_coord).xyz * transmission;
    vec3 l_vec = normalize(world_pos.xyz - light_pos);
    float attenuation = max(smoothstep(0.95, 1.0, abs(dot(l_vec, - light_dir))), 0.05);

    float ln = dot(l, normal) * attenuation;

    // Frensel 
    float fresnel = fres_bias + fres_scale * pow(min(0.0, 1.0-dot(normal, view_dir)), fres_power);

    //Diffuse light
	vec3 diffuse = light_color.xyz * mix(vec3(1.0, 1.0, 1.0), vec3(1.0), (ln*0.5 + 0.5)) * (1 - light_color.w);
	//vec3 diffuse = light_color.xyz * mix(vec3(0.29, 0.379, 0.59), vec3(1.0), (ln*0.5 + 0.5));

	vec3 sky_color = vec3(0.8078, 0.4784, 0.2078)*1.2;
	vec3 ground_color = vec3(1.0, 1.0, 1.0);

	vec3 r_eye = reflect(view_dir, normal).xyz;
	vec3 r_world = (inverse(model_view)*vec4(r_eye, 0.0)).xyz;

	vec3 reflect = vec3(1.0) + mix(ground_color, sky_color, smoothstep(0.15, 0.25, r_world.y));

    vec3 final_color = diffuse + (mix(refract, reflect, fresnel) + specular) * light_color.w;

    frag_color = vec4(final_color.xyz, 1.0);

    gl_FragDepth = depth;
}