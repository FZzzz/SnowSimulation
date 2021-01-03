#version 410

layout (location = 0) in vec3 aPos;
layout (location = 1) in float aTemperature;

uniform mat4 pvm;
uniform float point_size;
uniform vec3 light_pos;
uniform vec3 camera_pos;
uniform mat4 view;
uniform vec3 point_color;

uniform vec3 hottest_color;
uniform vec3 coolest_color;

uniform float hottest_temperature;
uniform float coolest_temperature;

out vec3 light_dir;
out vec3 vertex_point_color;

// x: current temperature
vec3 CalThermograph(float x)// const float xmin = 0.0, const double xmax = 1.0)
{
    vec3 ans_color;

    float l = hottest_temperature - coolest_temperature;
    //if(abs(l) < 1e-10) return vec3(0,0,0);
    
    const int ncolors = 2;
    vec3 base[ncolors];
    //base[0] =  vec3(0.0, 0.0, 0.0);
    base[0] =  vec3(0.0, 0.0, 1.0);
    //base[1] =  vec3(0.0, 1.0, 1.0);
    //base[2] =  vec3(0.0, 1.0, 0.0);
    //base[1] =  vec3(1.0, 1.0, 0.0);
    base[1] =  vec3(1.0, 0.0, 0.0);
    //base[4] =  vec3(1.0, 1.0, 1.0);
                            
    x = clamp(((x - coolest_temperature) / l), 0.0, 1.0) * (ncolors - 1);
    int i = int(x);
    float dx = x - floor(x);
    ans_color.r = mix(base[i].r, base[i+1].r, dx);
    ans_color.g = mix(base[i].g, base[i+1].g, dx);
    ans_color.b = mix(base[i].b, base[i+1].b, dx);

    return ans_color;
}

vec3 calc_temperature_color(float temperature)
{
    vec3 ans_color;
    if(temperature < 0)
    {
        float region = 0 - coolest_temperature;
        float alpha = (temperature - coolest_temperature) / region;

        vec3 h_color = vec3(1,0,1);
        vec3 c_color = vec3(0,0,1);
        return mix(c_color, h_color, alpha);
    }
    else
    {
        float region = hottest_temperature - 0;
        float alpha = (temperature - 0) / region;

        vec3 h_color = vec3(1,0,0);
        vec3 c_color = vec3(1,0,1);
        return mix(c_color, h_color, alpha);
    }
}

void main()
{
    light_dir = normalize(vec3(light_pos - vec3(view * vec4(aPos, 1.f))));

    float temp_region = hottest_temperature - coolest_temperature;
    float alpha = (aTemperature - coolest_temperature) / temp_region;
    
    //vertex_point_color = CalThermograph(aTemperature);
    //vertex_point_color = mix(coolest_color, hottest_color, alpha);
    vertex_point_color = calc_temperature_color(aTemperature);

    vec3 v = camera_pos - aPos;
    float dist = sqrt(dot(v,v));

    gl_Position = pvm * vec4(aPos, 1.0f);
    gl_PointSize = point_size / dist;
}