#version 450 core
in vec4 outColor;
in vec3 position;
in vec3 normal;
out vec4 FragColor;

uniform vec3 eyePos;

vec3 ambient = vec3(0.2);
vec3 lightColor = vec3(1.0, 1.0, 1.0);

void main()
{
    FragColor = outColor;
}
