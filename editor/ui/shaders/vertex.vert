#version 450 core
layout (location = 0) in vec3 vPos;
layout (location = 1) in vec3 vNormal;

out vec4 outColor;
out vec3 position;
out vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec4 inColor;

void main()
{
    gl_Position = projection * view * model * vec4(vPos, 1.0);
    outColor = inColor;
    position = vec3(model * vec4(vPos, 1.0));
    normal = vec3(transpose(inverse(model)) * vec4(vNormal, 0.0));
}
