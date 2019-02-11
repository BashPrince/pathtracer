#version 450 core
layout (location = 0) in vec3 vPos;
layout (location = 1) in vec2 uv;

out vec2 outUv;

void main()
{
    gl_Position = vec4(vPos, 1.0);
    outUv = uv;
}
