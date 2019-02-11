#version 450 core
in vec2 outUv;
out vec4 FragColor;

uniform sampler2D viewportTexture;

void main()
{
    FragColor = texture(viewportTexture, outUv);
}
