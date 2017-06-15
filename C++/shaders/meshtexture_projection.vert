#version 330
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texCoord;
out vec2 oTexCoord;


uniform mat4 modelViewProj;

void main()
{
	gl_Position = modelViewProj * vec4(position, 1.0);
	oTexCoord = texCoord;
}