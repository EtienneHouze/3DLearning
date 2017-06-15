#version 330
layout (location = 0) in vec4 position;
out float realDepth;


uniform mat4 modelViewProj;

void main()
{
	gl_Position = modelViewProj * position;
	realDepth = gl_Position.w;
}