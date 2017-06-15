#version 330
layout (location = 0) in vec4 position;
out float realAltitude;


uniform mat4 modelViewProj;

void main()
{
	gl_Position = modelViewProj * position;
	realAltitude = position.z;
}