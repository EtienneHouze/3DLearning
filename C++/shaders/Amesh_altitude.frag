#version 330

in float realAltitude;

layout (location=0) out float altitude;

void main() 
{
	altitude = realAltitude;
}