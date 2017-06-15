#version 330

in float realDepth;

layout (location=0) out float depth;

void main() 
{
	depth = realDepth;
}