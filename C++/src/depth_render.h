#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <fstream>

#include <random>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_items_with_id_3.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "camera.h"

#pragma warning(disable: 4700)

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

typedef CGAL::Simple_cartesian<double>     Kernel;
typedef Kernel::Vector_3 Vector_3;
typedef CGAL::Polyhedron_3<Kernel, CGAL::Polyhedron_items_with_id_3>         Polyhedron;

inline bool checkGLErrors(const std::string& msg)
{
	GLenum error = glGetError();
	if (error == GL_NO_ERROR) return true;
	if (!msg.empty()) std::cerr << msg << ": ";
	std::cerr << "GL error: " << error << std::endl;
	exit(1);
	return false;
}

class AltitudeRender
{
public:
	AltitudeRender() {
		//width = 100;
		//height = 100;
	};

	void init(const Camera &c) {
		initialized = true;
		
		glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH/* | GLUT_MULTISAMPLE*/);
		//glutInitWindowSize(width, height);
		window = glutCreateWindow("glut window");
		glutHideWindow();
		
		glewInit();

		if (glIsFramebuffer(fbo)) {
			glDeleteFramebuffers(1, &fbo);
		}
		glGenFramebuffers(1, &fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);

		GLuint depthrenderbuffer=0;
		if (glIsRenderbuffer(depthrenderbuffer)) {
			glDeleteRenderbuffers(1, &depthrenderbuffer);
		}
		glGenRenderbuffers(1, &depthrenderbuffer);
		glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, c.width, c.height);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

		GLuint texture=0;
		if (glIsTexture(texture)) {
			glDeleteTextures(1, &texture);
		}
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glBindTexture(GL_TEXTURE_2D, 0);

		glBindTexture(GL_TEXTURE_2D, texture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, c.width, c.height, 0, GL_RGBA, GL_FLOAT, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
		glBindTexture(GL_TEXTURE_2D, 0);

		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
			std::cout << "Framebuffer for shadow map has a problem:" << std::endl;
			switch (glCheckFramebufferStatus(GL_FRAMEBUFFER)) {
			case GL_FRAMEBUFFER_UNDEFINED:
				std::cout << "Framebuffer undefined" << std::endl;
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
				std::cout << "Framebuffer incomplete attachment" << std::endl;
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
				std::cout << "Framebuffer incomplete draw buffer" << std::endl;
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
				std::cout << "Framebuffer incomplete read buffer" << std::endl;
				break;
			case GL_FRAMEBUFFER_UNSUPPORTED:
				std::cout << "Framebuffer unsupported" << std::endl;
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
				std::cout << "Framebuffer incomplete multisample" << std::endl;
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
				std::cout << "Framebuffer incomplete layer targets" << std::endl;
			}
		}
	};

	void loadShaders(const std::string& vertexSource, const std::string& fragmentSource) {
		GLuint vertexShader, fragmentShader;

		std::ifstream file(vertexSource);
		if (!file.is_open())
		{
			std::cout << "argh" << std::endl;
			exit(1);
		}
		std::string text, line;
		while (std::getline(file, line)) {
			text += line + "\n";
		}
		const GLchar* Buffer = text.c_str();

		int size = (int)text.length();

		vertexShader = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertexShader, 1, &Buffer, &size);
		glCompileShader(vertexShader);

		GLint compiled;
		glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &compiled);
		if (compiled != GL_TRUE) {
			GLchar infoLog[200];
			GLint size;
			glGetShaderInfoLog(vertexShader, 200, &size, infoLog);

			std::cout << "Vertex shader log: " << infoLog << std::endl;
		}


		//////////////////////////////////////////////////////////////////////


		std::ifstream file2(fragmentSource);
		std::string text2, line2;
		while (std::getline(file2, line2)) {
			text2 += line2 + "\n";
		}
		const GLchar* Buffer2 = text2.c_str();
		int size2 = (int)text2.length();

		fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragmentShader, 1, &Buffer2, &size2);
		glCompileShader(fragmentShader);

		GLint compiled2;
		glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &compiled2);
		if (compiled2 != GL_TRUE) {
			GLchar infoLog[200];
			GLint size;
			glGetShaderInfoLog(fragmentShader, 200, &size, infoLog);

			std::cout << "Fragment shader log: " << infoLog << std::endl;
		}


		//////////////////////////////////////////////////////////////////////


		if (glIsProgram(altitudeProgram)) {
			glDeleteProgram(altitudeProgram);
		}
		altitudeProgram = glCreateProgram();
		glAttachShader(altitudeProgram, vertexShader);
		glAttachShader(altitudeProgram, fragmentShader);
		glLinkProgram(altitudeProgram);


		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);
	}

	 

	void loadPolyhedronToGPU(const Polyhedron &P) {

		std::vector<int> faces;
		faces.reserve(P.size_of_facets() * 3);

		for (auto it = P.facets_begin(); it != P.facets_end(); it++) {
			faces.push_back((int)it->halfedge()->vertex()->id());
			faces.push_back((int)it->halfedge()->next()->vertex()->id());
			faces.push_back((int)it->halfedge()->next()->next()->vertex()->id());
		}

		std::vector<float> positions;
		positions.reserve(P.size_of_facets() * 12);

		for (auto it = P.facets_begin(); it != P.facets_end(); it++) {
			auto pt1 = it->halfedge()->vertex()->point();
			positions.push_back((float)pt1.x());
			positions.push_back((float)pt1.y());
			positions.push_back((float)pt1.z());
			positions.push_back(1.0f);

			auto pt2 = it->halfedge()->next()->vertex()->point();
			positions.push_back((float)pt2.x());
			positions.push_back((float)pt2.y());
			positions.push_back((float)pt2.z());
			positions.push_back(1.0f);

			auto pt3 = it->halfedge()->next()->next()->vertex()->point();
			positions.push_back((float)pt3.x());
			positions.push_back((float)pt3.y());
			positions.push_back((float)pt3.z());
			positions.push_back(1.0f);
		}

		//indicesSize = (int)indicesVector.size();
		verticesSize = (int)P.size_of_facets() * 3;

		//////////////////////////////////////////////////////////////////////


		if (glIsBuffer(vertices)) {
			glDeleteBuffers(1, &vertices);
		}
		glGenBuffers(1, &vertices);

		glBindBuffer(GL_ARRAY_BUFFER, vertices);

		glBufferData(GL_ARRAY_BUFFER, verticesSize * 4 * sizeof(float), 0, GL_STATIC_DRAW);

		glBufferSubData(GL_ARRAY_BUFFER, 0, verticesSize * 4 * sizeof(float), &positions[0]);


		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void loadMeshToGPU(const std::vector<glm::vec3>& v, const std::vector<glm::vec2>& vt,
		const std::vector<std::vector<glm::ivec3> >& fvs,
		const std::vector<std::vector<glm::ivec3> >& fvts,
		const std::vector<std::string>& texFileNames) {


		if (sizeof(glm::vec3) != 3 * sizeof(float))
		{
			std::cout << "ERROR: glm::vec3 is not tightly packed. ";
			return;
		}

		if (sizeof(glm::vec2) != 2 * sizeof(float))
		{
			std::cout << "ERROR: glm::vec2 is not tightly packed. ";
			return;
		}

		size_t nTextures = fvs.size();
		if (fvts.size() != nTextures)
		{
			std::cout << "ERROR of size in loadMeshToGpu!!" << std::endl;
			return;
		}

		std::vector<glm::vec3> data_v;
		std::vector<glm::vec2> data_vt;

		int current = 0;
		for (size_t texId = 0; texId < nTextures; ++texId)
		{
			firsts.push_back(current);
			size_t nFacets = fvs[texId].size();
			const auto& fv = fvs[texId];
			const auto& fvt = fvts[texId];
			if (fvt.size() != nFacets)
			{
				std::cout << "ERROR of size 2 in loadMeshToGpu!!" << std::endl;
			}
			for (size_t i = 0; i < nFacets; ++i)
			{
				data_v.push_back(v[fv[i][0]]);
				data_v.push_back(v[fv[i][1]]);
				data_v.push_back(v[fv[i][2]]);

				data_vt.push_back(vt[fvt[i][0]]);
				data_vt.push_back(vt[fvt[i][1]]);
				data_vt.push_back(vt[fvt[i][2]]);
			}

			current += 3 * nFacets;
		}


		//////////////////////////////////////////////////////////////////////


		if (glIsBuffer(vertices)) {
			glDeleteBuffers(1, &vertices);
		}
		glGenBuffers(1, &vertices);
		glBindBuffer(GL_ARRAY_BUFFER, vertices);
		glBufferData(GL_ARRAY_BUFFER, data_v.size()*sizeof(glm::vec3), data_v.data(), GL_STATIC_DRAW);

		if (glIsBuffer(texCoords)) {
			glDeleteBuffers(1, &texCoords);
		}
		glGenBuffers(1, &texCoords);
		glBindBuffer(GL_ARRAY_BUFFER, texCoords);
		glBufferData(GL_ARRAY_BUFFER, data_vt.size()*sizeof(glm::vec2), data_vt.data(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		texNames = texFileNames;
	}

	void renderAltitude(const Camera &c, cv::Mat &depthMap)const {
		glViewport(0, 0, c.width, c.height);

		glUseProgram(altitudeProgram);

		glBindBuffer(GL_ARRAY_BUFFER, vertices);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);

		glm::mat4 modelViewProj;
		modelViewProj = computeModelViewProj(c);
		glUniformMatrix4fv(glGetUniformLocation(altitudeProgram, "modelViewProj"), 1, false, glm::value_ptr(modelViewProj));

		glBindFramebuffer(GL_FRAMEBUFFER, fbo);

		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);

		glDrawArrays(GL_TRIANGLES, 0, verticesSize);

		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadBuffer(GL_COLOR_ATTACHMENT0);

		depthMap = cv::Mat(c.height, c.width, CV_32F, cv::Scalar(10));
		glReadPixels(0, 0, c.width, c.height, GL_RED, GL_FLOAT, depthMap.data);
	}

	~AltitudeRender(){
		if (initialized) {
			destroy();
		}
	};
private:
	bool initialized = false;

	GLuint vertices;
	GLuint indices;//it seems we do not use it
	GLuint texCoords=0;

	std::vector<int> firsts;
	std::vector<std::string> texNames;

	GLuint altitudeProgram;
	GLuint textureProgram = 0;
	GLuint fbo;

	int indicesSize;
	int verticesSize;

	int width, height;

	int window;

	glm::mat4 computeModelViewProj(const Camera &c) const {
		float nearDist, farDist;
		nearDist = 1.0f;
		farDist = 10000.0f;

		float left, right, top, bottom;
		left = (0);
		right = -(c.width) * nearDist;
		bottom = (0);
		top = -(c.height) * nearDist;

		glm::mat4 alignPixels(1.0f);
		alignPixels[0][3] = -1.0f / c.width;
		alignPixels[1][3] = -1.0f / c.height;

		alignPixels = glm::transpose(alignPixels);

		glm::mat4 frustumMatrix(1.0f);

		frustumMatrix[0][0] = 2.0f * nearDist / (right - left);
		frustumMatrix[0][2] = (right + left) / (right - left);
		frustumMatrix[1][1] = 2.0f * nearDist / (top - bottom);
		frustumMatrix[1][2] = (top + bottom) / (top - bottom);
		frustumMatrix[2][2] = -(farDist + nearDist) / (farDist - nearDist);
		frustumMatrix[2][3] = -2.0f * farDist * nearDist / (farDist - nearDist);
		frustumMatrix[3][2] = -1.0f;
		frustumMatrix[3][3] = 0.0f;

		frustumMatrix = glm::transpose(frustumMatrix);

		glm::mat4 K(1.0f);
		K[0][0] = (float)c.focalLength;
		K[0][2] = (float)c.principalPoint.x;
		K[1][1] = (float)c.focalLength /** ((double)c.height / c.width )*/;
		K[1][2] = (float)c.principalPoint.y;
		K = glm::transpose(K);


		glm::mat4 modelView(1.0f);

		cv::Mat vector = c.rotation * c.center;

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				modelView[i][j] = (float)c.rotation.at<double>(i, j);
			}
			modelView[i][3] = -(float)vector.at<double>(i, 0);
		}
		modelView = glm::transpose(modelView);



		modelView = K * modelView;

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				modelView[j][i] *= -1;
			}
		}
		glm::mat4 modelViewProj;
		modelViewProj = alignPixels * frustumMatrix * modelView;

		return modelViewProj;
	}

	void destroy() {
		glDeleteBuffers(1, &vertices);
		glDeleteBuffers(1, &indices);
		glDeleteProgram(altitudeProgram);
		glDeleteFramebuffers(1, &fbo);
		glutLeaveMainLoop();
		glutDestroyWindow(window);
	}
};


class TextureRender
{
public:
	TextureRender() {
		//width = 100;
		//height = 100;
	};

	void init(int w, int h) {
		std::cout << "TextureRender::init(" << w << ", " << h << ")" << std::endl;
		initialized = true;
		glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
		window = glutCreateWindow("glut window");
		glutHideWindow();

		glewInit();


		if (glIsFramebuffer(fbo)) {
			glDeleteFramebuffers(1, &fbo);
		}
		glGenFramebuffers(1, &fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);

		depthrenderbuffer = 0;
		glGenRenderbuffers(1, &depthrenderbuffer);
		glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

		outTexture = 0;
		glGenTextures(1, &outTexture);
		glBindTexture(GL_TEXTURE_2D, outTexture);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

		//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, outTexture, 0);
		//glBindTexture(GL_TEXTURE_2D, 0);

		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
			std::cout << "Framebuffer for shadow map has a problem:" << std::endl;
			switch (glCheckFramebufferStatus(GL_FRAMEBUFFER)) {
			case GL_FRAMEBUFFER_UNDEFINED:
				std::cout << "Framebuffer undefined" << std::endl;
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
				std::cout << "Framebuffer incomplete attachment" << std::endl;
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
				std::cout << "Framebuffer incomplete draw buffer" << std::endl;
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
				std::cout << "Framebuffer incomplete read buffer" << std::endl;
				break;
			case GL_FRAMEBUFFER_UNSUPPORTED:
				std::cout << "Framebuffer unsupported" << std::endl;
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
				std::cout << "Framebuffer incomplete multisample" << std::endl;
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
				std::cout << "Framebuffer incomplete layer targets" << std::endl;
			}
		}

		std::cout << "END TextureRender::init(" << w << ", " << h << ")" << std::endl;
	};

	void loadShaders(const std::string& vertexSource, const std::string& fragmentSource) {
		GLuint vertexShader, fragmentShader;
		std::cout << "loadShader" << std::endl;
		std::ifstream file(vertexSource);
		if (!file.is_open())
		{
			std::cout << "Can not read vertex shader" << vertexSource << std::endl;
			exit(1);
		}
		std::string text, line;
		while (std::getline(file, line)) {
			text += line + "\n";
		}
		const GLchar* Buffer = text.c_str();

		int size = (int)text.length();

		vertexShader = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertexShader, 1, &Buffer, &size);
		glCompileShader(vertexShader);

		GLint compiled;
		glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &compiled);
		if (compiled != GL_TRUE) {
			GLchar infoLog[200];
			GLint size;
			glGetShaderInfoLog(vertexShader, 200, &size, infoLog);

			std::cout << "Vertex shader log: " << infoLog << std::endl;
		}


		//////////////////////////////////////////////////////////////////////


		std::ifstream file2(fragmentSource);
		if (!file2.is_open())
		{
			std::cout << "Can not read fragment shader" <<fragmentSource<< std::endl;
			exit(1);
		}
		std::string text2, line2;
		while (std::getline(file2, line2)) {
			text2 += line2 + "\n";
		}
		const GLchar* Buffer2 = text2.c_str();
		int size2 = (int)text2.length();

		fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragmentShader, 1, &Buffer2, &size2);
		glCompileShader(fragmentShader);

		GLint compiled2;
		glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &compiled2);
		if (compiled2 != GL_TRUE) {
			GLchar infoLog[200];
			GLint size;
			glGetShaderInfoLog(fragmentShader, 200, &size, infoLog);

			std::cout << "Fragment shader log: " << infoLog << std::endl;
		}


		//////////////////////////////////////////////////////////////////////


		if (glIsProgram(textureProgram)) {
			glDeleteProgram(textureProgram);
		}
		textureProgram = glCreateProgram();
		glAttachShader(textureProgram, vertexShader);
		glAttachShader(textureProgram, fragmentShader);
		glLinkProgram(textureProgram);


		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);

		std::cout << "END load shader" << std::endl;
	}


	void loadMeshToGPU(const std::vector<glm::vec3>& v, const std::vector<glm::vec2>& vt,
		const std::vector<std::vector<glm::ivec3> >& fvs,
		const std::vector<std::vector<glm::ivec3> >& fvts,
		const std::vector<std::string>& texFileNames) {

		std::cout << "loadMeshtoGPU" << std::endl;

		if (sizeof(glm::vec3) != 3 * sizeof(float))
		{
			std::cout << "ERROR: glm::vec3 is not tightly packed. ";
			return;
		}

		if (sizeof(glm::vec2) != 2 * sizeof(float))
		{
			std::cout << "ERROR: glm::vec2 is not tightly packed. ";
			return;
		}

		size_t nTextures = fvs.size();
		if (fvts.size() != nTextures)
		{
			std::cout << "ERROR of size in loadMeshToGpu!!" << std::endl;
			return;
		}

		std::vector<glm::vec3> data_v;
		std::vector<glm::vec2> data_vt;
		firsts.clear();
		int current = 0;
		for (size_t texId = 0; texId < nTextures; ++texId)
		{
			firsts.push_back(current);
			size_t nFacets = fvs[texId].size();
			const auto& fv = fvs[texId];
			const auto& fvt = fvts[texId];
			if (fvt.size() != nFacets)
			{
				std::cout << "ERROR of size 2 in loadMeshToGpu!!" << std::endl;
			}
			for (size_t i = 0; i < nFacets; ++i)
			{
				data_v.push_back(v[fv[i][0]]);
				data_v.push_back(v[fv[i][1]]);
				data_v.push_back(v[fv[i][2]]);

				data_vt.push_back(vt[fvt[i][0]]);
				data_vt.push_back(vt[fvt[i][1]]);
				data_vt.push_back(vt[fvt[i][2]]);
			}

			current += 3 * nFacets;
		}
		firsts.push_back(current);

		//////////////////////////////////////////////////////////////////////


		if (glIsBuffer(vertices)) 
		{
			glDeleteBuffers(1, &vertices);
		}
		glGenBuffers(1, &vertices);
		glBindBuffer(GL_ARRAY_BUFFER, vertices);
		glBufferData(GL_ARRAY_BUFFER, data_v.size() * sizeof(glm::vec3), data_v.data(), GL_STATIC_DRAW);

		if (glIsBuffer(texCoords)) 
		{
			glDeleteBuffers(1, &texCoords);
		}

		glGenBuffers(1, &texCoords);
		glBindBuffer(GL_ARRAY_BUFFER, texCoords);
		glBufferData(GL_ARRAY_BUFFER, data_vt.size() * sizeof(glm::vec2), data_vt.data(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		texNames = texFileNames;

	}

	void renderTexture(const Camera &c, cv::Mat &render) {
		checkGLErrors("M01");

		glViewport(0, 0, c.width, c.height);//condition: c.width and c.height has the same dimension as in init(w, h)
		glUseProgram(textureProgram);

		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertices);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
		
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, texCoords);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
		

		{
			glm::mat4 modelViewProj;
			modelViewProj = computeModelViewProj(c);
			auto matrixLocation = glGetUniformLocation(textureProgram, "modelViewProj");
			glUniformMatrix4fv(matrixLocation, 1, false, glm::value_ptr(modelViewProj));
		}
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);

		glClearColor(0.0, 0.0, 0.0, 0.0);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);
		

		checkGLErrors("M07");

		//we have to draw with many textures
		for (int tex = 0; tex < texNames.size(); ++tex)
		{
			//load the image
			cv::Mat image;
			std::cout << "Load texture "<<tex<<" from " << texNames[tex] << std::endl;
			image = cv::imread(texNames[tex], CV_LOAD_IMAGE_COLOR);
			std::cout << "texture.col, row, channel =" << image.cols << " " << image.rows <<" "<<image.channels()<< std::endl;
			cv::flip(image, image, 0);
			glGenTextures(1, &inTexture);
			glBindTexture(GL_TEXTURE_2D, inTexture);
			// Set texture clamping method
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

			glTexImage2D(GL_TEXTURE_2D,     // Type of texture
				0,                 // Pyramid level (for mip-mapping) - 0 is the top level
				GL_RGB,            // Internal colour format to convert to
				image.cols,          
				image.rows,          
				0,                 // Border width in pixels (can either be 1 or 0)
				GL_BGR, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
				GL_UNSIGNED_BYTE,  // Image data type
				image.ptr());        // The actual image data itself

			//mipmap 
			glGenerateMipmap(GL_TEXTURE_2D);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);


			
#if 0
			//put it in the shader in the general form
			{
				int location = glGetUniformLocation(textureProgram, "tex");

				if (location >= 0)
				{
					//note that we use "textureId+1", meaning we active GL_TEXTURE1+textureId
					//because we want other texture operation, using slot GL_TEXTURE0, does not interfere with this function					
					glActiveTexture(GL_TEXTURE0 + 1);
					glBindTexture(GL_TEXTURE_2D, inTexture);				
					glUniform1i(location, 1);					
					
					glActiveTexture(GL_TEXTURE0);//reset the default active texture
				}
				else
				{
					std::cout << "Can not find location=" << location << std::endl;
				}
			}
#else
			//because we use only one texture, this code is enough for the shader to read the texture !
			{
				glBindTexture(GL_TEXTURE_2D, inTexture);
			}
#endif
			//drawing
			
			//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); // debug only
			glDrawArrays(GL_TRIANGLES, firsts[tex], firsts[tex + 1] - firsts[tex]);
			

			glDeleteTextures(1, &inTexture);
			inTexture = 0;
		}
	
		glBindTexture(GL_TEXTURE_2D, outTexture);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);

		render = cv::Mat(c.height, c.width, CV_8UC3, cv::Scalar(0));
		//cv::mat: BGR
		glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, render.data);

		glUseProgram(0);
	}

	~TextureRender() {
		if (initialized) {
			destroy();
		}
	};
private:
	bool initialized = false;

	GLuint vertices;
	GLuint texCoords = 0;

	GLuint outTexture, inTexture;
	GLuint depthrenderbuffer;
	std::vector<int> firsts;
	std::vector<std::string> texNames;

	GLuint textureProgram = 0;
	GLuint fbo=0;

	int indicesSize;
	int verticesSize;

	int window;

	glm::mat4 computeModelViewProj(const Camera &c) const {
		float nearDist, farDist;
		nearDist = 1.0f;
		farDist = 10000.0f;

		float left, right, top, bottom;
		left = (0);
		right = -(c.width) * nearDist;
		bottom = (0);
		top = -(c.height) * nearDist;

		glm::mat4 alignPixels(1.0f);
		alignPixels[0][3] = -1.0f / c.width;
		alignPixels[1][3] = -1.0f / c.height;

		alignPixels = glm::transpose(alignPixels);

		glm::mat4 frustumMatrix(1.0f);

		frustumMatrix[0][0] = 2.0f * nearDist / (right - left);
		frustumMatrix[0][2] = (right + left) / (right - left);
		frustumMatrix[1][1] = 2.0f * nearDist / (top - bottom);
		frustumMatrix[1][2] = (top + bottom) / (top - bottom);
		frustumMatrix[2][2] = -(farDist + nearDist) / (farDist - nearDist);
		frustumMatrix[2][3] = -2.0f * farDist * nearDist / (farDist - nearDist);
		frustumMatrix[3][2] = -1.0f;
		frustumMatrix[3][3] = 0.0f;

		frustumMatrix = glm::transpose(frustumMatrix);

		glm::mat4 K(1.0f);
		K[0][0] = (float)c.focalLength;
		K[0][2] = (float)c.principalPoint.x;
		K[1][1] = (float)c.focalLength /** ((double)c.height / c.width )*/;
		K[1][2] = (float)c.principalPoint.y;
		K = glm::transpose(K);


		glm::mat4 modelView(1.0f);

		cv::Mat vector = c.rotation * c.center;

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				modelView[i][j] = (float)c.rotation.at<double>(i, j);
			}
			modelView[i][3] = -(float)vector.at<double>(i, 0);
		}
		modelView = glm::transpose(modelView);



		modelView = K * modelView;

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				modelView[j][i] *= -1;
			}
		}
		glm::mat4 modelViewProj;
		modelViewProj = alignPixels * frustumMatrix * modelView;

		return modelViewProj;
	}

	void destroy() {
		glDeleteBuffers(1, &vertices);
		glDeleteBuffers(1, &texCoords);
		glDeleteProgram(textureProgram);
		glDeleteFramebuffers(1, &fbo);
		glutLeaveMainLoop();
		glutDestroyWindow(window);
	}
};

