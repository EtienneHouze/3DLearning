#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <random>

#include "obj_parser.h"
#include "xml_parser.h"
#include "depth_render.h"
#include <boost/filesystem.hpp>

#include <omp.h>

using namespace std;

int main(int argc, char** argv) {
	std::string xmlFile = "";
	std::string meshFile = "";
	std::string shaderFolder = "shaders";
	std::cout << "\n";
	for (int i = 0; i < argc; i++) {

		if (std::strcmp(argv[i], "--help") == 0) {
			std::cout << "Command-line options for the altitude render algorithm:\n\n";

			std::cout << "-X [path] : Specify the path to the XML file with camera information (REQUIRED)\n";
			std::cout << "-M [path] : Specify the path to the OBJ mesh (OPTIONAL)\n";
			std::cout << "--help : Print command-line options.\n\n";

			std::cout << "All results will be stored in a Results folder next to the executable.\n";

			std::system("pause");
			return 0;
		}

		if (std::strcmp(argv[i], "-M") == 0) {
			meshFile = argv[i + 1];
		}
		if (std::strcmp(argv[i], "-X") == 0) {
			xmlFile = argv[i + 1];
		}
	}

	if (meshFile.length() == 0 || xmlFile.length() == 0) {
		std::cout << "Some required informations have not been provided. \n\n";

		std::cout << "Command-line options for the altitude render algorithm:\n\n";


		std::cout << "-X [path] : Specify the path to the XML file with camera information (REQUIRED)\n";
		std::cout << "-M [path] : Specify the path to the OBJ mesh (OPTIONAL)\n";
		std::cout << "--help : Print command-line options.\n\n";

		std::cout << "All results will be stored in a Results folder next to the executable.\n";
		std::system("pause");
		return -1;
	}

	double time = omp_get_wtime();
	std::string folder = "./Result/RGB";

	boost::filesystem::path Dir(folder);
	boost::filesystem::create_directories(Dir);

	std::vector<Camera> cameras;
	XmlParser xmlParser(xmlFile);
	xmlParser.parseCameras(cameras);

	
	std::vector<glm::vec3> v;
	std::vector<glm::vec2> vt;
	std::vector<std::vector<glm::ivec3> > fvs;
	std::vector<std::vector<glm::ivec3> > fvts;
	std::vector<std::string> texFileNames;

	std::cout << "Begin readObjFile..." << std::endl;
	readObjFile(meshFile, v, vt, fvs, fvts, texFileNames);
	for (int i=0; i<texFileNames.size(); ++i)
	{
		std::cout << texFileNames[i] << std::endl;
		std::cout << fvs[i].size() << " " << fvts[i].size() << std::endl;
	}


	int argc2 = 0;
	char* argv2 = "bllbll";
	glutInit(&argc2, &argv2);

	TextureRender dr;
	dr.init(cameras[0].width, cameras[0].height);
	dr.loadShaders("./shaders/meshtexture_projection.vert", "./shaders/meshtexture_projection.frag");
	dr.loadMeshToGPU(v, vt, fvs, fvts, texFileNames);

	cout << "Begin rendering..." << endl;
	for (int c = 0; c < cameras.size(); c++) {
		cout << "\n----CAMERA---- " << c << endl;
		cv::Mat render;
		dr.renderTexture(cameras[c], render);

		std::cout << "Write image to: " << folder + "/" + cameras[c].filename.substr(0, cameras[c].filename.length() - 4) + ".png" << std::endl;
		cv::imwrite(folder + "/" + cameras[c].filename.substr(0,cameras[c].filename.length() - 4) + ".png", render);

	}

	std::system("pause");
	return (0);
}