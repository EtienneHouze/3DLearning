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

int main(int argc, char** argv) {
	std::string xmlFile = "";
	std::string meshFile = "";
	std::string shaderFolder = "shaders";
	std::string  folder = "./Result/Depth";
	std::cout << "\n";
	for (int i = 0; i < argc; i++) {

		if (std::strcmp(argv[i], "--help") == 0) {
			std::cout << "Command-line options for the altitude render algorithm:\n\n";

			std::cout << "-X [path] : Specify the path to the XML file with camera information (REQUIRED)\n";
			std::cout << "-M [path] : Specify the path to the OBJ mesh (OPTIONAL)\n";
			std::cout << "-F [path] : Specify thpath to the output (OPTIONAL)\n";
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
		if (std::strcmp( argv[i], "-F" ) == 0) {
			folder = argv[i + 1];
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
	/*std::string folder = "./Result";

	boost::filesystem::path Dir(folder);
	boost::filesystem::create_directories(Dir);*/


	boost::filesystem::path Dir(folder);
	boost::filesystem::create_directories(Dir);

	std::vector<Camera> cameras;
	XmlParser xmlParser(xmlFile);
	xmlParser.parseCameras(cameras);

	ObjParser objParser(meshFile);
	Polyhedron P;
	objParser.parse(P);

	int argc2 = 0;
	char* argv2 = "bllbll";
	glutInit(&argc2, &argv2);

	AltitudeRender dr;
	dr.init(cameras[0]);
	dr.loadShaders("./shaders/Amesh_altitude.vert", "./shaders/Amesh_altitude.frag");
	dr.loadPolyhedronToGPU(P);


	for (int c = 0; c < cameras.size(); c++) {
		cv::Mat altitude;
		dr.renderAltitude(cameras[c], altitude);
		altitude = -altitude;
		cv::imwrite(folder + "/" + cameras[c].filename.substr(0, cameras[c].filename.length() - 4) + ".png", altitude);
		//std::ofstream outputAltitude(folder + "/" + cameras[c].filename.substr(0,cameras[c].filename.length() - 4) + ".txt");

		//outputAltitude << altitude;
	}

	//for (int c = 0; c < 1; c++) {
	//	cv::Mat altitude;
	//	dr.renderAltitude(cameras[c], altitude);
	//	//cv::imwrite(folder + "/" + cameras[c].filename.substr(0, cameras[c].filename.length() - 4) + ".png", altitude);
	//	std::ofstream outputAltitude(folder + "/" + cameras[c].filename.substr(0,cameras[c].filename.length() - 4) + ".txt");

	//	outputAltitude << altitude;
	//}

	std::system("pause");
	return (0);
}