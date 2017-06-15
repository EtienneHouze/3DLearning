#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <omp.h>

#define NOGDI

#include "camera.h"
#include "obj_parser.h"
#include "xml_parser.h"
#include "mesh_intersection_structure.h"
#include "point_projector.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <boost/filesystem.hpp>

int main(int argc, char** argv) {

	std::string xmlFile = "";
	std::string meshFile = "";
	std::string pictureFolder = "";
	std::string shaderFolder = "shaders";
	std::cout << "\n";
	for (int i = 0; i < argc; i++) {

		if (std::strcmp(argv[i], "--help") == 0) {
			std::cout << "Command-line options for point projection:\n\n";

			std::cout << "-X [path] : Specify the path to the XML file with camera information (REQUIRED)\n";
			std::cout << "-M [path] : Specify the path to the OBJ mesh (OPTIONAL)\n";
			std::cout << "--help : Print command-line options.\n\n";

			std::cout << "The resulting projections will be stored in the Projections.txt file next to the OBJ mesh provided.\n";
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

		std::cout << "Command-line options for point projection:\n\n";


		std::cout << "-X [path] : Specify the path to the XML file with camera information (REQUIRED)\n";
		std::cout << "-M [path] : Specify the path to the OBJ mesh (OPTIONAL)\n";
		std::cout << "--help : Print command-line options.\n\n";

		std::cout << "The resulting projections will be stored in the Projections.txt file in the Result folder next to the executable.\n";
		std::system("pause");
		return -1;
	}
	double time = omp_get_wtime();
	std::string folder = "./Result";

	boost::filesystem::path Dir(folder);
	boost::filesystem::create_directories(Dir);

	std::vector<Camera> cameras;
	XmlParser xmlParser(xmlFile);
	xmlParser.parseCameras(cameras);

	ObjParser objParser(meshFile);
	Polyhedron P;
	objParser.parse(P);

	std::vector<Point_3> points(P.points_begin(), P.points_end());

	std::vector<std::vector<int>>visibility = std::vector<std::vector<int>>(P.size_of_vertices());

	MeshIntersectionStructure mis(P);

#pragma omp parallel for
	for (int pt = 0; pt < points.size(); pt++) {
		Point_3 a = points[pt];

		std::vector<int> visible;
		for (int i = 0; i < cameras.size(); i++) {
			cv::Mat c = cameras[i].center;
			Point_3 center(c.at<double>(0, 0), c.at<double>(1, 0), c.at<double>(2, 0));
			Point_3 a2 = a + 0.001 * (center - a);
			if (!mis.intersect(Segment(a2, center))) {
				visible.push_back(cameras[i].id);
			}
		}
		visibility[pt] = visible;
	}

	std::vector<std::vector<std::pair<int, cv::Point2d>>> projections;

	PointProjector pp;
	pp.computeAndSaveAllProjections(std::vector<Point_3>(P.points_begin(), P.points_end()), cameras, visibility, folder + "/Projections.txt", projections);

	std::cout << "Reprojected all points in all images: " << P.size_of_vertices() * cameras.size() << " projections in " << omp_get_wtime() - time << " seconds." << std::endl;
	std::cout << "Saved results in " << folder + "/Projections.txt file." << std::endl;

	//cv::Mat image = cv::imread("test/image0.jpg");
	//cv::pyrDown(image, image);
	//for (int i = 0; i < projections.size(); i++) {
	//	for (auto pair : projections[i]) {
	//		if (pair.first == 0)
	//			cv::circle(image, pair.second / 2, 2, cv::Vec3b(0, 0, 255), 1);
	//	}
	//}


	//cv::imshow("Result", image);
	//cv::waitKey();


	std::system("pause");
	return (0);
}