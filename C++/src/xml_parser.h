#pragma once


#include <string>
#include <iostream>
#include <ctime>
#include <map>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include <CGAL/Simple_cartesian.h>

#include "camera.h"


#define USE_TIE_POINTS true


typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point_3;
using boost::property_tree::ptree;

struct TiePoint {
	Point_3 position;
	std::map<int, cv::Point2d> projections;
};

inline std::ostream& operator<<(std::ostream& os, const TiePoint &pt) {
	os << pt.position << ":\n";

	int c = 0;
	for (auto p : pt.projections) {
		os << "\t" << p.first << ": " << p.second;
		if (c < pt.projections.size() - 1)
			std::cout << ",\n";
		else
			std::cout << ";";
		c++;
	}

	return os;
}

class XmlParser
{
public:
	XmlParser(const std::string &filename) : path(filename){};
	~XmlParser();

	void parseCameras(std::vector<Camera> &cameras);
	void parseCamerasAndTiePoints(std::vector<Camera> &cameras, std::vector<TiePoint> &tiePoints);


	std::string path;
};

