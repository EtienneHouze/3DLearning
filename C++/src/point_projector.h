#pragma once



#include <fstream>

#include <CGAL/simple_cartesian.h>

#include "camera.h"

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point_3;

class PointProjector
{
public:
	PointProjector() {};
	~PointProjector();
	std::vector<cv::Point2d> projectPoints(
		const std::vector<Point_3> &points,
		const Camera &c,
		const std::vector<std::vector<int>> &visibility
	);
	cv::Point2d projectPoint(const Point_3 &p,
		const Camera &c,
		const double &width,
		const double &height
	);

	void computeAndSaveAllProjections(const std::vector<Point_3> &points,
		const std::vector<Camera> &cameras,
		const std::vector<std::vector<int>> &visibility,
		std::string &path,
		std::vector<std::vector<std::pair<int, cv::Point2d>>> &projections,
		std::vector<std::vector<double>> &positions
	);
private:

	cv::Point2d distortFunc(
		cv::Point2d &pt,
		const cv::Mat &distortion
	);



};

