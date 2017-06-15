#pragma once

#include <string>
#include <opencv2/core/core.hpp>

struct Camera
{
	Camera(){};
	~Camera(){};

	std::string filename;

	int width;
	int height;
	int id;

	cv::Mat rotation;
	cv::Mat center;
	cv::Mat distortion;
	cv::Point2d principalPoint;

	double focalLength;
	double sensorSize;
};

