#include "xml_parser.h"

XmlParser::~XmlParser()
{
}

void XmlParser::parseCameras(std::vector<Camera> &cameras) {

	std::clock_t time = std::clock();

	ptree pt;

	read_xml(path, pt);

	for (ptree::value_type &photogroup : pt.get_child("BlocksExchange.Block.Photogroups")) {

		int width = photogroup.second.get<int>("ImageDimensions.Width");
		int height = photogroup.second.get<int>("ImageDimensions.Height");
		double focalLength, sensorSize;
		try {
			focalLength = photogroup.second.get<double>("FocalLength");
			sensorSize = photogroup.second.get<double>("SensorSize");
			focalLength *= (double)width / sensorSize;
		}
		catch (std::exception e) {
			focalLength = photogroup.second.get<double>("FocalLengthPixels");
		}
		

		

		cv::Mat distortion(5, 1, CV_64F);
		try{
			
			distortion.at<double>(0, 0) = photogroup.second.get<double>("Distortion.K1");
			distortion.at<double>(1, 0) = photogroup.second.get<double>("Distortion.K2");
			distortion.at<double>(2, 0) = photogroup.second.get<double>("Distortion.P1");
			distortion.at<double>(3, 0) = photogroup.second.get<double>("Distortion.P2");
			distortion.at<double>(4, 0) = photogroup.second.get<double>("Distortion.K3");
		}
		catch (std::exception e) {
			std::cout << "No distortion found." << std::endl;
		}


		cv::Point2d principalPoint = cv::Point2d(photogroup.second.get<double>("PrincipalPoint.x"), photogroup.second.get<double>("PrincipalPoint.y"));


		for(ptree::value_type &v : photogroup.second.get_child("")) {
			if (v.first != "Photo") {
				continue;
			}
			else {
				Camera c;
				c.rotation = cv::Mat(3, 3, CV_64F);
				std::string imagePath = v.second.get<std::string>("ImagePath");
				c.filename = imagePath.substr(imagePath.find_last_of('\\') + 1);
				c.filename = c.filename.substr(c.filename.find_last_of('/') + 1);

				c.id = v.second.get<int>("Id");
			
				for (int i = 0; i < 3; i++) {
					for (int j = 0; j < 3; j++) {
						std::string coord = "Pose.Rotation.M_" + std::to_string(i) + std::to_string(j);
						c.rotation.at<double>(i, j) = v.second.get<double>(coord);
					}
				}
				c.center = cv::Mat(3, 1, CV_64F);
				c.center.at<double>(0, 0) = v.second.get<double>("Pose.Center.x");
				c.center.at<double>(1, 0) = v.second.get<double>("Pose.Center.y");
				c.center.at<double>(2, 0) = v.second.get<double>("Pose.Center.z");

				c.width = width;
				c.height = height;
				c.sensorSize = sensorSize;
				c.principalPoint = principalPoint;
				c.focalLength = focalLength;
				c.distortion = distortion;
				cameras.push_back(c);
			}
		}
	}

	std::cout << "Parsing of XML file done in " << (double)(std::clock() - time) / CLOCKS_PER_SEC << " seconds." << std::endl;
	std::cout << "Obtained " << cameras.size() << " cameras." << std::endl;
}

void XmlParser::parseCamerasAndTiePoints(std::vector<Camera> &cameras, std::vector<TiePoint> &tiePoints) {

	std::clock_t time = std::clock();

	ptree pt;

	read_xml(path, pt);

	for (ptree::value_type &photogroup : pt.get_child("BlocksExchange.Block.Photogroups")) {

		int width = photogroup.second.get<int>("ImageDimensions.Width");
		int height = photogroup.second.get<int>("ImageDimensions.Height");
		double focalLength, sensorSize;
		try {
			focalLength = photogroup.second.get<double>("FocalLength");
			sensorSize = photogroup.second.get<double>("SensorSize");
			focalLength *= (double)width / sensorSize;
		}
		catch (std::exception e) {
			focalLength = photogroup.second.get<double>("FocalLengthPixels");
		}




		cv::Mat distortion(5, 1, CV_64F);
		try{

			distortion.at<double>(0, 0) = photogroup.second.get<double>("Distortion.K1");
			distortion.at<double>(1, 0) = photogroup.second.get<double>("Distortion.K2");
			distortion.at<double>(2, 0) = photogroup.second.get<double>("Distortion.P1");
			distortion.at<double>(3, 0) = photogroup.second.get<double>("Distortion.P2");
			distortion.at<double>(4, 0) = photogroup.second.get<double>("Distortion.K3");
		}
		catch (std::exception e) {
			std::cout << "No distortion found." << std::endl;
		}


		cv::Point2d principalPoint = cv::Point2d(photogroup.second.get<double>("PrincipalPoint.x"), photogroup.second.get<double>("PrincipalPoint.y"));


		for (ptree::value_type &v : photogroup.second.get_child("")) {
			if (v.first != "Photo") {
				continue;
			}
			else {
				Camera c;
				c.rotation = cv::Mat(3, 3, CV_64F);
				std::string imagePath = v.second.get<std::string>("ImagePath");
				c.filename = imagePath.substr(imagePath.find_last_of('\\') + 1);
				c.filename = c.filename.substr(c.filename.find_last_of('/') + 1);

				c.id = v.second.get<int>("Id");

				for (int i = 0; i < 3; i++) {
					for (int j = 0; j < 3; j++) {
						std::string coord = "Pose.Rotation.M_" + std::to_string(i) + std::to_string(j);
						c.rotation.at<double>(i, j) = v.second.get<double>(coord);
					}
				}

				c.center = cv::Mat(3, 1, CV_64F);
				c.center.at<double>(0, 0) = v.second.get<double>("Pose.Center.x");
				c.center.at<double>(1, 0) = v.second.get<double>("Pose.Center.y");
				c.center.at<double>(2, 0) = v.second.get<double>("Pose.Center.z");

				c.width = width;
				c.height = height;
				c.sensorSize = sensorSize;
				c.principalPoint = principalPoint;
				c.focalLength = focalLength;
				c.distortion = distortion;
				cameras.push_back(c);
			}
		}
	}

	std::cout << "Obtained " << cameras.size() << " cameras." << std::endl;

	for (auto point : pt.get_child("BlocksExchange.Block.TiePoints")) {
		TiePoint tie;

		double x = point.second.get<double>("Position.x");
		double y = point.second.get<double>("Position.y");
		double z = point.second.get<double>("Position.z");

		tie.position = Point_3(x, y, z);

		for (auto measure : point.second.get_child("")) {
			if (measure.first == "Measurement") {
				std::pair<int, cv::Point2d> pair;
				int cameraId = measure.second.get<int>("PhotoId");

				for (int c = 0; c < cameras.size(); c++) {
					if (cameras[c].id == cameraId) {
						pair.first = c;
						break;
					}
				}

				pair.second.x = measure.second.get<double>("x");
				pair.second.y = measure.second.get<double>("y");

				tie.projections.insert(pair);
			}
		}

		tiePoints.push_back(tie);
	}

	for (int c = 0; c < cameras.size(); c++) {
		cameras[c].id = c;
	}

	std::cout << "Found " << tiePoints.size() << " tie points." << std::endl;
	std::cout << "Parsing of XML file done in " << (double)(std::clock() - time) / CLOCKS_PER_SEC << " seconds." << std::endl;
}