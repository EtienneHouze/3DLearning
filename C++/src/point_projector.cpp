#include "point_projector.h"



PointProjector::~PointProjector( )
{
}


std::vector<cv::Point2d> PointProjector::projectPoints( const std::vector<Point_3> &points,
														const Camera &c,
														const std::vector<std::vector<int>> &visibility
)
{
	std::vector<cv::Point2d> coords;

	int i = 0;
	for (Point_3 p : points) {
		std::vector<int> visible = visibility[i];
		bool skip = true;
		for (int ind : visible) {
			if (ind == c.id) {
				skip = false;
				break;
			}
		}
		if (skip) {
			i++;
			continue;
		}

		cv::Mat pTemp( 3, 1, CV_64F );
		pTemp.at<double>( 0, 0 ) = p.x( );
		pTemp.at<double>( 1, 0 ) = p.y( );
		pTemp.at<double>( 2, 0 ) = p.z( );

		cv::Mat vec = c.rotation * (pTemp - c.center);
		vec /= vec.at<double>( 2, 0 );

		cv::Point2d pt( vec.at<double>( 0, 0 ), vec.at<double>( 1, 0 ) );
		if (!c.distortion.empty( )) {
			pt = distortFunc( pt, c.distortion );
		}

		pt *= c.focalLength;
		pt += c.principalPoint;

		coords.push_back( pt );

		i++;
	}
	return coords;
}

cv::Point2d PointProjector::projectPoint( const Point_3 &p,
										  const Camera &c,
										  const double &width,
										  const double &height ) {
	cv::Mat pTemp( 3, 1, CV_64F );
	pTemp.at<double>( 0, 0 ) = p.x( );
	pTemp.at<double>( 1, 0 ) = p.y( );
	pTemp.at<double>( 2, 0 ) = p.z( );


	cv::Mat vec = c.rotation * (pTemp - c.center);
	if (vec.at<double>( 2, 0 ) < 0) {
		return cv::Point2d( 0, 0 );
	}
	vec /= vec.at<double>( 2, 0 );
	cv::Point2d pt( vec.at<double>( 0, 0 ), vec.at<double>( 1, 0 ) );

	if (pt.x > -width && pt.x < width && pt.y > -height && pt.y < height) {
		pt = distortFunc( pt, c.distortion );
	}
	else {
		return cv::Point2d( 0, 0 );
	}


	pt *= c.focalLength;
	pt += c.principalPoint;

	return pt;
}


void PointProjector::computeAndSaveAllProjections( const std::vector<Point_3> &points, const std::vector<Camera> &cameras, const std::vector<std::vector<int>> &visibility, std::string &path, std::vector<std::vector<std::pair<int, cv::Point2d>>> &projections, std::vector<std::vector<double>> &positions ) {
	std::ofstream outputFile( path.c_str( ) );

	outputFile << cameras.size( ) << "\n";
	for (int i = 0; i < cameras.size( ); i++) {
		outputFile << cameras[i].filename << " " << cameras[i].id << "\n";
	}

	std::vector<double> widths( cameras.size( ) );
	std::vector<double> heights( cameras.size( ) );

	for (int c = 0; c < cameras.size( ); c++) {
		widths[c] = (double)(cameras[c].width) / (cameras[c].focalLength);
		heights[c] = (double)(cameras[c].height) / (cameras[c].focalLength);
	}

	std::vector<std::vector<std::pair<int, cv::Point2d>>> tempProjections( points.size( ) );
	std::vector<std::vector<double>> tempCoords( points.size( ) );
	/*
		#pragma omp parallel for*/
	for (int i = 0; i < points.size( ); i++) {
		std::vector<double> coord_3d( 3 );  //Contient les coordonnées 3D du point en cours
		coord_3d[0] = points[i].x( );
		coord_3d[1] = points[i].y( );
		coord_3d[2] = points[i].z( );
		for (int j = 0; j < visibility[i].size( ); j++) {
			int idx = visibility[i][j];
			Camera c = cameras[idx];
			cv::Point2d coord = projectPoint( points[i], c, widths[idx], heights[idx] );
			if (coord.x == 0 && coord.y == 0) {
				continue;
			}
			if (coord.x >= 0.0 && coord.x < (c.width - 1) && coord.y >= 0.0 && coord.y < (c.height - 1)) {
				tempProjections[i].push_back( std::pair<int, cv::Point2d>( c.id, coord ) );
				tempCoords[i] = coord_3d;					// On stocke dans le vecteur coord_3d qui contiendra donc toutes les coordonnées 3d de tous les points.
			}
		}
	}

	//std::vector<std::vector<std::pair<int, cv::Point2d>>> projections;

	for (int i = 0; i < tempProjections.size( ); i++) {
		if (tempProjections[i].size( ) > 0) {
			projections.push_back( tempProjections[i] );
			positions.push_back( tempCoords[i] );
		}
	}

	outputFile << projections.size( ) << std::endl;
	for (int i = 0; i < projections.size( ); i++) {
		outputFile << positions[i][0] << " " << positions[i][1] << " " << positions[i][2] << " ";
		for (int j = 0; j < projections[i].size( ); j++) {
			outputFile << projections[i][j].first << " " << projections[i][j].second.x << " " << projections[i][j].second.y << " ";
		}
		if (i != projections.size( ) - 1) {
			outputFile << "\n";
		}
	}
}

cv::Point2d PointProjector::distortFunc( cv::Point2d &pt, const cv::Mat& distortion ) {

	double sqrRadius = (pt.x * pt.x + pt.y * pt.y);
	double coeff1 = 1.0 + sqrRadius * (distortion.at<double>( 0, 0 ) + sqrRadius * (distortion.at<double>( 1, 0 ) + sqrRadius * distortion.at<double>( 4, 0 )));

	cv::Point2d newPt = cv::Point2d( coeff1 * pt.x
									 + 2 * distortion.at<double>( 3, 0 ) * pt.x * pt.y
									 + distortion.at<double>( 2, 0 ) * (sqrRadius + 2 * pt.x * pt.x),
									 coeff1 * pt.y
									 + 2 * distortion.at<double>( 2, 0 ) * pt.x * pt.y
									 + distortion.at<double>( 3, 0 ) * (sqrRadius + 2 * pt.y * pt.y) );

	return newPt;

}
