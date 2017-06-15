#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <memory>
#include <ctime>

#include <glm/glm.hpp>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>
#include <CGAL/Polyhedron_items_with_id_3.h>
#include <CGAL/Polyhedron_3.h>

// A modifier creating a triangle with the incremental builder.
template <class HDS>
class Build_Obj : public CGAL::Modifier_base<HDS> {
public:
	Build_Obj(const std::string &filename) : path(filename) {}

	std::string path;

	void operator()(HDS& hds) {

		CGAL::Polyhedron_incremental_builder_3<HDS> B(hds, true);
		B.begin_surface(3, 1, 6);
		typedef typename HDS::Vertex   Vertex;
		typedef typename Vertex::Point Point;
		std::ifstream file(path);

		if (!file.good()) {
			std::cout << "File not found" << std::endl;
		}

		std::string line;
		while (std::getline(file, line)) {
			std::stringstream ss(line);
			std::string type;
			ss >> type;

			if (type == "v") {
				Point p;
				ss >> p;
				B.add_vertex(p);
			}
			if (type == "f") {
				B.begin_facet();
				int i, j, k;
				std::string vertex;

				ss >> vertex;
				std::stringstream ss2(vertex, '/');
				ss2 >> i;
				
				ss >> vertex;
				std::stringstream ss3(vertex, '/');
				ss3 >> j;

				ss >> vertex;
				std::stringstream ss4(vertex, '/');
				ss4 >> k;
				//std::cout << i << " " << j << " " << k << std::endl;
				B.add_vertex_to_facet(i-1);
				B.add_vertex_to_facet(j-1);
				B.add_vertex_to_facet(k-1);
				B.end_facet();
			}
		}
		B.end_surface();
	}
};


typedef CGAL::Simple_cartesian<double>     Kernel;
typedef CGAL::Polyhedron_3<Kernel, CGAL::Polyhedron_items_with_id_3>         Polyhedron;
typedef Polyhedron::HalfedgeDS             HalfedgeDS;
typedef CGAL::Polyhedron_incremental_builder_3<Kernel> Builder;

class ObjParser
{
public:
	ObjParser(const std::string &filename) : path(filename){};
	~ObjParser(){};

	void parse(Polyhedron &P);

private:
	std::string path;
};

bool readObjFile(const std::string &filename, std::vector<glm::vec3>& v,
	std::vector<glm::vec2>& vt,
	std::vector<std::vector<glm::ivec3> >& fvs, std::vector<std::vector<glm::ivec3> >& fvts,
	std::vector<std::string>& texFileNames);