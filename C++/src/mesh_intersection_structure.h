#pragma once

#include <iostream>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/boost/graph/graph_traits_Polyhedron_3.h>
#include <CGAL/Polyhedron_items_with_id_3.h>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef Kernel::Plane_3 Plane;
typedef Kernel::Vector_3 Vector_3;
typedef Kernel::Ray_3 Ray;
typedef Kernel::Segment_3 Segment;
typedef CGAL::Polyhedron_3<Kernel, CGAL::Polyhedron_items_with_id_3> Polyhedron;
typedef CGAL::AABB_face_graph_triangle_primitive<Polyhedron> Primitive;
typedef CGAL::AABB_traits<Kernel, Primitive> Traits;
typedef CGAL::AABB_tree<Traits> Tree;
typedef boost::optional< Tree::Intersection_and_primitive_id<Ray>::Type > Ray_intersection;

class MeshIntersectionStructure
{
public:
	MeshIntersectionStructure(Polyhedron &P) : mesh(P) {
		tree = new Tree(faces(P).first, faces(P).second, P);
	};

	~MeshIntersectionStructure() {
		delete tree;
	};

	bool intersect(const Ray &r);
	bool intersect(const Segment &s);
	std::vector<Ray_intersection> all_intersections(const Ray &r);
	
private:
	Polyhedron mesh;
	Tree* tree;
};

