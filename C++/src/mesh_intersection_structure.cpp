#include "mesh_intersection_structure.h"


bool MeshIntersectionStructure::intersect(const Ray &r) {
	return tree->do_intersect(r);
}

bool MeshIntersectionStructure::intersect(const Segment &s) {
	return tree->do_intersect(s);
}

std::vector<Ray_intersection> MeshIntersectionStructure::all_intersections(const Ray &r) {
	std::vector<Ray_intersection> intersections;
	tree->all_intersections(r, std::back_inserter(intersections));

	return intersections;
}