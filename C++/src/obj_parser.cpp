#include <boost/filesystem.hpp>
#include "obj_parser.h"

using namespace boost::filesystem;
void ObjParser::parse(Polyhedron &P) {

	std::clock_t time = clock();
	Build_Obj<HalfedgeDS> obj(path);
	P.delegate(obj);
	
	int i = 0; 
	for (auto it = P.vertices_begin(); it != P.vertices_end(); it++) {
		it->id() = i;
		i++;
	}

	int j = 0;
	for (auto it = P.facets_begin(); it != P.facets_end(); it++) {
		it->id() = j;
		j++;
	}

	std::cout << "Parsing of .obj file done in " << (double)(clock() - time) / CLOCKS_PER_SEC << " seconds." << std::endl;
	std::cout << "Mesh: " << P.size_of_vertices() << " vertices, " << P.size_of_facets() << " faces." << std::endl;

}


bool readObjFile(const std::string &filename, std::vector<glm::vec3>& v,
	std::vector<glm::vec2>& vt,
	std::vector<std::vector<glm::ivec3> >& fvs, std::vector<std::vector<glm::ivec3> >& fvts,
	std::vector<std::string>& texFileNames)
{


	std::ifstream file(filename);

	if (!file.good()) {
		std::cout << "File not found" << std::endl;
		return false;
	}

	path p(filename);
	std::string line;

	std::vector<glm::ivec3> fv;
	std::vector<glm::ivec3> fvt;
	std::string textureName = "";
	while (std::getline(file, line))
	{
		std::stringstream ss(line);
		std::string type;
		ss >> type;

		if (type == "v")
		{
			glm::vec3 vertex;
			ss >> vertex[0] >> vertex[1] >> vertex[2];
			v.push_back(vertex);
		}

		if (type == "vt")
		{
			glm::vec2 coord;
			ss >> coord[0] >> coord[1];
			vt.push_back(coord);
		}

		if (type == "usemtl")
		{
			if (!fv.empty() && !fvt.empty())
			{
				fvs.push_back(fv);
				fvts.push_back(fvt);
				texFileNames.push_back(p.parent_path().append(textureName).string());
			}

			//new information
			ss >> textureName;
			textureName.append(".jpg");//jpg extension
			fv.clear();
			fvt.clear();

		}

		if (type == "f")
		{
			glm::ivec3 f1, f2;
			std::string vertex;
			char c;

			ss >> vertex;
			std::stringstream ss2(vertex) ;
			ss2 >> f1[0] >> c >> f2[0]; //c for '/'

			ss >> vertex;
			std::stringstream ss3(vertex);
			ss3 >> f1[1] >> c >> f2[1];

			ss >> vertex;
			std::stringstream ss4(vertex);
			ss4 >> f1[2] >>  c >> f2[2];

			//because indices begin at 1, we have to subtract it
			f1 = f1 - glm::ivec3(1, 1, 1);
			f2 = f2 - glm::ivec3(1, 1, 1);

			fv.push_back(f1);
			fvt.push_back(f2);
		}
	}
	//last element if exists
	if (!fv.empty() && !fvt.empty())
	{
		fvs.push_back(fv);
		fvts.push_back(fvt);
		texFileNames.push_back(p.parent_path().append(textureName).string());
	}
}
