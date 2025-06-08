#include "internal_helpers/mesh_loader.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include "dgd/mesh_loader.h"

namespace dgd {

namespace internal {

void SetVertexMeshFromObjFile(const std::string& filename, MeshProperties& mp) {
  MeshLoader ml{};

  try {
    ml.LoadObj(filename);
    if (!ml.MakeVertexGraph(mp.vert, mp.vgraph)) {
      throw std::runtime_error("Qhull error: Failed to parse the file");
    }
    if ((mp.inradius = ml.ComputeInradius(mp.interior_point)) <= 0.0) {
      throw std::runtime_error("Nonpositive inradius");
    }
  } catch (const std::runtime_error& e) {
    std::cerr << "Error loading mesh from file '" << filename
              << "': " << e.what() << std::endl;
    throw;
  }
  mp.nvert = static_cast<int>(mp.vert.size());
}

void SetFacetMeshFromObjFile(const std::string& filename, MeshProperties& mp) {
  MeshLoader ml{};

  try {
    ml.LoadObj(filename);
    if (!ml.MakeFacetGraph(mp.normal, mp.offset, mp.fgraph,
                           mp.interior_point)) {
      throw std::runtime_error("Qhull error: Failed to parse the file");
    }
    if ((mp.inradius = ml.ComputeInradius(mp.normal, mp.offset,
                                          mp.interior_point)) <= 0.0) {
      throw std::runtime_error("Nonpositive inradius");
    }
  } catch (const std::runtime_error& e) {
    std::cerr << "Error loading mesh from file '" << filename
              << "': " << e.what() << std::endl;
    throw;
  }

  mp.nfacet = static_cast<int>(mp.normal.size());
}

void SetFacetMeshFromVertices(const std::vector<Vec3r>& vert,
                              MeshProperties& mp) {
  MeshLoader ml{};

  try {
    ml.ProcessPoints(vert);
    if (!ml.MakeFacetGraph(mp.normal, mp.offset, mp.fgraph,
                           mp.interior_point)) {
      throw std::runtime_error("Qhull error: Failed to process vertices");
    }
    if ((mp.inradius = ml.ComputeInradius(mp.normal, mp.offset,
                                          mp.interior_point)) <= 0.0) {
      throw std::runtime_error("Nonpositive inradius");
    }
  } catch (const std::runtime_error& e) {
    std::cerr << "Error computing h-rep from v-rep" << std::endl;
    throw;
  }

  mp.nfacet = static_cast<int>(mp.normal.size());
}

}  // namespace internal

}  // namespace dgd
