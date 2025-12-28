#include "internal_helpers/mesh_loader.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include "dgd/mesh_loader.h"

namespace dgd {

namespace bench {

void MeshProperties::SetVertexMeshFromObjFile(const std::string& filename) {
  MeshLoader ml{};

  try {
    ml.LoadObj(filename);
    if (!ml.MakeVertexGraph(vert, vgraph)) {
      throw std::runtime_error("Qhull error: Failed to parse the file");
    }
    inradius = ml.ComputeInradius(interior_point);
    if (inradius <= 0.0) {
      throw std::runtime_error("Nonpositive inradius");
    }
  } catch (const std::runtime_error& e) {
    std::cerr << "Error loading mesh from file '" << filename
              << "': " << e.what() << std::endl;
    throw;
  }
  nvert = static_cast<int>(vert.size());
  name = filename;
}

void MeshProperties::SetFacetMeshFromObjFile(const std::string& filename) {
  MeshLoader ml{};

  try {
    ml.LoadObj(filename);
    if (!ml.MakeFacetGraph(normal, offset, fgraph, interior_point)) {
      throw std::runtime_error("Qhull error: Failed to parse the file");
    }
    inradius = ml.ComputeInradius(normal, offset, interior_point);
    if (inradius <= 0.0) {
      throw std::runtime_error("Nonpositive inradius");
    }
  } catch (const std::runtime_error& e) {
    std::cerr << "Error loading mesh from file '" << filename
              << "': " << e.what() << std::endl;
    throw;
  }
  nfacet = static_cast<int>(normal.size());
  name = filename;
}

void MeshProperties::SetFacetMeshFromVertices(const std::vector<Vec3r>& vert) {
  MeshLoader ml{};

  try {
    ml.ProcessPoints(vert);
    if (!ml.MakeFacetGraph(normal, offset, fgraph, interior_point)) {
      throw std::runtime_error("Qhull error: Failed to process vertices");
    }
    inradius = ml.ComputeInradius(normal, offset, interior_point);
    if (inradius <= 0.0) {
      throw std::runtime_error("Nonpositive inradius");
    }
  } catch (const std::runtime_error& e) {
    std::cerr << "Error computing h-rep from v-rep" << std::endl;
    throw;
  }
  nfacet = static_cast<int>(normal.size());
}

}  // namespace bench

}  // namespace dgd
