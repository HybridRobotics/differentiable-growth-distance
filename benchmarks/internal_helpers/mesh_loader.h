#ifndef DGD_BENCHMARKS_INTERNAL_HELPERS_MESH_LOADER_H_
#define DGD_BENCHMARKS_INTERNAL_HELPERS_MESH_LOADER_H_

#include <string>
#include <vector>

#include "dgd/data_types.h"

namespace dgd {

namespace bench {

// Mesh properties.
struct MeshProperties {
  std::vector<Vec3r> vert, normal;
  std::vector<Real> offset;
  std::vector<int> vgraph, fgraph;
  Vec3r interior_point;
  Real inradius = 0.0;
  int nvert = 0;
  int nfacet = 0;
  std::string name;

  // Sets vertex mesh from .obj file.
  void SetVertexMeshFromObjFile(const std::string& filename);

  // Sets facet mesh from .obj file.
  void SetFacetMeshFromObjFile(const std::string& filename);

  // Set facet mesh from vertices.
  void SetFacetMeshFromVertices(const std::vector<Vec3r>& vert);

  // Sets the origin as the center of the vertex mesh.
  void SetZeroVertexCenter();

  // Sets the origin as the center of the facet mesh.
  void SetZeroFacetCenter();
};

inline void MeshProperties::SetZeroVertexCenter() {
  for (auto& v : vert) v -= interior_point;
}

inline void MeshProperties::SetZeroFacetCenter() {
  for (int i = 0; i < nfacet; ++i) offset[i] += normal[i].dot(interior_point);
}

}  // namespace bench

}  // namespace dgd

#endif  // DGD_BENCHMARKS_INTERNAL_HELPERS_MESH_LOADER_H_
