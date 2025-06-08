#ifndef DGD_BENCHMARKS_INTERNAL_HELPERS_MESH_LOADER_H_
#define DGD_BENCHMARKS_INTERNAL_HELPERS_MESH_LOADER_H_

#include <string>
#include <vector>

#include "dgd/data_types.h"

namespace dgd {

namespace internal {

// Mesh properties.
struct MeshProperties {
  std::vector<Vec3r> vert, normal;
  std::vector<Real> offset;
  std::vector<int> vgraph, fgraph;
  Vec3r interior_point;
  Real inradius = 0.0;
  int nvert = 0;
  int nfacet = 0;
};

// Sets vertex and facets meshes from .obj file.
void SetVertexMeshFromObjFile(const std::string& filename, MeshProperties& mp);

void SetFacetMeshFromObjFile(const std::string& filename, MeshProperties& mp);

// Set facet mesh from vertices.
void SetFacetMeshFromVertices(const std::vector<Vec3r>& vert,
                              MeshProperties& mp);

// Sets the center of the vertex mesh as the origin.
inline void SetZeroVertexCenter(MeshProperties& mp) {
  for (auto& v : mp.vert) {
    v -= mp.interior_point;
  }
}

// Sets the center of the facet mesh as the origin.
inline void SetZeroFacetCenter(MeshProperties& mp) {
  for (int i = 0; i < mp.nfacet; ++i) {
    mp.offset[i] += mp.normal[i].dot(mp.interior_point);
  }
}

}  // namespace internal

}  // namespace dgd

#endif  // DGD_BENCHMARKS_INTERNAL_HELPERS_MESH_LOADER_H_
