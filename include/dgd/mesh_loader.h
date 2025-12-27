// Copyright 2021 DeepMind Technologies Limited
// Copyright 2025 Akshay Thirugnanam
//
// Copied and adapted from MuJoCo
// (https://github.com/google-deepmind/mujoco/blob/main/src/user/user_mesh.cc)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @brief 3D mesh loader class.
 */

#ifndef DGD_MESH_LOADER_H_
#define DGD_MESH_LOADER_H_

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "dgd/data_types.h"

namespace dgd {

/**
 * @brief Class for loading 3D meshes and computing the vertex and facet
 * adjacency graphs of the convex hull.
 */
class MeshLoader {
 public:
  /**
   * @brief Constructs a new Mesh Loader object.
   *
   * @param maxhullvert Maximum number of convex hull vertices.
   */
  explicit MeshLoader(int maxhullvert = 10000);

  /**
   * @brief Loads a mesh object from file or parses from object string.
   *
   * @note This function internally calls ProcessPoints.
   *
   * See
   * https://github.com/tinyobjloader/tinyobjloader/blob/release/loader_example.cc
   *
   * @param input   Mesh wavefront filename (*.obj) or object string.
   * @param is_file Whether input is a filename or an object string.
   */
  void LoadObj(const std::string& input, bool is_file = true);

  /**
   * @brief Converts points to double precision and removes duplicates.
   *
   * T can be float, double, or Vec3r.
   *
   * @tparam T   Floating-point or 3D vector type.
   * @param  pts Vector of 3D point coordinates.
   */
  template <typename T>
  void ProcessPoints(const std::vector<T>& pts);

  /**
   * @brief Constructs convex hull (in V-rep) and vertex adjacency graph from
   * the stored vector of points.
   *
   * graph is a vector of size (2 + 2*nvert + 3*nface) containing (in order):
   * nvert: int
   *    Number of convex hull vertices.
   * nface: int
   *    Number of triangulated convex hull faces.
   * vert_edgeadr: int[nvert]
   *    For each vertex in the convex hull, this is the offset of the edge
   *    record for that vertex in edge_localid (starts from zero).
   * edge_localid: int[nvert + 3*nface]
   *    This contains a sequence of edge records, one for each vertex in the
   *    convex hull. Each edge record is an array of vertex indices terminated
   *    with -1. For example, say the record for vertex 7 is: 3, 4, 5, 9, -1.
   *    This means that vertex 7 belongs to 4 edges, and the other ends of
   *    these edges are vertices 3, 4, 5, 9. In this way every edge is
   *    represented twice, in the edge records of its two vertices. Note that
   *    for a closed triangular mesh (such as the convex hulls used here), the
   *    number of edges is 3*nface/2.
   *
   * @param[out] vert  Convex hull vertices.
   * @param[out] graph Vertex adjacency graph.
   * @return     true (success) or false (failure).
   */
  bool MakeVertexGraph(std::vector<Vec3r>& vert, std::vector<int>& graph);

  /**
   * @brief Constructs convex hull (in H-rep) and facet adjacency graph from
   * the stored vector of points. Also returns an interior point.
   *
   * The convex hull is given by the set of inequalities:
   *    normal[i] * z + offset[i] <= 0    for 0 <= i < nfacet,
   * where normal[i] has unit 2-norm.
   *
   * graph is a vector of size (2 + 2*nfacet + 2*nridge) containing (in order):
   * nfacet: int
   *    Number of convex hull facets.
   * nridge: int
   *    Number of convex hull ridges (facet edges).
   * facet_ridgeadr: int[nfacet]
   *    For each facet in the convex hull, this is the offset of the ridge
   *    record for that facet in ridge_localid (starts from zero).
   * ridge_localid: int[nfacet + 2*nridge]
   *    This contains a sequence of ridge records, one for each facet in the
   *    convex hull. Each ridge record is an array of facet indices terminated
   *    with -1. For example, say the record for facet 7 is: 3, 4, 5, 9, -1.
   *    This means that facet 7 belongs to 4 ridges, and the other ends of
   *    these ridges are facets 3, 4, 5, 9 (in CCW order). In this way every
   *    ridge is represented twice, in the ridge records of its two facets.
   *    Note that for a polytope, the nfacet + nvert = nridge + 2.
   *
   * @param[out] normal         Facet normals of the convex hull.
   * @param[out] offset         Facet offsets of the convex hull.
   * @param[out] graph          Facet adjacency graph.
   * @param[out] interior_point A point in the convex hull interior.
   * @return     true (success) or false (failure).
   */
  bool MakeFacetGraph(std::vector<Vec3r>& normal, std::vector<Real>& offset,
                      std::vector<int>& graph, Vec3r& interior_point);

  /**
   * @brief Computes the inradius of a polytope at an interior point, given its
   * H-rep and the interior point.
   *
   * @param  normal         Facet normals of the convex hull.
   * @param  offset         Facet offsets of the convex hull.
   * @param  interior_point A point in the convex hull interior.
   * @return Inradius of the polytope at the interior point.
   */
  Real ComputeInradius(const std::vector<Vec3r>& normal,
                       const std::vector<Real>& offset,
                       const Vec3r& interior_point) const;

  /**
   * @brief Computes an interior point and the inradius (at the interior point)
   * for the stored vector of points.
   *
   * @note Use this function if normals and offsets are not available. This
   * function internally calls MakeFacetGraph.
   *
   * @param[in,out] interior_point A point in the convex hull interior.
   * @param         use_given_ip   Whether to compute the inradius at
   *                               interior_point or to compute a new interior
   *                               point.
   * @return        Inradius at the interior point.
   */
  Real ComputeInradius(Vec3r& interior_point, bool use_given_ip = false);

  /// @brief Number of points in the mesh.
  int npts() const;

  ~MeshLoader() = default;

 private:
  const int maxhullvert_;       /**< Maximum number of convex hull vertices. */
  std::vector<double> pts_;     /**< Mesh point data. */
  std::vector<float> normal_;   /**< Mesh normal vector data. */
  std::vector<int> face_;       /**< Mesh face vertex indices. */
  std::vector<int> facenormal_; /**< Mesh face normal vector indices. */
};

inline MeshLoader::MeshLoader(int maxhullvert)
    : maxhullvert_(maxhullvert),
      pts_(0),
      normal_(0),
      face_(0),
      facenormal_(0) {}

namespace detail {

/// @brief Point key struct for hash map.
template <typename T>
struct PointKey {
  T p[3];

  bool IsFinite() const {
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
  }

  bool operator==(const PointKey<T>& other) const {
    return (p[0] == other.p[0] && p[1] == other.p[1] && p[2] == other.p[2]);
  }

  std::size_t operator()(const PointKey<T>& ptk) const {
    // Combine all three hash values into a single hash value.
    return ((std::hash<T>()(ptk.p[0]) ^ (std::hash<T>()(ptk.p[1]) << 1)) >> 1) ^
           (std::hash<T>()(ptk.p[2]) << 1);
  }
};

/// @brief Scalar extractor for point types.
template <typename T, typename = void>
struct ScalarExtractor {
  using type = T;
};

/// @brief Scalar extractor specialization for Eigen types.
template <typename T>
struct ScalarExtractor<T, std::void_t<typename T::Scalar>> {
  using type = typename T::Scalar;
};

}  // namespace detail

template <typename T>
void MeshLoader::ProcessPoints(const std::vector<T>& pts) {
  static_assert(std::is_same<T, float>::value ||
                    std::is_same<T, double>::value ||
                    std::is_same<T, Vec3r>::value,
                "Points should be of float or double types");
  using R = typename detail::ScalarExtractor<T>::type;

  constexpr bool is_T_Vec3r = std::is_same<T, Vec3r>::value;
  auto pptr = [&pts](int i) -> const R* {
    if constexpr (is_T_Vec3r) {
      return &pts[i](0);
    } else {
      return &pts[3 * i];
    }
  };

  pts_.clear();
  int npts = static_cast<int>(pts.size());
  if constexpr (is_T_Vec3r) npts *= 3;

  if (npts % 3) {
    throw std::length_error("Point data must be a multiple of 3");
  }
  if (face_.size() % 3) {
    throw std::length_error("Face data must be a multiple of 3");
  }

  int idx = 0;
  std::unordered_map<detail::PointKey<R>, int, detail::PointKey<R>> point_map;

  // Populate point map with new point indices.
  for (int i = 0; i < (npts / 3); ++i) {
    const R* p = pptr(i);
    detail::PointKey<R> key = {p[0], p[1], p[2]};

    if (!key.IsFinite()) {
      throw std::runtime_error("Point coordinate " + std::to_string(i) +
                               " is not finite");
    }

    if (point_map.find(key) == point_map.end()) {
      point_map.insert({key, idx});
      ++idx;
    }
  }

  if (3 * idx == npts) {
    // No repeated points (just copy point data).
    pts_.reserve(npts);
    for (int i = 0; i < (npts / 3); ++i) {
      const R* p = pptr(i);
      for (int j = 0; j < 3; ++j) pts_.push_back(p[j]);
    }
    return;
  }

  // Update face point indices.
  for (int i = 0; i < static_cast<int>(face_.size()); ++i) {
    const R* p = pptr(face_[i]);
    detail::PointKey<R> key = {p[0], p[1], p[2]};
    face_[i] = point_map[key];
  }

  // Repopulate point data.
  pts_.resize(3 * idx);
  for (const auto& pair : point_map) {
    const detail::PointKey<R>& key = pair.first;
    const int idx = pair.second;
    // Double precision.
    for (int j = 0; j < 3; ++j) pts_[3 * idx + j] = key.p[j];
  }
}

inline int MeshLoader::npts() const {
  return static_cast<int>(pts_.size() / 3);
}

}  // namespace dgd

#endif  // DGD_MESH_LOADER_H_
