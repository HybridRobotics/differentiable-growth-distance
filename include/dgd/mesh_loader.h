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
 * @file mesh_loader.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-04-01
 * @brief 3D mesh loader class.
 */

#ifndef DGD_MESH_LOADER_H_
#define DGD_MESH_LOADER_H_

#include <cassert>
#include <cstddef>
#include <exception>
#include <functional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "dgd/data_types.h"

namespace dgd {

/**
 * @brief Class for loading 3D meshes and computing the vertex adjacency graph.
 */
class MeshLoader {
 public:
  /**
   * @brief Construct a new Mesh Loader object.
   *
   * @param maxhullvert Maximum number of mesh vertices (default = 10000).
   */
  explicit MeshLoader(int maxhullvert = 10000);

  /**
   * @brief Convert vertices to double precision and remove duplicates.
   *
   * @tparam T    Floating-point type (float or double).
   * @param  vert Vector of 3D vertex coordinates as a 1D array.
   */
  template <typename T>
  void ProcessVertices(const std::vector<T>& vert);

  /**
   * @brief Convert vertices to double precision and remove duplicates.
   *
   * @param vert Vector of 3D vertex coordinates.
   */
  void ProcessVertices(const std::vector<Vec3f>& vert);

  /**
   * @brief Load mesh object from file or parse from string.
   *
   * See
   * https://github.com/tinyobjloader/tinyobjloader/blob/release/loader_example.cc
   *
   * @param input   Mesh wavefront filename (.obj) or object string.
   * @param is_file Whether input is a filename or an object string
   *                (default = true).
   */
  void LoadOBJ(const std::string& input, bool is_file = true);

  /**
   * @brief Construct convex hull and vertex adjacency graph from stored vertex
   * list.
   *
   * graph is a vector of size (2 + 2*numvert + 3*numface) containing:
   * numvert
   *    Number of convex hull vertices.
   * numface
   *    Number of convex hull faces.
   * vert_edgeadr[numvert]
   *    For each vertex in the convex hull, this is the offset of the edge
   *    record for that vertex in edge_localid.
   * edge_localid[numvert+3*numface]
   *    This contains a sequence of edge records, one for each vertex in the
   *    convex hull. Each edge record is an array of vertex indices (in localid
   *    format) terminated with -1. For example, say the record for vertex 7 is:
   *    3, 4, 5, 9, -1. This means that vertex 7 belongs to 4 edges, and the
   *    other ends of these edges are vertices 3, 4, 5, 9. In this way every
   *    edge is represented twice, in the edge records of its two vertices.
   *    Note that for a closed triangular mesh (such as the convex hulls used
   *    here), the number of edges is 3*numface/2.
   *
   * @param[out] vert  Convex hull vertices.
   * @param[out] graph Vertex adjacency graph.
   * @return     true (success) or false (failure).
   */
  bool MakeGraph(std::vector<Vec3f>& vert, std::vector<int>& graph);

  /**
   * @brief Number of mesh vertices.
   */
  int nvert() const;

  ~MeshLoader() {};

 private:
  const int maxhullvert_; /**< Maximum number of vertices in the convex hull. */
  std::vector<double> vert_;    /**< Vertex data. */
  std::vector<float> normal_;   /**< Normal data. */
  std::vector<int> face_;       /**< Vertex indices. */
  std::vector<int> facenormal_; /**< Normal indices. */
};

inline MeshLoader::MeshLoader(int maxhullvert)
    : maxhullvert_(maxhullvert),
      vert_(0),
      normal_(0),
      face_(0),
      facenormal_(0) {}

// vertex key for hash map
template <typename T>
struct VertexKey {
  T v[3];

  bool operator==(const VertexKey<T>& other) const {
    return (v[0] == other.v[0] && v[1] == other.v[1] && v[2] == other.v[2]);
  }

  std::size_t operator()(const VertexKey<T>& vertex) const {
    // combine all three hash values into a single hash value
    return ((std::hash<T>()(vertex.v[0]) ^
             (std::hash<T>()(vertex.v[1]) << 1)) >>
            1) ^
           (std::hash<T>()(vertex.v[2]) << 1);
  }
};

// convert vertices to double precision (if needed) and remove repeated vertices
template <typename T>
void MeshLoader::ProcessVertices(const std::vector<T>& vert) {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "vertices should be of float or double types");

  vert_.clear();
  int nvert = static_cast<int>(vert.size());

  if (nvert % 3) {
    throw std::length_error("vertex data must be a multiple of 3");
  }
  if (face_.size() % 3) {
    throw std::length_error("face data must be a multiple of 3");
  }

  int index = 0;
  std::unordered_map<VertexKey<T>, int, VertexKey<T>> vertex_map;

  // populate vertex map with new vertex indices
  for (int i = 0; i < nvert; i += 3) {
    const T* v = &vert[i];

    if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2])) {
      std::string err = std::string("vertex coordinate ") + std::to_string(i) +
                        std::string(" is not finite");
      throw std::runtime_error(err);
    }

    VertexKey<T> key = {v[0], v[1], v[2]};
    if (vertex_map.find(key) == vertex_map.end()) {
      vertex_map.insert({key, index});
      ++index;
    }
  }

  // no repeated vertices (just copy vertex data)
  if (3 * index == nvert) {
    vert_.reserve(nvert);
    for (T v : vert) {
      vert_.push_back(v);
    }
    return;
  }

  // update face vertex indices
  for (int i = 0; i < static_cast<int>(face_.size()); ++i) {
    VertexKey<T> key = {vert[3 * face_[i]], vert[3 * face_[i] + 1],
                        vert[3 * face_[i] + 2]};
    face_[i] = vertex_map[key];
  }

  // repopulate vertex data
  vert_.resize(3 * index);
  for (const auto& pair : vertex_map) {
    const VertexKey<T>& key = pair.first;
    int index = pair.second;

    // double precision
    vert_[3 * index + 0] = key.v[0];
    vert_[3 * index + 1] = key.v[1];
    vert_[3 * index + 2] = key.v[2];
  }
}

void MeshLoader::ProcessVertices(const std::vector<Vec3f>& vert) {
  std::vector<Real> v(3 * vert.size());
  for (int i = 0; i < static_cast<int>(vert.size()); ++i) {
    v[3 * i] = vert[i](0);
    v[3 * i + 1] = vert[i](1);
    v[3 * i + 2] = vert[i](2);
  }
  ProcessVertices<Real>(v);
}

inline int MeshLoader::nvert() const {
  return static_cast<int>(vert_.size()) / 3;
}

}  // namespace dgd

#endif  // DGD_MESH_LOADER_H_
