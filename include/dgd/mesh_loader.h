// Copyright 2025 Akshay Thirugnanam
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
 * @brief Mesh loader class.
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

class MeshLoader {
 public:
  explicit MeshLoader(int maxhullvert = 10000);

  template <typename T>
  void ProcessVertices(const std::vector<T>& vert);

  void ProcessVertices(const std::vector<Vec3f>& vert);

  void LoadOBJ(const std::string& input, bool is_file = true);

  bool MakeGraph(std::vector<Vec3f>& vert, std::vector<int>& graph);

  int nvert() const;

  ~MeshLoader() {};

 private:
  const int maxhullvert_;      // maximum number of vertices in the convex hull
  std::vector<double> vert_;   // vertex data
  std::vector<float> normal_;  // normal data
  std::vector<int> face_;      // vertex indices
  std::vector<int> facenormal_;  // normal indices
};

inline MeshLoader::MeshLoader(int maxhullvert)
    : maxhullvert_(maxhullvert),
      vert_(0),
      normal_(0),
      face_(0),
      facenormal_(0) {}

/**
 * The implementation below is adapted from MuJoCo:
 * https://github.com/google-deepmind/mujoco/blob/main/src/user/user_mesh.cc
 */

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
      throw std::overflow_error(err);
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
