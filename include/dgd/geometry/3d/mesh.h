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
 * @file mesh.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-03-30
 * @brief 3D mesh class.
 */

#ifndef DGD_GEOMETRY_3D_MESH_H_
#define DGD_GEOMETRY_3D_MESH_H_

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/**
 * @brief 3D convex mesh class using hill-climbing for support function
 * computation.
 */
class Mesh : public ConvexSet<3> {
 public:
  /**
   * @brief Constructs a convex Mesh object.
   *
   * @attention The user must ensure that the mesh contains the origin in its
   * interior and that the inradius is correct.
   *
   * @see MeshLoader::MakeVertexGraph
   *
   * @param vert        Vertices of the mesh convex hull.
   * @param graph       Vertex adjacency graph of the mesh convex hull.
   * @param margin      Safety margin.
   * @param inradius    Polytope inradius.
   * @param thresh      (advanced) Support function threshold (default = 0.9).
   * @param guess_level (advanced) Guess level for the warm start index
   *                    (default = 2).
   */
  Mesh(const std::vector<Vec3f>& vert, const std::vector<int>& graph,
       Real margin, Real inradius, Real thresh = 0.9, int guess_level = 2);

  ~Mesh() {};

  Real SupportFunction(const Vec3f& n, Vec3f& sp) final;

  /**
   * @brief Gets the vertices of the mesh convex hull.
   *
   * @return Convex hull vertices.
   */
  const std::vector<Vec3f>& GetVertices() const;

  /**
   * @brief Gets the vertex adjacency graph of the mesh convex hull.
   *
   * @see MeshLoader::MakeVertexGraph
   *
   * @return Vertex adjacency graph.
   */
  const std::vector<int>& GetGraph() const;

  /**
   * @brief Returns the number of vertices in the mesh convex hull.
   *
   * @return Number of vertices.
   */
  int nvert() const;

 private:
  std::vector<Vec3f> vert_; /**< Convex hull vertices. */
  std::vector<int> graph_;  /**< Vertex adjacency graph. */
  const Real margin_;       /**< Safety margin. */
  const int nvert_;         /**< Number of vertices. */

  std::vector<int>::const_iterator vert_edgeadr_;
  std::vector<int>::const_iterator edge_localid_;
  std::vector<int> idx_ws0_;  // Initial guesses for idx_ws_.
  Vec3f n_prev_;              // Previous normal vector.
  const Real thresh_;         // Support function warm start threshold.
  const int guess_level_;     // Support function warm start level.
  int idx_ws_;                // Warm start index for support function.
};

inline Mesh::Mesh(const std::vector<Vec3f>& vert, const std::vector<int>& graph,
                  Real margin, Real inradius, Real thresh, int guess_level)
    : ConvexSet<3>(margin + inradius),
      vert_(vert),
      graph_(graph),
      margin_(margin),
      nvert_(static_cast<int>(vert.size())),
      thresh_(thresh),
      guess_level_(guess_level) {
  if ((inradius <= 0.0) || (margin < 0.0))
    throw std::domain_error("Invalid inradius or margin");
  if ((guess_level < 0) || (guess_level > 2))
    throw std::domain_error("Guess level is not 0, 1, or 2");

  if (vert.empty() || graph.size() < 2 || nvert_ != graph[0] ||
      graph.size() != 2 + 2 * nvert_ + 3 * graph[1])
    throw std::domain_error("Invalid graph or vertex set");

  vert_edgeadr_ = graph_.begin() + 2;
  edge_localid_ = vert_edgeadr_ + nvert_;

  n_prev_ = Vec3f::Zero();
  idx_ws_ = -1;

  // Set idx_ws0_ indices.
  //  Select (0, 8, 20) uniformly distributed normal vectors.
  std::vector<Vec3f> normals;
  Vec3f n;
  Real f[2]{1.0, -1.0};
  if (guess_level > 0) {
    // Add cube vertices.
    for (int i = 0; i < 8; ++i) {
      n = Vec3f(f[i % 2], f[(i / 2) % 2], f[(i / 4) % 2]);
      normals.push_back(n.normalized());
    }
  }
  if (guess_level > 1) {
    const Real gr_inv{Real(2.0 / (1.0 + std::sqrt(5.0)))};
    const Real ha{Real(1.0) + gr_inv};
    const Real hb{Real(1.0) - gr_inv * gr_inv};
    // Add regular dodecahedron vertices.
    // See https://en.wikipedia.org/wiki/Dodecahedron#Cartesian_coordinates
    for (int i = 0; i < 4; ++i) {
      n = Vec3f(0.0, f[i % 2] * ha, f[(i / 2) % 2] * hb);
      normals.push_back(n.normalized());
      n = Vec3f(f[i % 2] * ha, f[(i / 2) % 2] * hb, 0.0);
      normals.push_back(n.normalized());
      n = Vec3f(f[i % 2] * hb, 0.0, f[(i / 2) % 2] * ha);
      normals.push_back(n.normalized());
    }
  }

  //  Compute support function to set idx_ws0_.
  Vec3f sp;
  for (const auto& n : normals) {
    SupportFunction(n, sp);
    idx_ws0_.push_back(idx_ws_);
  }
}

inline Real Mesh::SupportFunction(const Vec3f& n, Vec3f& sp) {
  // If the current normal is much different than the previous normal,
  // compute a new warm start index.
  if (n_prev_.dot(n) < thresh_) {
    if (idx_ws0_.empty())
      idx_ws_ = 0;
    else {
      Real s{0.0}, smax{-kInf};
      for (int i : idx_ws0_)
        if ((s = n.dot(vert_[i])) > smax) {
          idx_ws_ = i;
          smax = s;
        }
    }
  }

  // Current best index, neighbour index, previous best index.
  int idx{idx_ws_}, nidx{-1}, pidx{-1};
  // Current support value, current best support value.
  Real s{0.0}, sv{n.dot(vert_[idx])};

  // Hill-climbing.
  do {
    pidx = idx;
    for (int i = *(vert_edgeadr_ + idx); (nidx = *(edge_localid_ + i)) > -1;
         ++i) {
      s = n.dot(vert_[nidx]);
      if (s > sv) {
        idx = nidx;
        sv = s;
      }
    }
  } while (idx != pidx);

  n_prev_ = n;
  idx_ws_ = idx;

  sp = vert_[idx] + margin_ * n;
  return sv + margin_;
}

inline const std::vector<Vec3f>& Mesh::GetVertices() const { return vert_; }

inline const std::vector<int>& Mesh::GetGraph() const { return graph_; }

inline int Mesh::nvert() const { return nvert_; }

}  // namespace dgd

#endif  // DGD_GEOMETRY_3D_MESH_H_
