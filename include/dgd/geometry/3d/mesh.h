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
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @brief 3D mesh class.
 */

#ifndef DGD_GEOMETRY_3D_MESH_H_
#define DGD_GEOMETRY_3D_MESH_H_

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
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
   * @attention The mesh must contain the origin in its interior, and the
   * inradius must be accurate.
   *
   * @see MeshLoader::MakeVertexGraph
   * @see MeshLoader::ComputeInradius
   *
   * @param vert        Vertices of the mesh convex hull.
   * @param graph       Vertex adjacency graph of the mesh convex hull.
   * @param inradius    Polytope inradius.
   * @param margin      Safety margin.
   * @param thresh      Support function threshold.
   * @param guess_level Guess level for the warm start index.
   */
  explicit Mesh(const std::vector<Vec3r>& vert, const std::vector<int>& graph,
                Real inradius, Real margin = Real(0.0), Real thresh = Real(0.9),
                int guess_level = 1, const std::string& name = "__Mesh__");

  ~Mesh() = default;

  Mesh(const Mesh&) = delete;
  Mesh& operator=(const Mesh&) = delete;
  Mesh(Mesh&&) noexcept = delete;
  Mesh& operator=(Mesh&&) noexcept = delete;

  Real SupportFunction(
      const Vec3r& n, Vec3r& sp,
      SupportFunctionHint<3>* hint = nullptr) const final override;

  Real SupportFunction(
      const Vec3r& n, SupportFunctionDerivatives<3>& deriv,
      SupportFunctionHint<3>* hint = nullptr) const final override;

  bool RequireUnitNormal() const final override;

  bool IsPolytopic() const final override;

  void PrintInfo() const final override;

  const std::vector<Vec3r>& vertices() const;

  /**
   * @brief Gets the vertex adjacency graph of the mesh convex hull.
   *
   * @see MeshLoader::MakeVertexGraph
   */
  const std::vector<int>& graph() const;

  int nvertices() const;

 private:
  std::vector<Vec3r> vert_; /**< Convex hull vertices. */
  std::vector<int> graph_;  /**< Vertex adjacency graph. */
  std::vector<int>::const_iterator vert_edgeadr_;
  std::vector<int>::const_iterator edge_localid_;

  const Real margin_;        /**< Safety margin. */
  const Real thresh_;        /**< Support function warm start threshold. */
  std::vector<int> idx_ws0_; /**< Initial guesses for idx_ws_. */
  const int nvert_;          /**< Number of vertices. */

  std::string name_; /**< Mesh name. */
};

inline Mesh::Mesh(const std::vector<Vec3r>& vert, const std::vector<int>& graph,
                  Real inradius, Real margin, Real thresh, int guess_level,
                  const std::string& name)
    : ConvexSet<3>(margin + inradius),
      vert_(vert),
      graph_(graph),
      margin_(margin),
      thresh_(thresh),
      nvert_(static_cast<int>(vert.size())),
      name_(name) {
  if ((inradius <= Real(0.0)) || (margin < Real(0.0))) {
    throw std::domain_error("Invalid inradius or margin");
  }
  if ((guess_level < 0) || (guess_level > 2)) {
    throw std::domain_error("Guess level is not 0, 1, or 2");
  }

  if (vert.empty() || graph.size() < 2 || nvert_ != graph[0] ||
      static_cast<int>(graph.size()) != 2 + 2 * nvert_ + 3 * graph[1]) {
    throw std::domain_error("Invalid graph or vertex set");
  }

  vert_edgeadr_ = graph_.begin() + 2;
  edge_localid_ = vert_edgeadr_ + nvert_;

  // Set idx_ws0_ indices.
  //  Select (0, 8, 20) uniformly distributed normal vectors.
  std::vector<Vec3r> normals;
  Vec3r n;
  Real f[2] = {Real(1.0), Real(-1.0)};
  if (guess_level > 0) {
    // Add cube vertices.
    for (int i = 0; i < 8; ++i) {
      n = Vec3r(f[i % 2], f[(i / 2) % 2], f[(i / 4) % 2]);
      normals.push_back(n.normalized());
    }
  }
  if (guess_level > 1) {
    // Inverse of the golden ratio.
    const Real gr_inv = Real(2.0 / (1.0 + std::sqrt(5.0)));
    const Real ha = Real(1.0) + gr_inv;
    const Real hb = Real(1.0) - gr_inv * gr_inv;
    // Add regular dodecahedron vertices.
    // See https://en.wikipedia.org/wiki/Dodecahedron#Cartesian_coordinates
    for (int i = 0; i < 4; ++i) {
      n = Vec3r(Real(0.0), f[i % 2] * ha, f[(i / 2) % 2] * hb);
      normals.push_back(n.normalized());
      n = Vec3r(f[i % 2] * ha, f[(i / 2) % 2] * hb, Real(0.0));
      normals.push_back(n.normalized());
      n = Vec3r(f[i % 2] * hb, Real(0.0), f[(i / 2) % 2] * ha);
      normals.push_back(n.normalized());
    }
  }

  //  Compute support function to set idx_ws0_.
  Vec3r sp;
  for (const auto& n : normals) {
    SupportFunctionHint<3> hint{};
    SupportFunction(n, sp, &hint);
    idx_ws0_.push_back(hint.idx_ws);
  }
}

inline Real Mesh::SupportFunction(const Vec3r& n, Vec3r& sp,
                                  SupportFunctionHint<3>* hint) const {
  // If the current normal is much different than the previous normal,
  // compute a new warm start index.
  int idx_ws = hint ? hint->idx_ws : 0;
  if (hint && hint->n_prev.dot(n) < thresh_) {
    if (idx_ws0_.empty()) {
      idx_ws = 0;
    } else {
      Real s = Real(0.0), smax = -kInf;
      for (int i : idx_ws0_) {
        if ((s = n.dot(vert_[i])) > smax) {
          idx_ws = i;
          smax = s;
        }
      }
    }
  }

  assert(idx_ws >= 0);
  // Current best index, neighbour index, previous best index.
  int idx = idx_ws, nidx = -1, pidx = -1;
  // Current support value, current best support value.
  Real s = Real(0.0), sv = n.dot(vert_[idx]);

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

  if (hint) {
    hint->n_prev = n;
    hint->idx_ws = idx;
  }

  sp = vert_[idx] + margin_ * n;
  return sv + margin_;
}

inline Real Mesh::SupportFunction(const Vec3r& n,
                                  SupportFunctionDerivatives<3>& deriv,
                                  SupportFunctionHint<3>* hint) const {
  const Real sv = SupportFunction(n, deriv.sp, hint);

  deriv.differentiable = true;
  const int idx = hint->idx_ws;
  int nidx = -1;
  for (int i = *(vert_edgeadr_ + idx); (nidx = *(edge_localid_ + i)) > -1;
       ++i) {
    if (n.dot(vert_[nidx]) > sv - eps_diff()) {
      deriv.differentiable = false;
      break;
    }
  }
  if (deriv.differentiable) {
    deriv.Dsp = margin_ * (Matr<3, 3>::Identity() - n * n.transpose());
  }
  return sv;
}

inline bool Mesh::RequireUnitNormal() const { return (margin_ > Real(0.0)); }

inline bool Mesh::IsPolytopic() const { return (margin_ == Real(0.0)); }

inline void Mesh::PrintInfo() const {
  std::cout << "Type: Mesh (dim = 3)" << std::endl
            << "  Name: " << name_ << std::endl
            << "  #Vertices: " << vert_.size() << std::endl
            << "  Inradius: " << inradius_ << std::endl
            << "  Margin: " << margin_ << std::endl;
}

inline const std::vector<Vec3r>& Mesh::vertices() const { return vert_; }

inline const std::vector<int>& Mesh::graph() const { return graph_; }

inline int Mesh::nvertices() const { return nvert_; }

}  // namespace dgd

#endif  // DGD_GEOMETRY_3D_MESH_H_
