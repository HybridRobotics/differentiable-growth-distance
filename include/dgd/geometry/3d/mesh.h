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

#include <Eigen/Geometry>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/// @brief Hill-climbing type for the Mesh support function.
enum class HillClimbingType {
  Greedy, /**< Greedy ascent. */

  Optimal, /**< Optimal ascent. */
};

/**
 * @brief 3D convex mesh class using hill-climbing for support function
 * computation.
 *
 * @tparam HCT Hill-climbing type for the Mesh support function.
 */
template <HillClimbingType HCT>
class MeshImpl : public ConvexSet<3> {
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
  explicit MeshImpl(std::vector<Vec3r> vert, std::vector<int> graph,
                    Real inradius, Real margin = Real(0.0),
                    Real thresh = Real(0.9), int guess_level = 1,
                    const std::string& name = "__Mesh__");

  ~MeshImpl() = default;

  MeshImpl(const MeshImpl&) = delete;
  MeshImpl& operator=(const MeshImpl&) = delete;
  MeshImpl(MeshImpl&&) noexcept = delete;
  MeshImpl& operator=(MeshImpl&&) noexcept = delete;

  Real SupportFunction(
      const Vec3r& n, Vec3r& sp,
      SupportFunctionHint<3>* hint = nullptr) const final override;

  Real SupportFunction(
      const Vec3r& n, SupportFunctionDerivatives<3>& deriv,
      SupportFunctionHint<3>* hint = nullptr) const final override;

  bool RequireUnitNormal() const final override;

  void ComputeLocalGeometry(
      const NormalPair<3>& zn, SupportPatchHull<3>& sph, NormalConeSpan<3>& ncs,
      const BasePointHint<3>* hint = nullptr) const final override;

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

template <HillClimbingType HCT>
inline MeshImpl<HCT>::MeshImpl(std::vector<Vec3r> vert, std::vector<int> graph,
                               Real inradius, Real margin, Real thresh,
                               int guess_level, const std::string& name)
    : ConvexSet<3>(margin + Real(0.99) * inradius),
      vert_(std::move(vert)),
      graph_(std::move(graph)),
      margin_(margin),
      thresh_(thresh),
      nvert_(static_cast<int>(vert_.size())),
      name_(name) {
  if ((inradius <= Real(0.0)) || (margin < Real(0.0))) {
    throw std::domain_error("Invalid inradius or margin");
  }
  if ((guess_level < 0) || (guess_level > 2)) {
    throw std::domain_error("Guess level is not 0, 1, or 2");
  }

  if (vert_.empty() || graph_.size() < 2 || nvert_ != graph_[0] ||
      static_cast<int>(graph_.size()) != 2 + 2 * nvert_ + 3 * graph_[1]) {
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

template <HillClimbingType HCT>
inline Real MeshImpl<HCT>::SupportFunction(const Vec3r& n, Vec3r& sp,
                                           SupportFunctionHint<3>* hint) const {
  // If the current normal is much different than the previous normal,
  // compute a new warm start index.
  int idx = hint ? hint->idx_ws : 0;
  if (hint && hint->n_prev.dot(n) < thresh_) {
    if (idx_ws0_.empty()) {
      idx = 0;
    } else {
      Real s = Real(0.0), smax = -kInf;
      for (int i : idx_ws0_) {
        if ((s = n.dot(vert_[i])) > smax) {
          idx = i;
          smax = s;
        }
      }
    }
  }
  assert(idx >= 0);

  // Current best index, neighbour index, previous best index.
  int nidx = -1, pidx = -1;
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
        if constexpr (HCT == HillClimbingType::Greedy) break;
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

template <HillClimbingType HCT>
inline Real MeshImpl<HCT>::SupportFunction(const Vec3r& n,
                                           SupportFunctionDerivatives<3>& deriv,
                                           SupportFunctionHint<3>* hint) const {
  const Real sv = SupportFunction(n, deriv.sp, hint);

  deriv.differentiable = true;
  const int idx = hint->idx_ws;
  int nidx = -1;
  for (int i = *(vert_edgeadr_ + idx); (nidx = *(edge_localid_ + i)) > -1;
       ++i) {
    if (n.dot(vert_[nidx]) > sv - margin_ - eps_sp_) {
      deriv.differentiable = false;
      break;
    }
  }
  if (deriv.differentiable) {
    deriv.Dsp = margin_ * (Matr<3, 3>::Identity() - n * n.transpose());
  }
  return sv;
}

template <HillClimbingType HCT>
inline bool MeshImpl<HCT>::RequireUnitNormal() const {
  return (margin_ > Real(0.0));
}

template <HillClimbingType HCT>
inline void MeshImpl<HCT>::ComputeLocalGeometry(
    const NormalPair<3>& zn, SupportPatchHull<3>& sph, NormalConeSpan<3>& ncs,
    const BasePointHint<3>* hint) const {
  // See the implementation in the Polytope class for reference.
  Vec3r e_perp;
  Real sv_e;
  bool check_edge = false;

  auto set_edge = [&](int i, int j) -> void {
    const Matr<3, 3>& s = (*hint->s);
    e_perp = (s.col(j) - s.col(i)).cross(zn.n).normalized();
    sv_e = e_perp.dot(s.col(i));
    check_edge = true;
  };

  // Compute the normal cone span.
  if (!IsPolytopic()) {
    ncs.span_dim = 1;  // Normal cone is a ray.
  } else {
    Vec3i sidx;
    int ns;
    if ((!hint) || ((ns = hint->ComputeSimplexIndices(sidx)) < 0)) {
      ncs.span_dim = 3;  // Over-approximation.
    } else {
      const Vec3r& bc = (*hint->bc);
      if (ns == 1) {  // Unique simplex point (base point must be a vertex).
        ncs.span_dim = 3;  // Normal cone is a 3D cone.
      } else if (ns == 2) {
        // Two unique simplex points (base point can be on a face/edge/vertex).
        const Real bc1 = bc.dot(sidx.cast<Real>());
        if ((bc1 <= eps_p_) || (bc1 >= Real(1.0) - eps_p_)) {
          ncs.span_dim = 3;  // Normal cone is a 3D cone.
        } else {
          set_edge(0, 2 - sidx(1));
        }
      } else {  // Three unique simplex points.
        const Real eps = Real(0.5) * eps_p_;
        // Check all face/edge/vertex combinations.
        if (bc(0) <= eps) {
          if ((bc(1) <= eps) || (bc(2) <= eps)) {
            ncs.span_dim = 3;
          } else {
            set_edge(1, 2);
          }
        } else if (bc(1) <= eps) {
          if (bc(2) <= eps) {
            ncs.span_dim = 3;
          } else {
            set_edge(0, 2);
          }
        } else if (bc(2) <= eps) {
          set_edge(0, 1);
        } else {
          ncs.span_dim = 1;
        }
      }
    }
  }

  // Compute the support function value and vertex index.
  int idx = (hint && hint->sfh) ? hint->sfh->idx_ws : 0;
  int nidx = -1, pidx = -1;
  Real s = Real(0.0), sv = zn.n.dot(vert_[idx]);
  // Hill-climbing.
  do {
    pidx = idx;
    for (int i = *(vert_edgeadr_ + idx); (nidx = *(edge_localid_ + i)) > -1;
         ++i) {
      s = zn.n.dot(vert_[nidx]);
      if (s > sv) {
        idx = nidx;
        sv = s;
        if constexpr (HCT == HillClimbingType::Greedy) break;
      }
    }
  } while (idx != pidx);

  // Compute the support patch affine hull using adjacency graph.
  sph.aff_dim = 0;
  int idx_e = -1;
  for (int i = *(vert_edgeadr_ + idx); (nidx = *(edge_localid_ + i)) > -1;
       ++i) {
    if (zn.n.dot(vert_[nidx]) >= sv - eps_d_) {
      // nidx lies on the support patch.
      ++sph.aff_dim;
      idx_e = nidx;
    }
  }
  if (sph.aff_dim > 2) {
    sph.aff_dim = 2;
  } else if (sph.aff_dim == 1) {
    // Support patch is a line segment.
    sph.basis.col(0) = (vert_[idx_e] - vert_[idx]).normalized();
  }

  // Compute the normal cone span for the edge case.
  if (check_edge) {
    Real s_e, sv_pm = -kInf;
    bool sv_p = false;
    for (int i = *(vert_edgeadr_ + idx); (nidx = *(edge_localid_ + i)) > -1;) {
      if (zn.n.dot(vert_[nidx]) >= sv - eps_d_) {
        s_e = e_perp.dot(vert_[nidx]);
        if (s_e > sv_e + eps_d_) {
          sv_p = true;
          break;
        }
        if (s_e > sv_pm) {
          sv_pm = s_e;
          i = *(vert_edgeadr_ + nidx);
          continue;
        }
      }
      ++i;
    }
    sv_pm = kInf;
    bool sv_m = false;
    for (int i = *(vert_edgeadr_ + idx); (nidx = *(edge_localid_ + i)) > -1;) {
      if (zn.n.dot(vert_[nidx]) >= sv - eps_d_) {
        s_e = e_perp.dot(vert_[nidx]);
        if (s_e < sv_e - eps_d_) {
          sv_m = true;
          break;
        }
        if (s_e < sv_pm) {
          sv_pm = s_e;
          i = *(vert_edgeadr_ + nidx);
          continue;
        }
      }
      ++i;
    }

    if (sv_p && sv_m) {
      // Normal cone is a ray.
      ncs.span_dim = 1;
    } else {
      // Normal cone is a 2D cone.
      ncs.span_dim = 2;
      ncs.basis.col(0) = e_perp;
    }
  }
}

template <HillClimbingType HCT>
inline bool MeshImpl<HCT>::IsPolytopic() const {
  return (margin_ == Real(0.0));
}

template <HillClimbingType HCT>
inline void MeshImpl<HCT>::PrintInfo() const {
  std::cout << "Type: Mesh (dim = 3)" << std::endl
            << "  Name: " << name_ << std::endl
            << "  #Vertices: " << vert_.size() << std::endl
            << "  Inradius: " << inradius_ << std::endl
            << "  Margin: " << margin_ << std::endl;
}

template <HillClimbingType HCT>
inline const std::vector<Vec3r>& MeshImpl<HCT>::vertices() const {
  return vert_;
}

template <HillClimbingType HCT>
inline const std::vector<int>& MeshImpl<HCT>::graph() const {
  return graph_;
}

template <HillClimbingType HCT>
inline int MeshImpl<HCT>::nvertices() const {
  return nvert_;
}

using Mesh = MeshImpl<HillClimbingType::Greedy>;

}  // namespace dgd

#endif  // DGD_GEOMETRY_3D_MESH_H_
