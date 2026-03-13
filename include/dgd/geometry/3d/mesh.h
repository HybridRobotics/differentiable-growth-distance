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
#include "dgd/geometry/geometry_utils.h"

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
   * inradius must be an accurate lower bound.
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

  void ComputeLocalGeometryImpl(const NormalPair<3>& zn,
                                SupportPatchHull<3>& sph,
                                NormalConeSpan<3>& ncs,
                                bool check_margin = true,
                                BasePointHint<3>* hint = nullptr) const;

  void ComputeLocalGeometry(
      const NormalPair<3>& zn, SupportPatchHull<3>& sph, NormalConeSpan<3>& ncs,
      BasePointHint<3>* hint = nullptr) const final override;

  bool ProjectionDerivative(
      const Vec3r& p, const Vec3r& pi, Matr<3, 3>& d_pi_p,
      BasePointHint<3>* hint = nullptr) const final override;

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
    : ConvexSet<3>(margin + inradius),
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
  SupportFunctionHint<3>
      hint_tmp;  // Note: hint passed to function can be null.
  if (hint) hint_tmp = *hint;
  const Real sv = SupportFunction(n, deriv.sp, &hint_tmp);
  if (hint) *hint = hint_tmp;

  deriv.differentiable = true;
  const int idx = hint_tmp.idx_ws;
  int nidx = -1;
  for (int i = *(vert_edgeadr_ + idx); (nidx = *(edge_localid_ + i)) > -1;
       ++i) {
    if (n.dot(vert_[nidx]) > sv - margin_ - eps_sp_) {
      deriv.differentiable = false;
      break;
    }
  }
  if (deriv.differentiable) {
    deriv.d_sp_n.noalias() = -margin_ * (n * n.transpose());
    deriv.d_sp_n.diagonal().array() += margin_;
  }
  return sv;
}

template <HillClimbingType HCT>
inline bool MeshImpl<HCT>::RequireUnitNormal() const {
  return (margin_ > Real(0.0));
}

template <HillClimbingType HCT>
inline void MeshImpl<HCT>::ComputeLocalGeometryImpl(
    const NormalPair<3>& zn, SupportPatchHull<3>& sph, NormalConeSpan<3>& ncs,
    bool check_margin, BasePointHint<3>* hint) const {
  const int ns = (hint) ? detail::MergeIndices(*hint, eps_p_) : -1;
  Vec3r e_perp = Vec3r::Zero();
  Real sv_e = Real(0.0);
  bool check_edge = false;

  // Compute the normal cone span.
  if (check_margin && !IsPolytopic()) {
    // Normal cone is a ray.
    ncs.span_dim = 1;
  } else if (ns < 1) {
    ncs.span_dim = 3;  // Over-approximation.
  } else {
    ncs.span_dim = 4 - ns;
    if (ns == 2) {
      const Vec3r& v1 = vert_[hint->idx(0)];
      e_perp = (v1 - vert_[hint->idx(1)]).cross(zn.n).normalized();
      sv_e = e_perp.dot(v1);
      check_edge = true;
    }
  }

  Real sv = Real(0.0);
  int idx = 0, nidx = -1, pidx = -1;
  if (ns > 0) {
    idx = hint->idx(0);
    sv = zn.n.dot(vert_[idx]);
  } else {
    Real s = Real(0.0);
    sv = zn.n.dot(vert_[idx]);
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
  }

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
inline void MeshImpl<HCT>::ComputeLocalGeometry(const NormalPair<3>& zn,
                                                SupportPatchHull<3>& sph,
                                                NormalConeSpan<3>& ncs,
                                                BasePointHint<3>* hint) const {
  ComputeLocalGeometryImpl(zn, sph, ncs, true, hint);
}

template <HillClimbingType HCT>
inline bool MeshImpl<HCT>::ProjectionDerivative(const Vec3r& p, const Vec3r& pi,
                                                Matr<3, 3>& d_pi_p,
                                                BasePointHint<3>* hint) const {
  NormalPair<3> zn;
  const Real dist = (p - pi).norm();
  zn.n = (p - pi) / dist;
  zn.z = pi - margin_ * zn.n;  // Unused.

  SupportPatchHull<3> sph;
  NormalConeSpan<3> ncs;
  ComputeLocalGeometryImpl(zn, sph, ncs, false, hint);

  if (sph.aff_dim + ncs.span_dim > 3) return false;

  if (sph.aff_dim == 2) {
    // Projection lies on a face.
    d_pi_p.noalias() = -(zn.n * zn.n.transpose());
    d_pi_p.diagonal().array() += Real(1.0);
  } else {
    // Projection lies on a vertex/edge.
    const Real s = margin_ / (dist + margin_);
    d_pi_p.noalias() = -s * (zn.n * zn.n.transpose());
    d_pi_p.diagonal().array() += s;
    if (sph.aff_dim == 1) {
      d_pi_p.noalias() += (Real(1.0) - s) * (sph.basis * sph.basis.transpose());
    }
  }

  return true;
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
