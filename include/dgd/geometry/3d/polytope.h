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
 * @brief 3D polytope class.
 */

#ifndef DGD_GEOMETRY_3D_POLYTOPE_H_
#define DGD_GEOMETRY_3D_POLYTOPE_H_

#include <Eigen/Geometry>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/// @brief 3D convex polytope class.
class Polytope : public ConvexSet<3> {
 public:
  /**
   * @param vert     Vector of n three-dimensional vertices.
   * @param inradius Polytope inradius.
   * @param margin   Safety margin.
   * @param thresh   Support function threshold.
   */
  explicit Polytope(std::vector<Vec3r> vert, Real inradius,
                    Real margin = Real(0.0), Real thresh = Real(0.75));

  ~Polytope() = default;

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

  int nvertices() const;

 private:
  const std::vector<Vec3r> vert_; /**< Polytope vertices. */
  const Real margin_;             /**< Safety margin. */

  const Real thresh_;  // Support function threshold.
};

inline Polytope::Polytope(std::vector<Vec3r> vert, Real inradius, Real margin,
                          Real thresh)
    : ConvexSet<3>(margin + Real(0.99) * inradius),
      vert_(std::move(vert)),
      margin_(margin),
      thresh_(thresh) {
  if ((margin < Real(0.0)) || (inradius <= Real(0.0))) {
    throw std::domain_error("Invalid margin or inradius");
  }
}

inline Real Polytope::SupportFunction(const Vec3r& n, Vec3r& sp,
                                      SupportFunctionHint<3>* hint) const {
  // Current best index.
  int idx = (hint && hint->n_prev.dot(n) > thresh_) ? hint->idx_ws : 0;
  assert(idx >= 0);
  // Current support value, current best support value.
  Real s = Real(0.0), sv = n.dot(vert_[idx]);

  for (int i = 0; i < nvertices(); ++i) {
    s = n.dot(vert_[i]);
    if (s > sv) {
      idx = i;
      sv = s;
    }
  }

  if (hint) {
    hint->n_prev = n;
    hint->idx_ws = idx;
  }

  sp = vert_[idx] + margin_ * n;
  return sv + margin_;
}

inline Real Polytope::SupportFunction(const Vec3r& n,
                                      SupportFunctionDerivatives<3>& deriv,
                                      SupportFunctionHint<3>* hint) const {
  int idx = (hint && hint->n_prev.dot(n) > thresh_) ? hint->idx_ws : 0;
  Real s = Real(0.0), sv = n.dot(vert_[idx]);

  deriv.differentiable = true;
  for (int i = 0; i < nvertices(); ++i) {
    s = n.dot(vert_[i]);
    if (s > sv) {
      deriv.differentiable = (s >= sv + eps_sp_);
      idx = i;
      sv = s;
    } else {
      if (s > sv - eps_sp_) deriv.differentiable = false;
    }
  }
  if (deriv.differentiable) {
    deriv.Dsp = margin_ * (Matr<3, 3>::Identity() - n * n.transpose());
  }

  if (hint) {
    hint->n_prev = n;
    hint->idx_ws = idx;
  }

  deriv.sp = vert_[idx] + margin_ * n;
  return sv + margin_;
}

inline bool Polytope::RequireUnitNormal() const {
  return (margin_ > Real(0.0));
}

inline void Polytope::ComputeLocalGeometry(const NormalPair<3>& zn,
                                           SupportPatchHull<3>& sph,
                                           NormalConeSpan<3>& ncs,
                                           const BasePointHint<3>* hint) const {
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
    // Normal cone is a ray.
    ncs.span_dim = 1;
  } else {
    Vec3i sidx;
    int ns;
    if ((!hint) || ((ns = hint->ComputeSimplexIndices(sidx)) < 0)) {
      ncs.span_dim = 3;  // Over-approximation.
    } else {
      // A simplex point with a positive barycentric coordinate must be a vertex
      // on the support patch.
      // The growth distance algorithm output satisfies this because the stored
      // inradius is positive and strictly less than the actual inradius.
      const Vec3r& bc = (*hint->bc);
      if (ns == 1) {  // Unique simplex point (base point must be a vertex).
        // Normal cone is a 3D cone.
        ncs.span_dim = 3;
      } else if (ns == 2) {
        // Two unique simplex points (base point can be on a face/edge/vertex).
        const Real bc1 = bc.dot(sidx.cast<Real>());
        if ((bc1 <= eps_p_) || (bc1 >= Real(1.0) - eps_p_)) {
          // Base point is a vertex and the normal cone is a 3D cone.
          ncs.span_dim = 3;
        } else {
          // Base point is on a face/edge and both simplex points must lie on
          // the support patch.
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

  // Compute the support function value.
  Real sv = Real(0.0);
  if ((hint) && (hint->s) && (hint->bc)) {
    sv = zn.n.dot((*hint->s) * (*hint->bc));
  } else {
    Real s;
    for (int i = 0; i < nvertices(); ++i) {
      s = zn.n.dot(vert_[i]);
      if (s > sv) sv = s;
    }
  }

  // Compute the support patch affine hull.
  sph.aff_dim = -1;
  Real s_e;
  Vec2i idx_e;  // Edge vector vertex indices.
  bool sv_p = false, sv_m = false;
  for (int i = 0; i < nvertices(); ++i) {
    if (zn.n.dot(vert_[i]) >= sv - eps_d_) {
      ++sph.aff_dim;
      if (sph.aff_dim < 2) idx_e(sph.aff_dim) = i;
      if (check_edge) {
        s_e = e_perp.dot(vert_[i]);
        sv_p |= (s_e > sv_e + eps_d_);
        sv_m |= (s_e < sv_e - eps_d_);
      }
    }
  }
  assert(sph.aff_dim >= 0);
  if (sph.aff_dim > 2) {
    sph.aff_dim = 2;
  } else if (sph.aff_dim == 1) {
    // Support patch is a line segment.
    sph.basis.col(0) = (vert_[idx_e(1)] - vert_[idx_e(0)]).normalized();
  }

  // Compute the normal cone span for the edge case.
  if (check_edge) {
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

inline bool Polytope::IsPolytopic() const { return (margin_ == Real(0.0)); }

inline void Polytope::PrintInfo() const {
  std::cout << "Type: Polytope (dim = 3)" << std::endl
            << "  #Vertices: " << vert_.size() << std::endl;
  for (const auto& v : vert_) {
    std::cout << "    (" << v(0) << ", " << v(1) << ", " << v(2) << ")"
              << std::endl;
  }
  std::cout << "  Inradius: " << inradius_ << std::endl
            << "  Margin: " << margin_ << std::endl;
}

inline const std::vector<Vec3r>& Polytope::vertices() const { return vert_; }

inline int Polytope::nvertices() const {
  return static_cast<int>(vert_.size());
}

}  // namespace dgd

#endif  // DGD_GEOMETRY_3D_POLYTOPE_H_
