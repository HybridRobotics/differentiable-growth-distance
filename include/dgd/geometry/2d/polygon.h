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
 * @brief 2D convex polygon class.
 */

#ifndef DGD_GEOMETRY_2D_POLYGON_H_
#define DGD_GEOMETRY_2D_POLYGON_H_

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"
#include "dgd/geometry/geometry_utils.h"

namespace dgd {

/// @brief 2D convex polygon class.
class Polygon : public ConvexSet<2> {
 public:
  /**
   * @attention The polygon must contain the origin in its interior, and the
   * vertices must be in counter-clockwise (CCW) order.
   *
   * @see GrahamScan
   * @see ComputePolygonInradius
   *
   * @param vert     Vector of n two-dimensional vertices in CCW order.
   * @param inradius Polygon inradius.
   * @param margin   Safety margin.
   */
  explicit Polygon(std::vector<Vec2r> vert, Real inradius,
                   Real margin = Real(0.0));

  ~Polygon() = default;

  Real SupportFunction(
      const Vec2r& n, Vec2r& sp,
      SupportFunctionHint<2>* hint = nullptr) const final override;

  Real SupportFunction(
      const Vec2r& n, SupportFunctionDerivatives<2>& deriv,
      SupportFunctionHint<2>* hint = nullptr) const final override;

  bool RequireUnitNormal() const final override;

  void ComputeLocalGeometryImpl(const NormalPair<2>& zn,
                                SupportPatchHull<2>& sph,
                                NormalConeSpan<2>& ncs,
                                bool check_margin = true,
                                BasePointHint<2>* hint = nullptr) const;

  void ComputeLocalGeometry(
      const NormalPair<2>& zn, SupportPatchHull<2>& sph, NormalConeSpan<2>& ncs,
      BasePointHint<2>* hint = nullptr) const final override;

  bool ProjectionDerivative(
      const Vec2r& p, const Vec2r& pi, Matr<2, 2>& d_pi_p,
      BasePointHint<2>* hint = nullptr) const final override;

  bool IsPolytopic() const final override;

  void PrintInfo() const final override;

  const std::vector<Vec2r>& vertices() const;

  int nvertices() const;

 private:
  const std::vector<Vec2r> vert_; /**< Polygon vertices. */
  const Real margin_;             /**< Safety margin. */
};

inline Polygon::Polygon(std::vector<Vec2r> vert, Real inradius, Real margin)
    : ConvexSet<2>(margin + inradius), vert_(std::move(vert)), margin_(margin) {
  if ((margin < Real(0.0)) || (inradius <= Real(0.0))) {
    throw std::domain_error("Invalid margin or inradius");
  }
}

inline Real Polygon::SupportFunction(const Vec2r& n, Vec2r& sp,
                                     SupportFunctionHint<2>* hint) const {
  // Other methods/options:
  // 1. Early termination of the loop.
  // 2. Hill-climbing algorithm.
  // 3. Bisection method.
  int idx = 0;
  Real s = Real(0.0), sv = n.dot(vert_[idx]);

  for (int i = 1; i < nvertices(); ++i) {
    s = n.dot(vert_[i]);
    if (s > sv) {
      idx = i;
      sv = s;
    }
  }

  if (hint) hint->idx_ws = idx;

  sp = vert_[idx] + margin_ * n;
  return sv + margin_;
}

inline Real Polygon::SupportFunction(const Vec2r& n,
                                     SupportFunctionDerivatives<2>& deriv,
                                     SupportFunctionHint<2>* hint) const {
  SupportFunctionHint<2>
      hint_tmp;  // Note: hint passed to function can be null.
  const Real sv = SupportFunction(n, deriv.sp, &hint_tmp);
  if (hint) hint->idx_ws = hint_tmp.idx_ws;

  const int idx = hint_tmp.idx_ws;
  const int prev = (idx == 0) ? nvertices() - 1 : idx - 1;
  const int next = (idx == nvertices() - 1) ? 0 : idx + 1;
  if (std::max(n.dot(vert_[prev]), n.dot(vert_[next])) >
      sv - margin_ - eps_sp_) {
    deriv.differentiable = false;
  } else {
    const Vec2r t(n(1), -n(0));
    deriv.d_sp_n.noalias() = margin_ * (t * t.transpose());
    deriv.differentiable = true;
  }
  return sv;
}

inline bool Polygon::RequireUnitNormal() const { return (margin_ > Real(0.0)); }

inline void Polygon::ComputeLocalGeometryImpl(const NormalPair<2>& zn,
                                              SupportPatchHull<2>& sph,
                                              NormalConeSpan<2>& ncs,
                                              bool check_margin,
                                              BasePointHint<2>* hint) const {
  // Compute the normal cone span.
  if (check_margin && !IsPolytopic()) {
    // Normal cone is a ray.
    ncs.span_dim = 1;
  } else {
    int ns = 0;
    if ((!hint) || ((ns = detail::MergeIndices(*hint, eps_p_)) < 1)) {
      ncs.span_dim = 2;  // Over-approximation.
    } else {
      ncs.span_dim = 3 - ns;
    }
  }

  // Compute the support patch affine hull.
  sph.aff_dim = 0;
  Real s = Real(0.0), sv = zn.n.dot(vert_[0]);
  for (int i = 1; i < nvertices(); ++i) {
    s = zn.n.dot(vert_[i]);
    if (s > sv) {
      sph.aff_dim = (s <= sv + eps_d_);
      sv = s;
    } else {
      if (s >= sv - eps_d_) sph.aff_dim = 1;
    }
  }
}

inline void Polygon::ComputeLocalGeometry(const NormalPair<2>& zn,
                                          SupportPatchHull<2>& sph,
                                          NormalConeSpan<2>& ncs,
                                          BasePointHint<2>* hint) const {
  ComputeLocalGeometryImpl(zn, sph, ncs, true, hint);
}

inline bool Polygon::ProjectionDerivative(const Vec2r& p, const Vec2r& pi,
                                          Matr<2, 2>& d_pi_p,
                                          BasePointHint<2>* hint) const {
  NormalPair<2> zn;
  const Real dist = (p - pi).norm();
  zn.n = (p - pi) / dist;
  zn.z = pi - margin_ * zn.n;  // Unused.

  SupportPatchHull<2> sph;
  NormalConeSpan<2> ncs;
  ComputeLocalGeometryImpl(zn, sph, ncs, false, hint);

  if (sph.aff_dim == 0) {
    // Projection lies on a vertex.
    const Real s = margin_ / (dist + margin_);
    d_pi_p.noalias() = -s * (zn.n * zn.n.transpose());
    d_pi_p.diagonal().array() += s;
  } else {
    if (ncs.span_dim == 2) return false;
    // Projection lies on an edge.
    d_pi_p.noalias() = -(zn.n * zn.n.transpose());
    d_pi_p.diagonal().array() += Real(1.0);
  }

  return true;
}

inline bool Polygon::IsPolytopic() const { return (margin_ == Real(0.0)); }

inline void Polygon::PrintInfo() const {
  std::cout << "Type: Polygon (dim = 2)" << std::endl
            << "  #Vertices: " << vert_.size() << std::endl;
  for (const auto& v : vert_) {
    std::cout << "    (" << v(0) << ", " << v(1) << ")" << std::endl;
  }
  std::cout << "  Inradius: " << inradius_ << std::endl
            << "  Margin: " << margin_ << std::endl;
}

inline const std::vector<Vec2r>& Polygon::vertices() const { return vert_; }

inline int Polygon::nvertices() const { return static_cast<int>(vert_.size()); }

}  // namespace dgd

#endif  // DGD_GEOMETRY_2D_POLYGON_H_
