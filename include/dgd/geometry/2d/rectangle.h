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
 * @brief 2D rectangle class.
 */

#ifndef DGD_GEOMETRY_2D_RECTANGLE_H_
#define DGD_GEOMETRY_2D_RECTANGLE_H_

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/// @brief Axis-aligned 2D rectangle class.
class Rectangle : public ConvexSet<2> {
 public:
  /**
   * @param hlx,hly Half side lengths.
   * @param margin  Safety margin.
   */
  explicit Rectangle(Real hlx, Real hly, Real margin = Real(0.0));

  ~Rectangle() = default;

  Real SupportFunction(
      const Vec2r& n, Vec2r& sp,
      SupportFunctionHint<2>* /*hint*/ = nullptr) const final override;

  Real SupportFunction(
      const Vec2r& n, SupportFunctionDerivatives<2>& deriv,
      SupportFunctionHint<2>* /*hint*/ = nullptr) const final override;

  bool RequireUnitNormal() const final override;

  void ComputeLocalGeometry(
      const NormalPair<2>& zn, SupportPatchHull<2>& sph, NormalConeSpan<2>& ncs,
      BasePointHint<2>* /*hint*/ = nullptr) const final override;

  bool ProjectionDerivative(
      const Vec2r& p, const Vec2r& /*pi*/, Matr<2, 2>& d_pi_p,
      BasePointHint<2>* /*hint*/ = nullptr) const final override;

  Real Bounds(Vec2r* min = nullptr, Vec2r* max = nullptr) const final override;

  bool IsPolytopic() const final override;

  void PrintInfo() const final override;

 private:
  const Real hlx_;    /**< Half x-axis side length. */
  const Real hly_;    /**< Half y-axis side length. */
  const Real margin_; /**< Safety margin. */
};

inline Rectangle::Rectangle(Real hlx, Real hly, Real margin)
    : ConvexSet<2>(), hlx_(hlx), hly_(hly), margin_(margin) {
  if ((hlx <= Real(0.0)) || (hly <= Real(0.0)) || (margin < Real(0.0))) {
    throw std::domain_error("Invalid axis lengths or margin");
  }
  set_inradius(std::min(hlx, hly) + margin);
}

inline Real Rectangle::SupportFunction(const Vec2r& n, Vec2r& sp,
                                       SupportFunctionHint<2>* /*hint*/) const {
  sp = margin_ * n;
  sp(0) += std::copysign(hlx_, n(0));
  sp(1) += std::copysign(hly_, n(1));
  return sp.dot(n);
}

inline Real Rectangle::SupportFunction(const Vec2r& n,
                                       SupportFunctionDerivatives<2>& deriv,
                                       SupportFunctionHint<2>* /*hint*/) const {
  const Real diff = std::min(std::abs(hlx_ * n(0)), std::abs(hly_ * n(1)));
  if (diff < Real(0.5) * eps_sp_) {
    deriv.differentiable = false;
  } else {
    const Vec2r t(n(1), -n(0));
    deriv.d_sp_n.noalias() = margin_ * (t * t.transpose());
    deriv.differentiable = true;
  }
  return SupportFunction(n, deriv.sp);
}

inline bool Rectangle::RequireUnitNormal() const {
  return (margin_ > Real(0.0));
}

inline void Rectangle::ComputeLocalGeometry(const NormalPair<2>& zn,
                                            SupportPatchHull<2>& sph,
                                            NormalConeSpan<2>& ncs,
                                            BasePointHint<2>* /*hint*/) const {
  if ((std::abs(zn.n(0)) <= eps_d_) || (std::abs(zn.n(1)) <= eps_d_)) {
    // Support patch is a line segment.
    sph.aff_dim = 1;
  } else {
    // Support patch is a point.
    sph.aff_dim = 0;
  }

  if ((margin_ > Real(0.0)) || (hlx_ - std::abs(zn.z(0)) > eps_p_) ||
      (hly_ - std::abs(zn.z(1)) > eps_p_)) {
    // Normal cone is a ray.
    ncs.span_dim = 1;
  } else {
    // Normal cone is a 2D cone.
    ncs.span_dim = 2;
  }
}

inline bool Rectangle::ProjectionDerivative(const Vec2r& p, const Vec2r& /*pi*/,
                                            Matr<2, 2>& d_pi_p,
                                            BasePointHint<2>* /*hint*/) const {
  const Real dx = std::abs(p(0)) - hlx_;
  const Real dy = std::abs(p(1)) - hly_;
  if ((dx > eps_p_) && (dy > eps_p_)) {
    // Projection lies on a vertex.
    const Vec2r v(std::copysign(dx, p(0)), std::copysign(dy, p(1)));
    const Real v2 = v.squaredNorm();
    const Real s = margin_ / std::sqrt(v2);
    d_pi_p.noalias() = -(s / v2) * (v * v.transpose());
    d_pi_p.diagonal().array() += s;
    return true;
  } else if (dx < -eps_p_) {
    // Projection lies on the top/bottom edge.
    d_pi_p.setZero();
    d_pi_p(0, 0) = Real(1.0);
    return true;
  } else if (dy < -eps_p_) {
    // Projection lies on the left/right edge.
    d_pi_p.setZero();
    d_pi_p(1, 1) = Real(1.0);
    return true;
  }

  return false;
}

inline Real Rectangle::Bounds(Vec2r* min, Vec2r* max) const {
  if (min) *min = -Vec2r(hlx_ + margin_, hly_ + margin_);
  if (max) *max = Vec2r(hlx_ + margin_, hly_ + margin_);
  return Real(2.0) * Vec2r(hlx_ + margin_, hly_ + margin_).norm();
}

inline bool Rectangle::IsPolytopic() const { return (margin_ == Real(0.0)); }

inline void Rectangle::PrintInfo() const {
  std::cout << "Type: Rectangle (dim = 2)" << std::endl
            << "  Half axis lengths: (x: " << hlx_ << ", y: " << hly_ << ")"
            << std::endl
            << "  Margin: " << margin_ << std::endl;
}

}  // namespace dgd

#endif  // DGD_GEOMETRY_2D_RECTANGLE_H_
