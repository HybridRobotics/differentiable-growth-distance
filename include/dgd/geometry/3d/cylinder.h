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
 * @file cylinder.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-03-06
 * @brief 3D cylinder class.
 */

#ifndef DGD_GEOMETRY_3D_CYLINDER_H_
#define DGD_GEOMETRY_3D_CYLINDER_H_

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/**
 * @brief Axis-aligned 3D cylinder class.
 *
 * @note The cylinder axis is oriented along the x-axis.
 */
class Cylinder : public ConvexSet<3> {
 public:
  /**
   * @param hlx    Half axis length.
   * @param radius Radius.
   * @param margin Safety margin.
   */
  explicit Cylinder(Real hlx, Real radius, Real margin = 0.0);

  ~Cylinder() = default;

  Real SupportFunction(
      const Vec3r& n, Vec3r& sp,
      SupportFunctionHint<3>* /*hint*/ = nullptr) const final override;

  Real SupportFunction(
      const Vec3r& n, SupportFunctionDerivatives<3>& deriv,
      SupportFunctionHint<3>* /*hint*/ = nullptr) const final override;

  bool RequireUnitNormal() const final override;

  bool IsPolytopic() const final override;

  void PrintInfo() const final override;

 private:
  const Real hlx_;    /**< Half axis length. */
  const Real radius_; /**< Radius. */
  const Real margin_; /**< Safety margin. */
};

inline Cylinder::Cylinder(Real hlx, Real radius, Real margin)
    : ConvexSet<3>(), hlx_(hlx), radius_(radius), margin_(margin) {
  if ((hlx <= 0.0) || (radius <= 0.0) || (margin < 0.0)) {
    throw std::domain_error("Invalid axis length, radius, or margin");
  }
  set_inradius(std::min(hlx, radius) + margin);
}

inline Real Cylinder::SupportFunction(const Vec3r& n, Vec3r& sp,
                                      SupportFunctionHint<3>* /*hint*/) const {
  const Real k = std::sqrt(n(1) * n(1) + n(2) * n(2));
  sp = margin_ * n;
  if (k >= kEps) sp.tail<2>() += radius_ * n.tail<2>() / k;
  sp(0) += std::copysign(hlx_, n(0));
  return sp.dot(n);
}

inline Real Cylinder::SupportFunction(const Vec3r& n,
                                      SupportFunctionDerivatives<3>& deriv,
                                      SupportFunctionHint<3>* /*hint*/) const {
  const Real k2 = n(1) * n(1) + n(2) * n(2);
  const Real k = std::sqrt(k2);
  deriv.sp = margin_ * n;
  if (radius_ * k < Real(0.5) * eps_diff()) {
    deriv.differentiable = false;
  } else {
    if (hlx_ * std::abs(n(0)) < Real(0.5) * eps_diff()) {
      deriv.differentiable = false;
    } else {
      deriv.Dsp = margin_ * (Matr<3, 3>::Identity() - n * n.transpose());
      const Vec2r t = Vec2r(n(2), -n(1));
      deriv.Dsp.block<2, 2>(1, 1) += radius_ / (k2 * k) * t * t.transpose();
      deriv.differentiable = true;
    }
    deriv.sp.tail<2>() += radius_ * n.tail<2>() / k;
  }
  deriv.sp(0) += std::copysign(hlx_, n(0));
  return deriv.sp.dot(n);
}

inline bool Cylinder::RequireUnitNormal() const { return (margin_ > 0.0); }

inline bool Cylinder::IsPolytopic() const { return false; }

inline void Cylinder::PrintInfo() const {
  std::cout << "Type: Cylinder (dim = 3)" << std::endl
            << "  Half axis length: " << hlx_ << std::endl
            << "  Radius: " << radius_ << std::endl
            << "  Margin: " << margin_ << std::endl;
}

}  // namespace dgd

#endif  // DGD_GEOMETRY_3D_CYLINDER_H_
