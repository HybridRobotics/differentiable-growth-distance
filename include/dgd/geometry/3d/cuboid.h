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
 * @brief 3D cuboid class.
 */

#ifndef DGD_GEOMETRY_3D_CUBOID_H_
#define DGD_GEOMETRY_3D_CUBOID_H_

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/// @brief Axis-aligned 3D cuboid class.
class Cuboid : public ConvexSet<3> {
 public:
  /**
   * @param hlx,hly,hlz Half side lengths.
   * @param margin      Safety margin.
   */
  explicit Cuboid(Real hlx, Real hly, Real hlz, Real margin = Real(0.0));

  ~Cuboid() = default;

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
  const Real hlx_;    /**< Half x-axis side length. */
  const Real hly_;    /**< Half y-axis side length. */
  const Real hlz_;    /**< Half z-axis side length. */
  const Real margin_; /**< Safety margin. */
};

inline Cuboid::Cuboid(Real hlx, Real hly, Real hlz, Real margin)
    : ConvexSet<3>(), hlx_(hlx), hly_(hly), hlz_(hlz), margin_(margin) {
  if ((hlx <= Real(0.0)) || (hly <= Real(0.0)) || (hlz <= Real(0.0)) ||
      (margin < Real(0.0))) {
    throw std::domain_error("Invalid axis lengths or margin");
  }
  set_inradius(std::min({hlx, hly, hlz}) + margin);
}

inline Real Cuboid::SupportFunction(const Vec3r& n, Vec3r& sp,
                                    SupportFunctionHint<3>* /*hint*/) const {
  sp = margin_ * n;
  sp(0) += std::copysign(hlx_, n(0));
  sp(1) += std::copysign(hly_, n(1));
  sp(2) += std::copysign(hlz_, n(2));
  return sp.dot(n);
}

inline Real Cuboid::SupportFunction(const Vec3r& n,
                                    SupportFunctionDerivatives<3>& deriv,
                                    SupportFunctionHint<3>* /*hint*/) const {
  const Real diff = std::min(
      {std::abs(hlx_ * n(0)), std::abs(hly_ * n(1)), std::abs(hlz_ * n(2))});
  if (diff < Real(0.5) * eps_diff()) {
    deriv.differentiable = false;
  } else {
    deriv.Dsp = margin_ * (Matr<3, 3>::Identity() - n * n.transpose());
    deriv.differentiable = true;
  }
  return SupportFunction(n, deriv.sp);
}

inline bool Cuboid::RequireUnitNormal() const { return (margin_ > Real(0.0)); }

inline bool Cuboid::IsPolytopic() const { return (margin_ == Real(0.0)); }

inline void Cuboid::PrintInfo() const {
  std::cout << "Type: Cuboid (dim = 3)" << std::endl
            << "  Half axis lengths: (x: " << hlx_ << ", y: " << hly_
            << ", z: " << hlz_ << ")" << std::endl
            << "  Margin: " << margin_ << std::endl;
}

}  // namespace dgd

#endif  // DGD_GEOMETRY_3D_CUBOID_H_
