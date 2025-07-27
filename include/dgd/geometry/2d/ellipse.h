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
 * @file ellipse.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-02-28
 * @brief 2D ellipse class.
 */

#ifndef DGD_GEOMETRY_2D_ELLIPSE_H_
#define DGD_GEOMETRY_2D_ELLIPSE_H_

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/**
 * @brief Axis-aligned 2D ellipse class.
 */
class Ellipse : public ConvexSet<2> {
 public:
  /**
   * @param hlx,hly Half axis lengths.
   * @param margin  Safety margin.
   */
  explicit Ellipse(Real hlx, Real hly, Real margin = 0.0);

  ~Ellipse() = default;

  Real SupportFunction(
      const Vec2r& n, Vec2r& sp,
      SupportFunctionHint<2>* /*hint*/ = nullptr) const final override;

  Real SupportFunction(
      const Vec2r& n, SupportFunctionDerivatives<2>& deriv,
      SupportFunctionHint<2>* /*hint*/ = nullptr) const final override;

  bool RequireUnitNormal() const final override;

  bool IsPolytopic() const final override;

  void PrintInfo() const final override;

 private:
  const Real hlx2_;   /**< Square of the half x-axis length. */
  const Real hly2_;   /**< Square of the half y-axis length. */
  const Real margin_; /**< Safety margin. */
};

inline Ellipse::Ellipse(Real hlx, Real hly, Real margin)
    : ConvexSet<2>(), hlx2_(hlx * hlx), hly2_(hly * hly), margin_(margin) {
  if ((hlx <= 0.0) || (hly <= 0.0) || (margin < 0.0)) {
    throw std::domain_error("Invalid axis lengths or margin");
  }
  set_inradius(std::min(hlx, hly) + margin);
}

inline Real Ellipse::SupportFunction(const Vec2r& n, Vec2r& sp,
                                     SupportFunctionHint<2>* /*hint*/) const {
  const Real k = std::sqrt(hlx2_ * n(0) * n(0) + hly2_ * n(1) * n(1));
  sp(0) = (hlx2_ / k + margin_) * n(0);
  sp(1) = (hly2_ / k + margin_) * n(1);
  return k + margin_;
}

inline Real Ellipse::SupportFunction(const Vec2r& n,
                                     SupportFunctionDerivatives<2>& deriv,
                                     SupportFunctionHint<2>* /*hint*/) const {
  const Real k2 = hlx2_ * n(0) * n(0) + hly2_ * n(1) * n(1);
  const Real k = std::sqrt(k2);
  const Vec2r t = Vec2r(n(1), -n(0));
  deriv.Dsp = (margin_ + hlx2_ * hly2_ / (k2 * k)) * t * t.transpose();
  deriv.sp(0) = (hlx2_ / k + margin_) * n(0);
  deriv.sp(1) = (hly2_ / k + margin_) * n(1);
  deriv.differentiable = true;
  return k + margin_;
}

inline bool Ellipse::RequireUnitNormal() const { return (margin_ > 0.0); }

inline bool Ellipse::IsPolytopic() const { return false; }

inline void Ellipse::PrintInfo() const {
  std::cout << "Type: Ellipse (dim = 2)" << std::endl
            << "  Half axis lengths: (x: " << std::sqrt(hlx2_)
            << ", y: " << std::sqrt(hly2_) << ")" << std::endl
            << "  Margin: " << margin_ << std::endl;
}

}  // namespace dgd

#endif  // DGD_GEOMETRY_2D_ELLIPSE_H_
