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

/// @brief Axis-aligned 2D ellipse class.
class Ellipse : public ConvexSet<2> {
 public:
  /**
   * @param hlx,hly Half axis lengths.
   * @param margin  Safety margin.
   */
  explicit Ellipse(Real hlx, Real hly, Real margin = Real(0.0));

  ~Ellipse() = default;

  Real SupportFunction(
      const Vec2r& n, Vec2r& sp,
      SupportFunctionHint<2>* /*hint*/ = nullptr) const final override;

  Real SupportFunction(
      const Vec2r& n, SupportFunctionDerivatives<2>& deriv,
      SupportFunctionHint<2>* /*hint*/ = nullptr) const final override;

  bool RequireUnitNormal() const final override;

  void ComputeLocalGeometry(
      const NormalPair<2>& /*zn*/, SupportPatchHull<2>& sph,
      NormalConeSpan<2>& ncs,
      const BasePointHint<2>* /*hint*/ = nullptr) const final override;

  bool IsPolytopic() const final override;

  void PrintInfo() const final override;

 private:
  const Real hlx2_;   /**< Square of the half x-axis length. */
  const Real hly2_;   /**< Square of the half y-axis length. */
  const Real margin_; /**< Safety margin. */
};

inline Ellipse::Ellipse(Real hlx, Real hly, Real margin)
    : ConvexSet<2>(), hlx2_(hlx * hlx), hly2_(hly * hly), margin_(margin) {
  if ((hlx <= Real(0.0)) || (hly <= Real(0.0)) || (margin < Real(0.0))) {
    throw std::domain_error("Invalid axis lengths or margin");
  }
  set_inradius(std::min(hlx, hly) + margin);
}

inline Real Ellipse::SupportFunction(const Vec2r& n, Vec2r& sp,
                                     SupportFunctionHint<2>* /*hint*/) const {
  const Real k = std::sqrt(hlx2_ * n(0) * n(0) + hly2_ * n(1) * n(1));
  sp.array() = n.array() * (margin_ + Vec2r(hlx2_, hly2_).array() / k);
  return k + margin_;
}

inline Real Ellipse::SupportFunction(const Vec2r& n,
                                     SupportFunctionDerivatives<2>& deriv,
                                     SupportFunctionHint<2>* /*hint*/) const {
  const Real k = std::sqrt(hlx2_ * n(0) * n(0) + hly2_ * n(1) * n(1));
  const Real k_inv = Real(1.0) / k;
  const Vec2r g = Vec2r(hlx2_, hly2_) * k_inv;
  deriv.Dsp = (margin_ + g(0) * g(1) * k_inv) * Vec2r(n(1), -n(0)) *
              Vec2r(n(1), -n(0)).transpose();
  deriv.sp.array() = n.array() * (margin_ + g.array());
  deriv.differentiable = true;
  return k + margin_;
}

inline bool Ellipse::RequireUnitNormal() const { return (margin_ > Real(0.0)); }

inline void Ellipse::ComputeLocalGeometry(
    const NormalPair<2>& /*zn*/, SupportPatchHull<2>& sph,
    NormalConeSpan<2>& ncs, const BasePointHint<2>* /*hint*/) const {
  sph.aff_dim = 0;
  ncs.span_dim = 1;
}

inline bool Ellipse::IsPolytopic() const { return false; }

inline void Ellipse::PrintInfo() const {
  std::cout << "Type: Ellipse (dim = 2)" << std::endl
            << "  Half axis lengths: (x: " << std::sqrt(hlx2_)
            << ", y: " << std::sqrt(hly2_) << ")" << std::endl
            << "  Margin: " << margin_ << std::endl;
}

}  // namespace dgd

#endif  // DGD_GEOMETRY_2D_ELLIPSE_H_
