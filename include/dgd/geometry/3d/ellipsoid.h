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
 * @brief 3D ellipse class.
 */

#ifndef DGD_GEOMETRY_3D_ELLIPSOID_H_
#define DGD_GEOMETRY_3D_ELLIPSOID_H_

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/// @brief Axis-aligned 3D ellipsoid class.
class Ellipsoid : public ConvexSet<3> {
 public:
  /**
   * @param hlx,hly,hlz Half axis lengths.
   * @param margin      Safety margin.
   */
  explicit Ellipsoid(Real hlx, Real hly, Real hlz, Real margin = Real(0.0));

  ~Ellipsoid() = default;

  Real SupportFunction(
      const Vec3r& n, Vec3r& sp,
      SupportFunctionHint<3>* /*hint*/ = nullptr) const final override;

  Real SupportFunction(
      const Vec3r& n, SupportFunctionDerivatives<3>& deriv,
      SupportFunctionHint<3>* /*hint*/ = nullptr) const final override;

  bool RequireUnitNormal() const final override;

  void ComputeLocalGeometry(
      const NormalPair<3>& /*zn*/, SupportPatchHull<3>& sph,
      NormalConeSpan<3>& ncs,
      const BasePointHint<3>* /*hint*/ = nullptr) const final override;

  bool IsPolytopic() const final override;

  void PrintInfo() const final override;

 private:
  const Real hlx2_;   /**< Square of the half x-axis length. */
  const Real hly2_;   /**< Square of the half y-axis length. */
  const Real hlz2_;   /**< Square of the half z-axis length. */
  const Real margin_; /**< Safety margin. */
};

inline Ellipsoid::Ellipsoid(Real hlx, Real hly, Real hlz, Real margin)
    : ConvexSet<3>(),
      hlx2_(hlx * hlx),
      hly2_(hly * hly),
      hlz2_(hlz * hlz),
      margin_(margin) {
  if ((hlx <= Real(0.0)) || (hly <= Real(0.0)) || (hlz <= Real(0.0)) ||
      (margin < Real(0.0))) {
    throw std::domain_error("Invalid axis lengths or margin");
  }
  set_inradius(std::min({hlx, hly, hlz}) + margin);
}

inline Real Ellipsoid::SupportFunction(const Vec3r& n, Vec3r& sp,
                                       SupportFunctionHint<3>* /*hint*/) const {
  const Real k = std::sqrt(hlx2_ * n(0) * n(0) + hly2_ * n(1) * n(1) +
                           hlz2_ * n(2) * n(2));
  sp.array() = n.array() * (margin_ + Vec3r(hlx2_, hly2_, hlz2_).array() / k);
  return k + margin_;
}

inline Real Ellipsoid::SupportFunction(const Vec3r& n,
                                       SupportFunctionDerivatives<3>& deriv,
                                       SupportFunctionHint<3>* /*hint*/) const {
  const Real k = std::sqrt(hlx2_ * n(0) * n(0) + hly2_ * n(1) * n(1) +
                           hlz2_ * n(2) * n(2));
  const Real k_inv = Real(1.0) / k;
  const Vec3r g = Vec3r(hlx2_, hly2_, hlz2_) * k_inv;
  const Vec3r gn = g.cwiseProduct(n);
  deriv.Dsp = margin_ * (Matr<3, 3>::Identity() - n * n.transpose()) -
              gn * gn.transpose() * k_inv;
  deriv.Dsp += g.asDiagonal();
  deriv.sp = gn + margin_ * n;
  deriv.differentiable = true;
  return k + margin_;
}

inline bool Ellipsoid::RequireUnitNormal() const {
  return (margin_ > Real(0.0));
}

inline void Ellipsoid::ComputeLocalGeometry(
    const NormalPair<3>& /*zn*/, SupportPatchHull<3>& sph,
    NormalConeSpan<3>& ncs, const BasePointHint<3>* /*hint*/) const {
  sph.aff_dim = 0;
  ncs.span_dim = 1;
}

inline bool Ellipsoid::IsPolytopic() const { return false; }

inline void Ellipsoid::PrintInfo() const {
  std::cout << "Type: Ellipsoid (dim = 3)" << std::endl
            << "  Half axis lengths: (x: " << std::sqrt(hlx2_)
            << ", y: " << std::sqrt(hly2_) << ", z: " << std::sqrt(hlz2_) << ")"
            << std::endl
            << "  Margin: " << margin_ << std::endl;
}

}  // namespace dgd

#endif  // DGD_GEOMETRY_3D_ELLIPSOID_H_
