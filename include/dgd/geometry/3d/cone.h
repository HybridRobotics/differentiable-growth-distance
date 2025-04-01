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
 * @file cone.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-03-01
 * @brief 3D cone class.
 */

#ifndef DGD_GEOMETRY_3D_CONE_H_
#define DGD_GEOMETRY_3D_CONE_H_

#include <cassert>
#include <cmath>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/**
 * @brief Axis-aligned 3D cone class with radius \f$r\f$ and height \f$h\f$.
 *
 * @note The origin is located in the incenter of the cone. The center of the
 * base of the cone is at \f$(0, 0, -\rho)\f$, where \f$\rho\f$ is the inradius
 * of the cone and
 * \f[
 * \rho = \frac{r(\sqrt{r^2 + h^2} - r)}{h}.
 * \f]
 */
class Cone : public ConvexSet<3> {
 public:
  /**
   * @brief Constructs a Cone object.
   *
   * @param radius Radius.
   * @param height Height.
   * @param margin Safety margin.
   */
  Cone(Real radius, Real height, Real margin);

  ~Cone() {};

  Real SupportFunction(const Vec3f& n, Vec3f& sp) const final;

  Real GetOffset() const;

 private:
  const Real r_;      /**< Radius. */
  const Real h_;      /**< Height. */
  Real sha_;          /**< Sine of the cone half angle. */
  Real rho_;          /**< Cone inradius (not considering the safety margin). */
  const Real margin_; /**< Safety margin. */
};

inline Cone::Cone(Real radius, Real height, Real margin)
    : ConvexSet<3>(), r_(radius), h_(height), margin_(margin) {
  assert((radius > Real(0.0)) && (height > Real(0.0)));
  assert(margin >= Real(0.0));
  sha_ = r_ / std::sqrt(r_ * r_ + h_ * h_);
  rho_ = (std::sqrt(r_ * r_ + h_ * h_) * r_ - r_ * r_) / h_;
  SetInradius(rho_ + margin);
}

inline Real Cone::SupportFunction(const Vec3f& n, Vec3f& sp) const {
  sp = margin_ * n;
  if (n(2) >= sha_) {
    // The cone vertex is the support point.
    sp(2) += (h_ - rho_);
    return (h_ - rho_) * n(2) + margin_;
  } else {
    // The support point lies in the cone base.
    const Real k{std::sqrt(n(0) * n(0) + n(1) * n(1))};
    if (k > kEps) sp.head<2>() += r_ * n.head<2>() / k;
    sp(2) -= rho_;
    return sp.dot(n);
  }
}

inline Real Cone::GetOffset() const { return rho_; }

}  // namespace dgd

#endif  // DGD_GEOMETRY_3D_CONE_H_
