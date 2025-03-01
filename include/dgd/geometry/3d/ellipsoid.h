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
 * @file ellipsoid.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-03-01
 * @brief 3D ellipse class.
 */

#ifndef DGD_GEOMETRY_3D_ELLIPSOID_H_
#define DGD_GEOMETRY_3D_ELLIPSOID_H_

#include <cassert>
#include <cmath>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/**
 * @brief Axis-aligned 3D ellipsoid class.
 */
class Ellipsoid : public ConvexSet<3> {
 public:
  /**
   * @brief Constructs an Ellipsoid object.
   *
   * @param hlx,hly,hlz Half axis lengths.
   * @param margin      Safety margin.
   */
  Ellipsoid(Real hlx, Real hly, Real hlz, Real margin);

  ~Ellipsoid() {};

  Real SupportFunction(const Vec3f& n, Vec3f& sp) const final;

  template <typename Derived>
  Real SupportFunction(const MatrixBase<Derived>& n, Vec3f& sp) const;

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
  assert((hlx > Real(0.0)) && (hly > Real(0.0)) && (hlz > Real(0.0)));
  assert(margin >= Real(0.0));
  SetInradius(std::min({hlx, hly, hlz}) + margin);
}

template <typename Derived>
inline Real Ellipsoid::SupportFunction(const MatrixBase<Derived>& n,
                                       Vec3f& sp) const {
  static_assert(Derived::RowsAtCompileTime == 3, "Size of normal is not 3!");

  const Real k{std::sqrt(hlx2_ * n(0) * n(0) + hly2_ * n(1) * n(1) +
                         hlz2_ * n(2) * n(2))};
  sp(0) = (hlx2_ / k + margin_) * n(0);
  sp(1) = (hly2_ / k + margin_) * n(1);
  sp(2) = (hlz2_ / k + margin_) * n(2);
  return k + margin_;
}

inline Real Ellipsoid::SupportFunction(const Vec3f& n, Vec3f& sp) const {
  return SupportFunction<Vec3f>(n, sp);
}

}  // namespace dgd

#endif  // DGD_GEOMETRY_3D_ELLIPSOID_H_
