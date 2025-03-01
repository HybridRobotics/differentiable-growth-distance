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
 * @file cuboid.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-03-01
 * @brief 3D cuboid class.
 */

#ifndef DGD_GEOMETRY_3D_CUBOID_H_
#define DGD_GEOMETRY_3D_CUBOID_H_

#include <cassert>
#include <cmath>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/**
 * @brief Axis-aligned 3D cuboid class.
 */
class Cuboid : public ConvexSet<3> {
 public:
  /**
   * @brief Constructs a Cuboid object.
   *
   * @param hlx,hly,hlz Half side lengths.
   * @param margin      Safety margin.
   */
  Cuboid(Real hlx, Real hly, Real hlz, Real margin);

  ~Cuboid() {};

  Real SupportFunction(const Vec3f& n, Vec3f& sp) const final;

  template <typename Derived>
  Real SupportFunction(const MatrixBase<Derived>& n, Vec3f& sp) const;

 private:
  const Real hlx_;    /**< Half x-axis side length. */
  const Real hly_;    /**< Half y-axis side length. */
  const Real hlz_;    /**< Half z-axis side length. */
  const Real margin_; /**< Safety margin. */
};

inline Cuboid::Cuboid(Real hlx, Real hly, Real hlz, Real margin)
    : ConvexSet<3>(), hlx_(hlx), hly_(hly), hlz_(hlz), margin_(margin) {
  assert((hlx > Real(0.0)) && (hly > Real(0.0)) && (hlz > Real(0.0)));
  assert(margin >= Real(0.0));
  SetInradius(std::min({hlx, hly, hlz}) + margin);
}

template <typename Derived>
inline Real Cuboid::SupportFunction(const MatrixBase<Derived>& n,
                                    Vec3f& sp) const {
  static_assert(Derived::RowsAtCompileTime == 3, "Size of normal is not 3!");

  sp = margin_ * n;
  sp(0) += std::copysign(hlx_, n(0));
  sp(1) += std::copysign(hly_, n(1));
  sp(2) += std::copysign(hlz_, n(2));
  return sp.dot(n);
}

inline Real Cuboid::SupportFunction(const Vec3f& n, Vec3f& sp) const {
  return SupportFunction<Vec3f>(n, sp);
}

}  // namespace dgd

#endif  // DGD_GEOMETRY_3D_CUBOID_H_
