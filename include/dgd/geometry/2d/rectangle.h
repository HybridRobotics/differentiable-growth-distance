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
 * @file rectangle.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-03-01
 * @brief 2D rectangle class.
 */

#ifndef DGD_GEOMETRY_2D_RECTANGLE_H_
#define DGD_GEOMETRY_2D_RECTANGLE_H_

#include <cmath>
#include <stdexcept>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/**
 * @brief Axis-aligned 2D rectangle class.
 */
class Rectangle : public ConvexSet<2> {
 public:
  /**
   * @brief Constructs a Rectangle object.
   *
   * @param hlx,hly Half side lengths.
   * @param margin  Safety margin.
   */
  explicit Rectangle(Real hlx, Real hly, Real margin);

  ~Rectangle() = default;

  Real SupportFunction(
      const Vec2r& n, Vec2r& sp,
      SupportFunctionHint<2>* /*hint*/ = nullptr) const final override;

  bool RequireUnitNormal() const final override;

 private:
  const Real hlx_;    /**< Half x-axis side length. */
  const Real hly_;    /**< Half y-axis side length. */
  const Real margin_; /**< Safety margin. */
};

inline Rectangle::Rectangle(Real hlx, Real hly, Real margin)
    : ConvexSet<2>(), hlx_(hlx), hly_(hly), margin_(margin) {
  if ((hlx <= 0.0) || (hly <= 0.0) || (margin < 0.0)) {
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

inline bool Rectangle::RequireUnitNormal() const { return (margin_ > 0.0); }

}  // namespace dgd

#endif  // DGD_GEOMETRY_2D_RECTANGLE_H_
