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
 * @file polygon.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-04-11
 * @brief 2D convex polygon class.
 */

#ifndef DGD_GEOMETRY_2D_POLYGON_H_
#define DGD_GEOMETRY_2D_POLYGON_H_

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/**
 * @brief 2D convex polygon class.
 */
class Polygon : public ConvexSet<2> {
 public:
  /**
   * @brief Constructs a Polygon object.
   *
   * @attention When used as a standalone set, the polygon must contain the
   * origin in its interior. This property is not enforced and must be
   * guaranteed by the user, whenever necessary.
   *
   * @see GrahamScan
   * @see ComputePolygonInradius
   *
   * @param vert     Vector of n two-dimensional vertices.
   * @param margin   Safety margin.
   * @param inradius Polygon inradius.
   */
  explicit Polygon(const std::vector<Vec2f>& vert, Real margin, Real inradius);

  ~Polygon() {};

  Real SupportFunction(
      const Vec2f& n, Vec2f& sp,
      SupportFunctionHint<2>* /*hint*/ = nullptr) const final override;

  bool RequireUnitNormal() const final override;

 private:
  const std::vector<Vec2f> vert_; /**< Polygon vertices. */
  const Real margin_;             /**< Safety margin. */
};

inline Polygon::Polygon(const std::vector<Vec2f>& vert, Real margin,
                        Real inradius)
    : ConvexSet<2>(margin + inradius), vert_(vert), margin_(margin) {
  if ((margin < 0.0) || (inradius <= 0.0))
    throw std::domain_error("Invalid margin or inradius");
}

inline Real Polygon::SupportFunction(const Vec2f& n, Vec2f& sp,
                                     SupportFunctionHint<2>* /*hint*/) const {
  // TODO: Implement hill-climbing/early termination(?)
  int idx{0};
  Real s{0.0}, sv{n.dot(vert_[idx])};

  for (int i = 1; i < static_cast<int>(vert_.size()); ++i) {
    s = n.dot(vert_[i]);
    if (s > sv) {
      idx = i;
      sv = s;
    }
  }

  sp = vert_[idx] + margin_ * n;
  return sv + margin_;
}

inline bool Polygon::RequireUnitNormal() const { return (margin_ > 0.0); }

}  // namespace dgd

#endif  // DGD_GEOMETRY_2D_POLYGON_H_
