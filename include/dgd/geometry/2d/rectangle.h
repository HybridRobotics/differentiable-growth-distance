#ifndef DGD_GEOMETRY_2D_RECTANGLE_H_
#define DGD_GEOMETRY_2D_RECTANGLE_H_

#include <Eigen/Core>
#include <cassert>
#include <cmath>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

class Rectangle : public ConvexSet<2> {
 public:
  Rectangle(Real hlx, Real hly, Real margin);

  ~Rectangle() {};

  Real SupportFunction(const Vec2f& n, Vec2f& sp) const final;

  template <typename Derived>
  Real SupportFunction(const Eigen::MatrixBase<Derived>& n, Vec2f& sp) const;

 private:
  const Real hlx_;
  const Real hly_;
};

inline Rectangle::Rectangle(Real hlx, Real hly, Real margin)
    : ConvexSet<2>(margin), hlx_(hlx), hly_(hly) {
  assert((hlx > Real(0.0)) && (hly > Real(0.0)));
}

template <typename Derived>
inline Real Rectangle::SupportFunction(const Eigen::MatrixBase<Derived>& n,
                                       Vec2f& sp) const {
  static_assert(Derived::RowsAtCompileTime == 2, "Size of normal is not 2!");

  sp(0) = std::copysign(hlx_, n(0));
  sp(1) = std::copysign(hly_, n(1));
  return sp.dot(n);
}

inline Real Rectangle::SupportFunction(const Vec2f& n, Vec2f& sp) const {
  return SupportFunction<Vec2f>(n, sp);
}

}  // namespace dgd

#endif  // DGD_GEOMETRY_2D_RECTANGLE_H_
