#ifndef DGD_GEOMETRY_2D_ELLIPSE_H_
#define DGD_GEOMETRY_2D_ELLIPSE_H_

#include <cassert>
#include <cmath>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

class Ellipse : public ConvexSet<2> {
 public:
  Ellipse(Real hlx, Real hly, Real margin);

  ~Ellipse() {};

  Real SupportFunction(const Vec2f& n, Vec2f& sp) const final;

  template <typename Derived>
  Real SupportFunction(const MatrixBase<Derived>& n, Vec2f& sp) const;

 private:
  const Real hlx2_;
  const Real hly2_;
};

inline Ellipse::Ellipse(Real hlx, Real hly, Real margin)
    : ConvexSet<2>(margin), hlx2_(hlx * hlx), hly2_(hly * hly) {
  assert((hlx > Real(0.0)) && (hly > Real(0.0)));
  SetInradius(std::min(hlx, hly));
}

template <typename Derived>
inline Real Ellipse::SupportFunction(const MatrixBase<Derived>& n,
                                     Vec2f& sp) const {
  static_assert(Derived::RowsAtCompileTime == 2, "Size of normal is not 2!");

  const Real k = std::sqrt(hlx2_ * n(0) * n(0) + hly2_ * n(1) * n(1));
  sp(0) = hlx2_ * n(0) / k;
  sp(1) = hly2_ * n(1) / k;
  return k;
}

inline Real Ellipse::SupportFunction(const Vec2f& n, Vec2f& sp) const {
  return SupportFunction<Vec2f>(n, sp);
}

}  // namespace dgd

#endif  // DGD_GEOMETRY_2D_ELLIPSE_H_
