#ifndef DGD_GEOMETRY_XD_CAPSULE_H_
#define DGD_GEOMETRY_XD_CAPSULE_H_

#include <cassert>
#include <cmath>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

template <int dim>
class Capsule : public ConvexSet<dim> {
 public:
  Capsule(Real hlx, Real radius, Real margin);

  ~Capsule() {};

  Real SupportFunction(const Vecf<dim>& n, Vecf<dim>& sp) const final;

  template <typename Derived>
  Real SupportFunction(const MatrixBase<Derived>& n, Vecf<dim>& sp) const;

 private:
  const Real hlx_;
  const Real radius_;
};

template <int dim>
inline Capsule<dim>::Capsule(Real hlx, Real radius, Real margin)
    : ConvexSet<dim>(margin, radius), hlx_(hlx), radius_(radius) {
  static_assert((dim == 2) || (dim == 3), "Incompatible dim!");
  assert((hlx > Real(0.0)) && (radius > Real(0.0)));
}

template <int dim>
template <typename Derived>
inline Real Capsule<dim>::SupportFunction(const MatrixBase<Derived>& n,
                                          Vecf<dim>& sp) const {
  static_assert(Derived::RowsAtCompileTime == dim,
                "Size of normal is not dim!");

  sp = radius_ * n;
  sp(0) += std::copysign(hlx_, n(0));
  return sp.dot(n);
}

template <int dim>
inline Real Capsule<dim>::SupportFunction(const Vecf<dim>& n,
                                          Vecf<dim>& sp) const {
  return SupportFunction<Vecf<dim>>(n, sp);
}

}  // namespace dgd

#endif  // DGD_GEOMETRY_XD_CAPSULE_H_
