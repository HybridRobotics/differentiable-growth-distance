#ifndef DGD_GEOMETRY_XD_SPHERE_H_
#define DGD_GEOMETRY_XD_SPHERE_H_

#include <cassert>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

template <int dim>
class Sphere : public ConvexSet<dim> {
 public:
  Sphere(Real radius, Real margin);

  ~Sphere() {};

  Real SupportFunction(const Vecf<dim>& n, Vecf<dim>& sp) const final;

  template <typename Derived>
  Real SupportFunction(const MatrixBase<Derived>& n, Vecf<dim>& sp) const;

 private:
  const Real radius_;
};

template <int dim>
inline Sphere<dim>::Sphere(Real radius, Real margin)
    : ConvexSet<dim>(margin, radius), radius_(radius) {
  static_assert((dim == 2) || (dim == 3), "Incompatible dim!");
  assert(radius > Real(0.0));
}

template <int dim>
template <typename Derived>
inline Real Sphere<dim>::SupportFunction(const MatrixBase<Derived>& n,
                                         Vecf<dim>& sp) const {
  static_assert(Derived::RowsAtCompileTime == dim,
                "Size of normal is not dim!");

  sp = radius_ * n;
  return radius_;
}

template <int dim>
inline Real Sphere<dim>::SupportFunction(const Vecf<dim>& n,
                                         Vecf<dim>& sp) const {
  return SupportFunction<Vecf<dim>>(n, sp);
}

typedef Sphere<2> Circle;

}  // namespace dgd

#endif  // DGD_GEOMETRY_XD_SPHERE_H_
