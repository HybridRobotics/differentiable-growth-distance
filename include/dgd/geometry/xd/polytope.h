#ifndef DGD_GEOMETRY_XD_POLYTOPE_H_
#define DGD_GEOMETRY_XD_POLYTOPE_H_

#include <Eigen/Dense>
#include <cassert>
#include <cmath>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

template <int dim>
class Polytope : public ConvexSet<dim> {
 public:
  Polytope(const MatXf<dim>& pts, Real margin, Real inradius);

  ~Polytope() {};

  Real SupportFunction(const Vecf<dim>& n, Vecf<dim>& sp) const final;

  template <typename Derived>
  Real SupportFunction(const MatrixBase<Derived>& n, Vecf<dim>& sp) const;

 private:
  const MatXf<dim> pts_;
};

template <int dim>
inline Polytope<dim>::Polytope(const MatXf<dim>& pts, Real margin,
                               Real inradius)
    : ConvexSet<dim>(margin, inradius), pts_(pts) {
  static_assert((dim == 2) || (dim == 3), "Incompatible dim!");

  const int num_pts = static_cast<int>(pts.cols());
  assert(num_pts >= dim + 1);

  MatXf<dim> aff_pts = pts.rightCols(num_pts - 1);
  aff_pts.colwise() -= pts.col(0);
  const Eigen::ColPivHouseholderQR<MatXf<dim>> qr(aff_pts);
  const int rank = static_cast<int>(qr.rank());
  assert(rank == dim);
}

template <int dim>
template <typename Derived>
inline Real Polytope<dim>::SupportFunction(const MatrixBase<Derived>& n,
                                           Vecf<dim>& sp) const {
  static_assert(Derived::RowsAtCompileTime == dim,
                "Size of normal is not dim!");

  int idx{0};
  const Real v = (pts_.transpose() * n).maxCoeff(&idx);
  sp = pts_.col(idx);
  return v;
}

template <int dim>
inline Real Polytope<dim>::SupportFunction(const Vecf<dim>& n,
                                           Vecf<dim>& sp) const {
  return SupportFunction<Vecf<dim>>(n, sp);
}

typedef Polytope<2> Polygon;

}  // namespace dgd

#endif  // DGD_GEOMETRY_XD_POLYTOPE_H_
