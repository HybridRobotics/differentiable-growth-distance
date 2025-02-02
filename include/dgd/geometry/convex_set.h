#ifndef DGD_GEOMETRY_CONVEX_SET_H_
#define DGD_GEOMETRY_CONVEX_SET_H_

#include <cassert>

#include "dgd/data_types.h"

namespace dgd {

template <int dim>
class ConvexSet {
 protected:
  ConvexSet(Real margin);

  Real margin_;

 public:
  virtual ~ConvexSet() {}

  virtual Real SupportFunction(const Vecf<dim>& n, Vecf<dim>& sp) const = 0;

  Real GetMargin() const;

  void SetMargin(Real margin);
};

template <int dim>
inline ConvexSet<dim>::ConvexSet(Real margin) : margin_(margin) {
  assert(margin >= Real(0.0));
}

template <int dim>
inline Real ConvexSet<dim>::GetMargin() const {
  return margin_;
}

template <int dim>
inline void ConvexSet<dim>::SetMargin(Real margin) {
  assert(margin >= Real(0.0));
  margin_ = margin;
}

}  // namespace dgd

#endif  // DGD_GEOMETRY_CONVEX_SET_H_
