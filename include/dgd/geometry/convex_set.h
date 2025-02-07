#ifndef DGD_GEOMETRY_CONVEX_SET_H_
#define DGD_GEOMETRY_CONVEX_SET_H_

#include <cassert>

#include "dgd/data_types.h"

namespace dgd {

/**
 * @brief Convex set abstract class implementing the support function.
 *
 * @tparam dim
 *
 * @note The convex set must be compact and solid. Also, 0 \in \mathbb{R}^dim
 * must be in the interior of the convex set (TODO: remove this requirement).
 */
template <int dim>
class ConvexSet {
 protected:
  ConvexSet();

  ConvexSet(Real margin);

  /**
   * @brief Construct a new Convex Set object
   *
   * @param margin
   * @param inradius (any number greater than zero and less than the radius of
   * the maximal circle at 0.)
   */
  ConvexSet(Real margin, Real inradius);

  Real margin_;
  Real inradius_;

 public:
  virtual ~ConvexSet() {}

  /**
   * @brief Support function for the convex set.
   *
   * @param n (normalized)
   * @param sp
   * @return v
   */
  virtual Real SupportFunction(const Vecf<dim>& n, Vecf<dim>& sp) const = 0;

  Real GetMargin() const;

  void SetMargin(Real margin);

  Real GetInradius() const;

  void SetInradius(Real inradius);
};

template <int dim>
inline ConvexSet<dim>::ConvexSet() : ConvexSet(Real(0.0), kEps) {}

template <int dim>
inline ConvexSet<dim>::ConvexSet(Real margin) : ConvexSet(margin, kEps) {}

template <int dim>
inline ConvexSet<dim>::ConvexSet(Real margin, Real inradius)
    : margin_(margin), inradius_(inradius) {
  assert(margin >= Real(0.0));
  assert(inradius > Real(0.0));
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

template <int dim>
inline Real ConvexSet<dim>::GetInradius() const {
  return inradius_;
}

template <int dim>
inline void ConvexSet<dim>::SetInradius(Real inradius) {
  assert(inradius > Real(0.0));
  inradius_ = inradius;
}

}  // namespace dgd

#endif  // DGD_GEOMETRY_CONVEX_SET_H_
