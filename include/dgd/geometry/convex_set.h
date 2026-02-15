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
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @brief Convex set abstract class implementation.
 */

#ifndef DGD_GEOMETRY_CONVEX_SET_H_
#define DGD_GEOMETRY_CONVEX_SET_H_

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include "dgd/data_types.h"

namespace dgd {

/**
 * @brief Support function hint.
 *
 * @tparam dim Dimension of the convex set.
 */
template <int dim>
struct SupportFunctionHint {
  /// @brief Normal vector (in local coordinates) at the previous iteration.
  Vecr<dim> n_prev = Vecr<dim>::Zero();

  /// @brief Integer hint for support function computation.
  int idx_ws = -1;
};

/**
 * @brief Second-order support function derivatives.
 *
 * @tparam dim Dimension of the convex set.
 */
template <int dim>
struct SupportFunctionDerivatives {
  /**
   * @brief Support point derivative with respect to the normal vector.
   *
   * @attention The support point map derivative is equal to the support
   * function Hessian, and it exists almost everywhere. Whenever it exists, it
   * must be positive semi-definite. Note that the normal vector lies in the
   * null space of the support point derivative.
   *
   * @see differentiable
   */
  Matr<dim, dim> Dsp;

  /// @brief Support point.
  Vecr<dim> sp;

  /**
   * @brief Differentiability of the support point map at the given normal
   * vector.
   */
  bool differentiable;
};

/**
 * @brief Normal vector and base point pair for a convex set.
 *
 * @note The base point lies on the boundary of the convex set.
 * @note The normal vector has unit norm and lies in the normal cone at the
 * base point.
 *
 * @tparam dim Dimension of the convex set.
 */
template <int dim>
struct NormalPair {
  /// @brief Base point in the convex set.
  Vecr<dim> z = Vecr<dim>::Zero();

  /// @brief Normal vector at the base point.
  Vecr<dim> n = Vecr<dim>::Zero();
};

/**
 * @brief Affine hull of the support patch of a convex set at a normal vector.
 *
 * @note The translation vector for the affine hull is not stored.
 * @note The affine dimension of the support patch is at most dim - 1.
 *
 * @tparam dim Dimension of the convex set.
 */
template <int dim>
struct SupportPatchHull {
  /**
   * @brief Linear subspace basis for the support patch affine hull.
   *
   * The basis is only used if \f$0 < \text{aff_dim} < \text{dim} - 1\f$.
   */
  Matr<dim, dim - 2> basis;

  /// @brief Affine dimension of the support patch (number of basis vectors).
  int aff_dim = dim - 1;
};

template <>
struct SupportPatchHull<2> {
  int aff_dim = 1;
};

/**
 * @brief Span of the normal cone of a convex set at a base point-normal vector
 * pair.
 *
 * @attention At a base point-normal vector pair \f$(z, n)\f$, the normal cone
 * span always contains \f$n\f$. The stored basis vectors are orthogonal to
 * \f$n\f$, and do not include it.
 *
 * @tparam dim Dimension of the convex set.
 */
template <int dim>
struct NormalConeSpan {
  /**
   * @brief Basis for the normal cone span, excluding the normal vector.
   *
   * The basis is only used if \f$1 < \text{span\_dim} < \text{dim}\f$.
   */
  Matr<dim, dim - 2> basis;

  /// @brief Dimension of the normal cone span.
  int span_dim = dim;
};

template <>
struct NormalConeSpan<2> {
  int span_dim = 2;
};

/**
 * @brief Hint for a base point in a convex set.
 *
 * @see NormalPair
 *
 * @tparam dim Dimension of the convex set.
 */
template <int dim>
struct BasePointHint {
  /// @brief Simplex representation of the base point.
  const Matr<dim, dim>* s = nullptr;

  /**
   * @brief Barycentric coordinates for the base point corresponding to the
   * simplex.
   */
  const Vecr<dim>* bc = nullptr;

  /// @brief Support function hint.
  const SupportFunctionHint<dim>* sfh = nullptr;

  /**
   * @brief Computes indices for unique simplex points.
   *
   * @param[out] idx Simplex indices.
   * @return     Number of unique simplex points; -1 if s or bc is null.
   */
  int ComputeSimplexIndices(Veci<dim>& idx) const;
};

template <int dim>
inline int BasePointHint<dim>::ComputeSimplexIndices(Veci<dim>& idx) const {
  if ((!s) || (!bc)) return -1;
  constexpr Real kEps2 = kEps * kEps;
  int max_idx = 0;
  idx(0) = 0;
  idx(1) = ((s->col(1) - s->col(0)).squaredNorm() < kEps2) ? 0 : ++max_idx;
  if constexpr (dim == 3) {
    if ((s->col(2) - s->col(0)).squaredNorm() < kEps2) {
      idx(2) = 0;
    } else if ((idx(1) == 1) && (s->col(2) - s->col(1)).squaredNorm() < kEps2) {
      idx(2) = 1;
    } else {
      idx(2) = ++max_idx;
    }
  }
  return max_idx + 1;
}

/**
 * @brief Convex set abstract class implementing the support function.
 *
 * @attention The convex set must be compact (i.e., closed and bounded) and
 * must contain the origin. When the set is solid (i.e., has a nonempty
 * interior), the origin must lie in the set interior and the inradius must be
 * positive. The origin is the center point of the convex set in its local
 * frame.
 *
 * @tparam dim Dimension of the convex set (2 or 3).
 */
template <int dim>
class ConvexSet {
 public:
  virtual ~ConvexSet() {}

  /**
   * @brief Computes the support function in the local frame.
   *
   * Implements the support function for the convex set \f$C\f$, which is given
   * by: \f{align*}{
   * s_v[C](n) & = \max_{x \in C} \langle n, x\rangle, \\
   * s_p[C](n) & \in \arg \max_{x \in C} \langle n, x\rangle,
   * \f}
   * where \f$s_v[C](\cdot)\f$ is the return value of the function.
   *
   * @note If the normal vector n is required to have unit 2-norm, the function
   * RequireUnitNormal should return true.
   *
   * @note Safety margins (in the 2-norm) can be directly included in the
   * support function computation (when n has unit 2-norm) as: \f{align*}{
   * s_v[C + B_m(0)](n) & = s_v[C](n) + m, \\
   * s_p[C + B_m(0)](n) & = s_p[C](n) + m \cdot n,
   * \f}
   * where \f$m\f$ is the safety margin. Note that the safety margin also
   * increases the inradius. A nonzero safety margin can result in slower
   * convergence for polytopes when using the cutting plane method.
   *
   * @see RequireUnitNormal
   *
   * @param[in]     n    Normal vector.
   * @param[out]    sp   Support point (any point at which the maximum for the
   *                     support function is attained).
   * @param[in,out] hint Additional hints.
   * @return        Support function value at the normal vector.
   */
  virtual Real SupportFunction(
      const Vecr<dim>& n, Vecr<dim>& sp,
      SupportFunctionHint<dim>* hint = nullptr) const = 0;

  /**
   * @brief Computes the support function and its higher-order derivatives.
   *
   * @see SupportFunctionDerivatives
   *
   * @param[in]     n     Normal vector.
   * @param[out]    deriv Support function derivatives.
   * @param[in,out] hint  Additional hints.
   * @return        Support function value at the normal vector.
   */
  virtual Real SupportFunction(const Vecr<dim>& n,
                               SupportFunctionDerivatives<dim>& deriv,
                               SupportFunctionHint<dim>* hint = nullptr) const;

  /**
   * @brief Returns the normalization requirement for the normal vector passed
   * to SupportFunction.
   *
   * If the return value is true, the argument n to SupportFunction will have
   * unit 2-norm.
   *
   * @see SupportFunction
   */
  virtual bool RequireUnitNormal() const;

  /**
   * @brief Computes the affine hull of the support patch and the span of the
   * normal cone at a base point-normal vector pair.
   *
   * @see NormalPair
   * @see SupportPatchHull
   * @see NormalConeSpan
   *
   * @param[in]  zn   Base point-normal vector pair.
   * @param[out] sph  Support patch affine hull.
   * @param[out] ncs  Normal cone span.
   * @param[in]  hint Additional hints for the base point.
   */
  virtual void ComputeLocalGeometry(
      const NormalPair<dim>& zn, SupportPatchHull<dim>& sph,
      NormalConeSpan<dim>& ncs, const BasePointHint<dim>* hint = nullptr) const;

  /**
   * @brief Computes the axis-aligned bounding box (AABB) of the convex set.
   *
   * @param[out] min Minimum corner of the bounding box.
   * @param[out] max Maximum corner of the bounding box.
   * @return     Diagonal length of the AABB.
   */
  virtual Real Bounds(Vecr<dim>* min = nullptr, Vecr<dim>* max = nullptr) const;

  /// @brief Returns true if the convex set is polytopic.
  virtual bool IsPolytopic() const;

  /// @brief Prints information about the convex set.
  virtual void PrintInfo() const;

  /// @brief Returns the dimension of the convex set.
  static constexpr int dimension();

  /// @brief Returns the support point differentiability tolerance.
  static Real eps_sp();

  /// @brief Returns the primal solution geometry tolerance.
  Real eps_p() const;

  /// @brief Sets the primal solution geometry tolerance.
  void set_eps_p(Real eps_p);

  /// @brief Returns the dual solution geometry tolerance.
  Real eps_d() const;

  /// @brief Sets the dual solution geometry tolerance.
  void set_eps_d(Real eps_d);

  /// @brief Returns the inradius of the convex set.
  Real inradius() const;

  /// @brief Sets the inradius of the convex set.
  void set_inradius(Real inradius);

 protected:
  ConvexSet();

  ConvexSet(Real inradius);

  /**
   * @brief Support point map differentiability tolerance.
   *
   * The support point map is considered differentiable at a unit normal vector
   * \f$n\f$ if the support point is unique and differentiable for all unit
   * normal vectors \f$n'\f$ such that \f{align*}{
   * s_v[C](n') - \langle n', s_p[C](n) \rangle < k \cdot \text{eps_sp_},
   * \f}
   * where \f$k > 1\f$ is a constant, \f$s_v[C](\cdot)\f$ is the support
   * function, and \f$s_p[C](\cdot)\f$ is the support point map.
   */
  static inline const Real eps_sp_ = dim * std::pow(kEps, Real(0.334));

  /// @brief Primal solution geometry tolerance.
  Real eps_p_ = dim * std::pow(kEps, Real(0.5));

  /// @brief Dual solution geometry tolerance.
  Real eps_d_ = dim * std::pow(kEps, Real(0.5));

  /**
   * @brief Convex set inradius at the origin.
   *
   * Radius of a ball that is centered at the origin and contained in the set.
   * Any nonnegative lower bound of the inradius will work. For solid sets, the
   * inradius must be positive. Note that larger values can help prevent
   * singularities in simplex computations and enable faster convergence.
   */
  Real inradius_;
};

template <int dim>
inline ConvexSet<dim>::ConvexSet() : ConvexSet(Real(0.0)) {}

template <int dim>
inline ConvexSet<dim>::ConvexSet(Real inradius) : inradius_(inradius) {
  if (inradius < Real(0.0)) {
    throw std::domain_error("Inradius is negative");
  }
}

template <int dim>
inline Real ConvexSet<dim>::SupportFunction(
    const Vecr<dim>& n, SupportFunctionDerivatives<dim>& deriv,
    SupportFunctionHint<dim>* hint) const {
  deriv.differentiable = false;
  return SupportFunction(n, deriv.sp, hint);
}

template <int dim>
inline bool ConvexSet<dim>::RequireUnitNormal() const {
  return true;
}

template <int dim>
inline void ConvexSet<dim>::ComputeLocalGeometry(
    const NormalPair<dim>& /*zn*/, SupportPatchHull<dim>& sph,
    NormalConeSpan<dim>& ncs, const BasePointHint<dim>* /*hint*/) const {
  sph.aff_dim = dim - 1;
  ncs.span_dim = dim;
}

template <int dim>
inline Real ConvexSet<dim>::Bounds(Vecr<dim>* min, Vecr<dim>* max) const {
  Vecr<dim> aabb_min, aabb_max, sp;
  for (int i = 0; i < dim; ++i) {
    Vecr<dim> n = Vecr<dim>::Zero();
    n(i) = Real(1.0);
    aabb_max(i) = SupportFunction(n, sp);
    aabb_min(i) = -SupportFunction(-n, sp);
  }
  if (min) *min = aabb_min;
  if (max) *max = aabb_max;
  return (aabb_max - aabb_min).norm();
}

template <int dim>
inline bool ConvexSet<dim>::IsPolytopic() const {
  return false;
}

template <int dim>
inline void ConvexSet<dim>::PrintInfo() const {
  std::cout << "Type: Abstract Convex Set (dim = " << dim << ")" << std::endl
            << "  Inradius: " << inradius_ << std::endl;
}

template <int dim>
constexpr int ConvexSet<dim>::dimension() {
  return dim;
}

template <int dim>
inline Real ConvexSet<dim>::eps_sp() {
  return eps_sp_;
}

template <int dim>
inline Real ConvexSet<dim>::eps_p() const {
  return eps_p_;
}

template <int dim>
inline void ConvexSet<dim>::set_eps_p(Real eps_p) {
  if (eps_p < Real(0.0)) {
    throw std::domain_error("Primal solution geometry tolerance is negative");
  }
  eps_p_ = eps_p;
}

template <int dim>
inline Real ConvexSet<dim>::eps_d() const {
  return eps_d_;
}

template <int dim>
inline void ConvexSet<dim>::set_eps_d(Real eps_d) {
  if (eps_d < Real(0.0)) {
    throw std::domain_error("Dual solution geometry tolerance is negative");
  }
  eps_d_ = eps_d;
}

template <int dim>
inline Real ConvexSet<dim>::inradius() const {
  return inradius_;
}

template <int dim>
inline void ConvexSet<dim>::set_inradius(Real inradius) {
  if (inradius < Real(0.0)) {
    throw std::domain_error("Inradius is negative");
  }
  inradius_ = inradius;
}

namespace detail {

/// @brief Convex set validator.
template <int dim, class C, bool check_dim = true>
struct ConvexSetValidator {
  static_assert((!check_dim) || (dim == 2) || (dim == 3),
                "Dimension must be 2 or 3");
  static_assert(std::is_base_of<ConvexSet<dim>, C>::value,
                "The convex set must inherit from ConvexSet");

  static constexpr bool valid = true;
};

}  // namespace detail

}  // namespace dgd

#endif  // DGD_GEOMETRY_CONVEX_SET_H_
