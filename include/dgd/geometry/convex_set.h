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
 * @tparam dim Dimension of the convex sets.
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
 * @tparam dim Dimension of the convex sets.
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

  /// @brief Returns true if the convex set is polytopic.
  virtual bool IsPolytopic() const;

  /// @brief Prints information about the convex set.
  virtual void PrintInfo() const;

  /// @brief Returns the dimension of the convex set.
  static constexpr int dimension();

  /// @brief Returns the inradius of the convex set.
  Real inradius() const;

  /// @brief Sets the inradius of the convex set.
  void set_inradius(Real inradius);

 protected:
  ConvexSet();

  ConvexSet(Real inradius);

  /**
   * @brief Support function differentiability tolerance.
   *
   * The support point map is considered differentiable at a unit normal vector
   * \f$n\f$ if the support point is unique and differentiable for all unit
   * normal vectors \f$n'\f$ such that \f{align*}{
   * s_v[C](n') - \langle n', s_p[C](n) \rangle < k \cdot \text{eps_diff()},
   * \f}
   * where \f$k > 1\f$ is a constant, \f$s_v[C](\cdot)\f$ is the support
   * function, and \f$s_p[C](\cdot)\f$ is the support point map.
   */
  static constexpr Real eps_diff();

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

template <int dim>
constexpr Real ConvexSet<dim>::eps_diff() {
  return dim * std::pow(kEps, Real(1.0 / 3.0));
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
