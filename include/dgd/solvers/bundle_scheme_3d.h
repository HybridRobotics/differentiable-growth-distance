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
 * @brief Bundle schemes for the three-dimensional growth distance problem.
 */

#ifndef DGD_SOLVERS_BUNDLE_SCHEME_3D_H_
#define DGD_SOLVERS_BUNDLE_SCHEME_3D_H_

// #include <cassert>
#include <Eigen/Dense>
#include <cmath>

// #include "eiquadprog/eiquadprog-rt.hpp"

#include "dgd/data_types.h"
#include "dgd/output.h"
#include "dgd/settings.h"
#include "dgd/solvers/debug.h"
#include "dgd/solvers/solver_options.h"
#include "dgd/solvers/solver_utils.h"

namespace dgd {

namespace detail {

/*
 * Local context.
 */

/// @brief Local context for the bundle scheme.
template <BcSolverType BST>
struct BundleScheme3Context;

template <>
struct BundleScheme3Context<BcSolverType::kCramer> {
  /**
   * @brief Simplex points in the aligned coordinates.
   *
   * The simplex points are in CCW order when projected to the x-y plane.
   */
  Matr<3, 3> s;
  /// @brief Barycentric coordinates.
  Vec3r bc;
  /// @brief Normal vector.
  Vec3r n;
  /// @brief Twice the signed area of the projected simplex (\f$> 0\f$).
  Real area;
};

template <>
struct BundleScheme3Context<BcSolverType::kLU> {
  /// @brief Simplex points in the aligned coordinates.
  Matr<3, 3> s;
  /**
   * @brief Edge vectors used to compute the barycentric coordinates and the
   * normal vector.
   */
  Matr<3, 2> e;
  /**
   * @brief Full pivot LU decomposition matrix.
   *
   * @note The upper triangular part is the U matrix, and lu(1, 0) is the L
   * element. lu(0, 0) is the inverse of the actual U(0, 0).
   */
  Matr<2, 2> lu;
  /// @brief Barycentric coordinates.
  Vec3r bc;
  /// @brief Normal vector.
  Vec3r n;
  /// @brief Twice the signed area of the projected triangle.
  Real area;
  /// @brief LU pivot row and column indices.
  int pr, pc, pri, pci;
};

/*
 * Initialization.
 */

/// @brief Initializes the simplex, normal vector, and barycentric coordinates.
template <BcSolverType BST>
void InitializeSimplex(BundleScheme3Context<BST>& bsc, Real r, Output<3>& out);

template <>
inline void InitializeSimplex<BcSolverType::kCramer>(
    BundleScheme3Context<BcSolverType::kCramer>& bsc, Real r, Output<3>& out) {
  bsc.s.col(0) = r * Vec3r(Real(0.5), Real(0.5), Real(0.0));
  bsc.s.col(1) = r * Vec3r(Real(-0.5), Real(0.5), Real(0.0));
  bsc.s.col(2) = Vec3r(Real(0.0), -r, Real(0.0));
  bsc.n = Vec3r::UnitZ();
  bsc.area = Real(1.5) * r * r;
  out.bc = Vec3r::Constant(Real(1.0 / 3.0));
}

template <>
inline void InitializeSimplex<BcSolverType::kLU>(
    BundleScheme3Context<BcSolverType::kLU>& bsc, Real r, Output<3>& out) {
  bsc.s.col(0) = r * Vec3r(Real(0.5), Real(0.5), Real(0.0));
  bsc.s.col(1) = r * Vec3r(Real(-0.5), Real(0.5), Real(0.0));
  bsc.s.col(2) = Vec3r(Real(0.0), -r, Real(0.0));
  bsc.lu(0, 0) = Real(2.0) / r;
  bsc.lu(1, 0) = Real(3.0);
  bsc.lu(0, 1) = -Real(0.5) * r;
  bsc.lu(1, 1) = Real(3.0) * r;
  bsc.n = Vec3r::UnitZ();
  bsc.area = Real(1.5) * r * r;
  bsc.pr = bsc.pc = 0;
  bsc.pri = bsc.pci = 1;
  out.bc = Vec3r::Constant(Real(1.0 / 3.0));
}

/**
 * @brief Initializes the convex set simplices corresponding to the Minkowski
 * difference set simplex.
 */
inline void InitializeSetSimplices(const MinkowskiDiffProp<3>& mdp,
                                   Output<3>& out) {
  out.s1.col(1).noalias() =
      out.r1_ * mdp.rot1.transpose() * Vec3r(Real(-0.5), Real(0.5), Real(0.0));
  out.s1.col(2) = -out.r1_ * mdp.rot1.transpose().col(1);
  out.s1.col(0) = -(out.s1.col(1) + out.s1.col(2));
  out.s2.col(1).noalias() =
      out.r2_ * mdp.rot2.transpose() * Vec3r(Real(0.5), Real(-0.5), Real(0.0));
  out.s2.col(2) = out.r2_ * mdp.rot2.transpose().col(1);
  out.s2.col(0) = -(out.s2.col(1) + out.s2.col(2));
}

/*
 * Cutting plane method functions.
 */

/**
 * @brief Computes the barycentric coordinates of a point with respect to the
 * projected simplex, assuming nondegeneracy.
 */
template <BcSolverType BST>
void ComputePointCoordinates(const Eigen::Ref<const Vec2r>& p,
                             BundleScheme3Context<BST>& bsc);

template <>
inline void ComputePointCoordinates<BcSolverType::kCramer>(
    const Eigen::Ref<const Vec2r>& p,
    BundleScheme3Context<BcSolverType::kCramer>& bsc) {
  bsc.bc(0) = ((bsc.s(0, 1) - p(0)) * (bsc.s(1, 2) - p(1)) -
               (bsc.s(1, 1) - p(1)) * (bsc.s(0, 2) - p(0)));
  bsc.bc(1) = ((bsc.s(0, 2) - p(0)) * (bsc.s(1, 0) - p(1)) -
               (bsc.s(1, 2) - p(1)) * (bsc.s(0, 0) - p(0)));
  bsc.bc.head<2>() /= bsc.area;
  bsc.bc(2) = Real(1.0) - (bsc.bc(0) + bsc.bc(1));
}

template <>
inline void ComputePointCoordinates<BcSolverType::kLU>(
    const Eigen::Ref<const Vec2r>& p,
    BundleScheme3Context<BcSolverType::kLU>& bsc) {
  // The threshold value below is the same as that used in Eigen's FullPivLU.
  if (std::abs(bsc.lu(0, 0) * bsc.lu(1, 1)) < Real(2.0) * kEps) {
    bsc.bc(bsc.pci) = Real(0.0);
    bsc.bc(bsc.pc) = p(bsc.pr) * bsc.lu(0, 0);
  } else {
    bsc.bc(bsc.pci) = (p(bsc.pri) - bsc.lu(1, 0) * p(bsc.pr)) / bsc.lu(1, 1);
    bsc.bc(bsc.pc) =
        (p(bsc.pr) - bsc.lu(0, 1) * bsc.bc(bsc.pci)) * bsc.lu(0, 0);
  }
  bsc.bc(2) = Real(1.0) - bsc.bc(0) - bsc.bc(1);
}

/**
 * @brief Updates the barycentric coordinates of the origin, and returns the
 * z-coordinate of the intersection point.
 */
template <BcSolverType BST>
Real UpdateOriginCoordinates(BundleScheme3Context<BST>& bsc, Vec3r& bc);

template <>
inline Real UpdateOriginCoordinates<BcSolverType::kCramer>(
    BundleScheme3Context<BcSolverType::kCramer>& bsc, Vec3r& bc) {
  // The projected (signed) simplex area is theoretically guaranteed to be
  // positive. However, some coordinates may become slightly negative.
  auto relu = [](Real x) -> Real { return std::max(x, Real(0.0)); };
  bc(0) = relu(bsc.s(0, 1) * bsc.s(1, 2) - bsc.s(1, 1) * bsc.s(0, 2));
  bc(1) = relu(bsc.s(0, 2) * bsc.s(1, 0) - bsc.s(1, 2) * bsc.s(0, 0));
  bc(2) = relu(bsc.s(0, 0) * bsc.s(1, 1) - bsc.s(1, 0) * bsc.s(0, 1));
  bsc.area = bc.sum();
  bc /= bsc.area;
  return bsc.s.row(2) * bc;
}

/**
 * @brief Projects the barycentric coordinates onto the feasible region (the
 * projection of the probability simplex) using the residual error metric, given
 * by \f$|(\text{bsc.s})_{-3} \cdot \text{bc}|_2^2\f$.
 */
inline void ProjectCoordinates(BundleScheme3Context<BcSolverType::kLU>& bsc,
                               Vec3r& bc) {
  auto project_edge = [&](const Eigen::Ref<const Vec2r>& eij, int i, int j,
                          int k) -> void {
    bc(i) = std::max(Real(0.0),
                     std::min(Real(1.0), -eij.dot(bsc.s.col(j).head<2>()) /
                                             eij.squaredNorm()));
    bc(j) = std::max(Real(0.0), Real(1.0) - bc(i));
    bc(k) = Real(0.0);
  };

  if (bc(0) < Real(0.0)) {
    if (bc(1) < Real(0.0)) {
      // Check vertex bc = (0, 0, 1).
      const Vec2r g = -bsc.e.topRows<2>().transpose() * bsc.s.col(2).head<2>();
      if ((g(0) < Real(0.0)) && (g(1) > Real(0.0))) {
        // Project onto bc(0) = 0.
        project_edge(bsc.e.col(1).head<2>(), 1, 2, 0);
      } else if ((g(0) > Real(0.0)) && (g(1) < Real(0.0))) {
        // Project onto bc(1) = 0.
        project_edge(bsc.e.col(0).head<2>(), 0, 2, 1);
      } else {
        bc = Vec3r::UnitZ();
      }
    } else {
      if (bc(2) < Real(0.0)) {
        // Check vertex bc = (0, 1, 0).
        const Vec2r g =
            -bsc.e.topRows<2>().transpose() * bsc.s.col(1).head<2>();
        if ((g(0) < Real(0.0)) && (g(1) < Real(0.0))) {
          // Project onto bc(0) = 0.
          project_edge(bsc.e.col(1).head<2>(), 1, 2, 0);
        } else if ((g(0) > Real(0.0)) && (g(0) > g(1))) {
          // Project onto bc(0) + bc(1) = 1.
          project_edge((bsc.s.col(0) - bsc.s.col(1)).head<2>(), 0, 1, 2);
        } else {
          bc = Vec3r::UnitY();
        }
      } else {
        // Project onto bc(0) = 0.
        project_edge(bsc.e.col(1).head<2>(), 1, 2, 0);
      }
    }
  } else if (bc(1) < Real(0.0)) {
    if (bc(2) < Real(0.0)) {
      // Check vertex bc = (1, 0, 0).
      const Vec2r g = -bsc.e.topRows<2>().transpose() * bsc.s.col(0).head<2>();
      if ((g(0) < Real(0.0)) && (g(1) < Real(0.0))) {
        // Project onto bc(1) = 0.
        project_edge(bsc.e.col(0).head<2>(), 0, 2, 1);
      } else if ((g(1) > Real(0.0)) && (g(1) > g(0))) {
        // Project onto bc(0) + bc(1) = 1.
        project_edge((bsc.s.col(0) - bsc.s.col(1)).head<2>(), 0, 1, 2);
      } else {
        bc = Vec3r::UnitX();
      }
    } else {
      // Project onto bc(1) = 0.
      project_edge(bsc.e.col(0).head<2>(), 0, 2, 1);
    }
  } else if (bc(2) < Real(0.0)) {
    // Project onto bc(0) + bc(1) = 1.
    project_edge((bsc.s.col(0) - bsc.s.col(1)).head<2>(), 0, 1, 2);
  }
}

/**
 * @brief Updates the barycentric coordinates of the origin and the edge
 * vectors when the projected simplex is degenerate, and returns the
 * z-coordinate of the intersection point.
 */
template <int ax>
Real UpdateOriginCoordinates1D(BundleScheme3Context<BcSolverType::kLU>& bsc,
                               Vec3r& bc) {
  auto update = [&bsc, &bc](Real len, int i, int j, int k) -> Real {
    // len is always greater than (or on the order of) rel_tol / inradius.
    bc(i) = std::abs(bsc.s(ax, j) / len);
    bc(j) = std::abs(Real(1.0) - bc(i));
    bc(k) = Real(0.0);
    return bsc.s.row(2) * bc;
  };

  if (std::signbit(bsc.s(ax, 0)) != std::signbit(bsc.s(ax, 2))) {
    // Edge 0-2 intersects the z-axis.
    // Check if vertex 1 is an improvement.
    const Real d = bsc.e(2, 0) * bsc.e(ax, 1) - bsc.e(ax, 0) * bsc.e(2, 1);
    if ((d > Real(0.0)) == std::signbit(bsc.s(ax, 0))) {
      // Vertex 1 is optimal.
      if (std::signbit(bsc.s(ax, 0)) == std::signbit(bsc.s(ax, 1))) {
        // Edge 1-2 is optimal.
        bsc.e.col(0) = Vec3r(bsc.e(1, 1), -bsc.e(0, 1), Real(0.0));
        return update(bsc.e(ax, 1), 1, 2, 0);
      } else {  // Edge 0-1 is optimal.
        bsc.e.col(0) = bsc.s.col(0) - bsc.s.col(1);
        bsc.e.col(1) = Vec3r(-bsc.e(1, 0), bsc.e(0, 0), Real(0.0));
        return update(bsc.e(ax, 0), 0, 1, 2);
      }
    } else {  // Edge 2-0 is optimal.
      bsc.e.col(1) = Vec3r(bsc.e(1, 0), -bsc.e(0, 0), Real(0.0));
      return update(bsc.e(ax, 0), 2, 0, 1);
    }
  } else if (std::signbit(bsc.s(ax, 1)) != std::signbit(bsc.s(ax, 2))) {
    // Edge 1-2 intersects the z-axis.
    // Check if vertex 0 is an improvement.
    const Real d = bsc.e(2, 1) * bsc.e(ax, 0) - bsc.e(2, 0) * bsc.e(ax, 1);
    if ((d > Real(0.0)) == std::signbit(bsc.s(ax, 1))) {
      // Vertex 0 is optimal.
      if (std::signbit(bsc.s(ax, 1)) == std::signbit(bsc.s(ax, 0))) {
        // Edge 2-0 is optimal.
        bsc.e.col(1) = Vec3r(-bsc.e(1, 0), bsc.e(0, 0), Real(0.0));
        return update(bsc.e(ax, 0), 2, 0, 1);
      } else {  // Edge 0-1 is optimal.
        bsc.e.col(0) = bsc.s.col(0) - bsc.s.col(1);
        bsc.e.col(1) = Vec3r(-bsc.e(1, 0), bsc.e(0, 0), Real(0.0));
        return update(bsc.e(ax, 0), 0, 1, 2);
      }
    } else {  // Edge 1-2 is optimal.
      bsc.e.col(0) = Vec3r(bsc.e(1, 1), -bsc.e(0, 1), Real(0.0));
      return update(bsc.e(ax, 1), 1, 2, 0);
    }
  } else {  // Degenerate case: The triangle does not intersect the z-axis.
    int idx;
    bsc.s.row(ax).cwiseAbs().minCoeff(&idx);
    bc(idx) = Real(1.0);
    bc(Inc<3>(idx)) = bc(Dec<3>(idx)) = Real(0.0);
    bsc.e = Matr<3, 3>::Identity().leftCols<2>();
    return bsc.s(2, idx);
  }
}

template <>
inline Real UpdateOriginCoordinates<BcSolverType::kLU>(
    BundleScheme3Context<BcSolverType::kLU>& bsc, Vec3r& bc) {
  // Compute triangle edges and signed area.
  bsc.e.col(0) = bsc.s.col(0) - bsc.s.col(2);
  bsc.e.col(1) = bsc.s.col(1) - bsc.s.col(2);
  bsc.area = bsc.e(0, 0) * bsc.e(1, 1) - bsc.e(1, 0) * bsc.e(0, 1);
  if (std::abs(bsc.area) > SolverSettings::kEpsArea3) {
    // The projected simplex is nondegenerate.
    // Compute the pivot row and column.
    bsc.pr = bsc.pc = 0;
    Real emax = std::abs(bsc.e(0, 0));
    if (std::abs(bsc.e(1, 0)) > emax) {
      emax = std::abs(bsc.e(1, 0));
      bsc.pr = 1;
    }
    if (std::abs(bsc.e(0, 1)) > emax) {
      bsc.pr = (std::abs(bsc.e(1, 1)) > std::abs(bsc.e(0, 1)));
      bsc.pc = 1;
    } else if (std::abs(bsc.e(1, 1)) > emax) {
      bsc.pr = bsc.pc = 1;
    }
    bsc.pri = Inv(bsc.pr);
    bsc.pci = Inv(bsc.pc);
    // Compute the LU decomposition matrix.
    // Note: |e(pr, pc)| is always greater than (or on the order of)
    // rel_tol / inradius.
    // Otherwise, the algorithm would have converged in the previous iteration.
    bsc.lu(0, 0) = Real(1.0) / bsc.e(bsc.pr, bsc.pc);
    bsc.lu(1, 0) = bsc.e(bsc.pri, bsc.pc) * bsc.lu(0, 0);
    bsc.lu(0, 1) = bsc.e(bsc.pr, bsc.pci);
    bsc.lu(1, 1) = bsc.e(bsc.pri, bsc.pci) - bsc.lu(1, 0) * bsc.lu(0, 1);
    // Solve for the barycentric coordinates.
    if (std::abs(bsc.lu(0, 0) * bsc.lu(1, 1)) < Real(2.0) * kEps) {
      bc(bsc.pci) = Real(0.0);
      bc(bsc.pc) = std::max(
          Real(0.0), std::min(Real(1.0), -bsc.s(bsc.pr, 2) * bsc.lu(0, 0)));
      bc(2) = std::max(Real(0.0), Real(1.0) - bc(0) - bc(1));
    } else {
      bc(bsc.pci) =
          (-bsc.s(bsc.pri, 2) + bsc.lu(1, 0) * bsc.s(bsc.pr, 2)) / bsc.lu(1, 1);
      bc(bsc.pc) =
          -(bsc.s(bsc.pr, 2) + bsc.lu(0, 1) * bc(bsc.pci)) * bsc.lu(0, 0);
      bc(2) = Real(1.0) - bc(0) - bc(1);
      ProjectCoordinates(bsc, bc);
    }
    return bsc.s.row(2) * bc;
  } else {  // The projected simplex is degenerate.
    if (bsc.e.row(0).lpNorm<Eigen::Infinity>() >
        bsc.e.row(1).lpNorm<Eigen::Infinity>()) {
      return UpdateOriginCoordinates1D<0>(bsc, bc);
    } else {
      return UpdateOriginCoordinates1D<1>(bsc, bc);
    }
  }
}

/**
 * @brief Updates the simplices and the barycentric coordinates and returns the
 * lower bound, given a support point.
 */
template <BcSolverType BST>
Real UpdateSimplex(const Vec3r& sp, const Vec3r& sp1, const Vec3r& sp2,
                   BundleScheme3Context<BST>& bsc, Output<3>& out,
                   int* idxn = nullptr);

template <>
inline Real UpdateSimplex<BcSolverType::kCramer>(
    const Vec3r& sp, const Vec3r& sp1, const Vec3r& sp2,
    BundleScheme3Context<BcSolverType::kCramer>& bsc, Output<3>& out,
    int* idxn) {
  int exiting_idx = 0;
  // The simplex is assumed to be nondegenerate when using Cramer's rule.
  ComputePointCoordinates(sp.head<2>(), bsc);
  // Perform one iteration of the Simplex algorithm.
  Real value = Real(1.0);
  for (int i = 0; i < 3; ++i) {
    if ((out.bc(i) < bsc.bc(i)) && (out.bc(i) < bsc.bc(i) * value)) {
      value = out.bc(i) / bsc.bc(i);
      exiting_idx = i;
    }
  }

  // Replace the exiting simplex point with the support point.
  bsc.s.col(exiting_idx) = sp;
  out.s1.col(exiting_idx) = sp1;
  out.s2.col(exiting_idx) = sp2;
  if (idxn) *idxn = exiting_idx;
  return UpdateOriginCoordinates(bsc, out.bc);
}

template <>
inline Real UpdateSimplex<BcSolverType::kLU>(
    const Vec3r& sp, const Vec3r& sp1, const Vec3r& sp2,
    BundleScheme3Context<BcSolverType::kLU>& bsc, Output<3>& out, int* idxn) {
  int exiting_idx = 0;
  if (std::abs(bsc.area) > SolverSettings::kEpsArea3) {
    ComputePointCoordinates((sp - bsc.s.col(2)).head<2>(), bsc);
    // Perform one iteration of the Simplex algorithm.
    Real value = Real(1.0);
    for (int i = 0; i < 3; ++i) {
      if ((out.bc(i) < bsc.bc(i)) && (out.bc(i) < bsc.bc(i) * value)) {
        value = out.bc(i) / bsc.bc(i);
        exiting_idx = i;
      }
    }
  } else {
    out.bc.minCoeff(&exiting_idx);
  }

  // Replace the exiting simplex point with the support point.
  bsc.s.col(exiting_idx) = sp;
  out.s1.col(exiting_idx) = sp1;
  out.s2.col(exiting_idx) = sp2;
  if (idxn) *idxn = exiting_idx;
  return UpdateOriginCoordinates(bsc, out.bc);
}

/// @brief Updates the normal vector for the cutting plane method.
template <BcSolverType BST>
void UpdateNormalCuttingPlane(const BundleScheme3Context<BST>& bsc, Vec3r& n);

template <>
inline void UpdateNormalCuttingPlane<BcSolverType::kCramer>(
    const BundleScheme3Context<BcSolverType::kCramer>& bsc, Vec3r& n) {
  // The simplex edges are used to compute the normal vector because the origin
  // may not exactly lie in the projected simplex interior.
  // n = (bsc.s.col(1) - bsc.s.col(0)).cross(bsc.s.col(2) - bsc.s.col(0));
  n(0) = (bsc.s(2, 2) - bsc.s(2, 0)) * (bsc.s(1, 1) - bsc.s(1, 0)) -
         (bsc.s(1, 2) - bsc.s(1, 0)) * (bsc.s(2, 1) - bsc.s(2, 0));
  n(1) = (bsc.s(0, 2) - bsc.s(0, 0)) * (bsc.s(2, 1) - bsc.s(2, 0)) -
         (bsc.s(2, 2) - bsc.s(2, 0)) * (bsc.s(0, 1) - bsc.s(0, 0));
  n(2) = bsc.area;
}

template <>
inline void UpdateNormalCuttingPlane<BcSolverType::kLU>(
    const BundleScheme3Context<BcSolverType::kLU>& bsc, Vec3r& n) {
  if (std::abs(bsc.area) > SolverSettings::kEpsArea3) {
    // Compute the normal vector using the simplex edges.
    n(0) = bsc.e(1, 0) * bsc.e(2, 1) - bsc.e(2, 0) * bsc.e(1, 1);
    n(1) = bsc.e(2, 0) * bsc.e(0, 1) - bsc.e(0, 0) * bsc.e(2, 1);
    n(2) = bsc.area;
  } else {
    n = bsc.e.col(0).cross(bsc.e.col(1));
  }
  if (n(2) < Real(0.0)) n = -n;
}

/*
 * Proximal bundle method functions.
 */

// /**
//  * @brief Updates the normal vector to the proximal point value.
//  *
//  * @note The cost matrix should be nonzero;
//  * otherwise, use the cutting plane update.
//  */
// template <BcSolverType BST>
// inline void UpdateNormalProximalPoint(const Matr<2, 2>& Q, const Vec3r& n_cp,
//                                       BundleScheme3Context<BST>& bsc) {
//   const Vec2r lmb_cp = n_cp.head<2>();
//   const Vec2r lmb = bsc.n.head<2>() / bsc.n(2) - lmb_cp;
//
//   // Solve min_x (c'x + 0.5 x'Px), s.t. Aineq x + bineq >= 0.
//   RtMatrixX<3, 3>::d P = kEps * Matr<3, 3>::Identity();
//   P.block<2, 2>(0, 0) = Q;
//   RtVectorX<3>::d c = Vec3r::Zero();
//   c.head<2>().noalias() = -(Q + kEps * Matr<2, 2>::Identity()) * lmb;
//   c(2) = Real(1.0);
//   const RtMatrixX<0, 3>::d Aeq;
//   const RtVectorX<0>::d beq(0);
//   RtMatrixX<3, 3>::d Aineq;
//   Aineq.leftCols<2>() = -bsc.s.topRows<2>().transpose();
//   Aineq.col(2) = Vec3r::Ones();
//   const RtVectorX<3>::d bineq = Vec3r::Zero();
//   RtVectorX<3>::d x = Vec3r::Zero();
//
//   {
//     using namespace eiquadprog::solvers;
//     RtEiquadprog<3, 0, 3> qp;
//     [[maybe_unused]] RtEiquadprog_status status;
//     status = qp.solve_quadprog(P, c, Aeq, beq, Aineq, bineq, x);
//     assert(status == RT_EIQUADPROG_OPTIMAL);
//   }
//
//   bsc.n(0) = static_cast<Real>(x(0)) + lmb_cp(0);
//   bsc.n(1) = static_cast<Real>(x(1)) + lmb_cp(1);
//   bsc.n(2) = Real(1.0);
// }

/*
 * Trust region Newton method functions.
 */

/**
 * @brief Updates the normal vector to the partial trust region Newton solution.
 */
template <BcSolverType BST>
inline void UpdateNormalPartialNewton(const Matr<2, 2>& hess, const Vec3r& n_cp,
                                      BundleScheme3Context<BST>& bsc, int idx) {
  // Compute the pseudoinverse of the cost matrix, assuming it is nonzero.
  Matr<2, 2> hess_inv;
  const Real det = hess.determinant();
  if (det > SolverSettings::kPinvTol3) {
    const Real det_inv = Real(1.0) / det;
    hess_inv(0, 0) = hess(1, 1) * det_inv;
    hess_inv(0, 1) = hess_inv(1, 0) = -hess(1, 0) * det_inv;
    hess_inv(1, 1) = hess(0, 0) * det_inv;
  } else {  // The Hessian is rank 1.
    if constexpr (SolverSettings::kSkipTrnIfSingularHess3) {
      bsc.n = n_cp;
      return;
    }
    const Real trace = hess.trace();
    hess_inv = hess / (trace * trace);
  }

  // Newton step solution.
  const Vec2r grad = bsc.s.col(idx).template head<2>();
  const Vec2r lmb = bsc.n.template head<2>() / bsc.n(2);
  const Vec2r lmb_opt = lmb - hess_inv * grad;
  // Residual error should be on the order of machine epsilon squared.
  Real res_err2 = Real(0.0);
  if constexpr (!SolverSettings::kSkipTrnIfSingularHess3) {
    res_err2 = (hess * (lmb_opt - lmb) + grad).squaredNorm();
  }
  // Check trust region feasibility and residual error.
  const Vec3r sv = bsc.s.template topRows<2>().transpose() * lmb_opt +
                   bsc.s.row(2).transpose();
  if ((res_err2 < SolverSettings::kPinvResErr3) &&
      (sv(idx) >= sv(Inc<3>(idx))) && (sv(idx) >= sv(Dec<3>(idx)))) {
    // Newton step solution lies within trust region bounds.
    bsc.n.template head<2>() = lmb_opt;
    bsc.n(2) = Real(1.0);
  } else {
    // Return the cutting plane solution.
    bsc.n = n_cp;
  }
}

// /**
//  * @brief Updates the normal vector to the full trust region Newton solution.
//  */
// template <BcSolverType BST>
// inline void UpdateNormalFullNewton(const Matr<2, 2>& hess, const Vec3r& n_cp,
//                                    BundleScheme3Context<BST>& bsc, int idx) {
//   const Vec2r lmb = bsc.n.head<2>() / bsc.n(2) - n_cp.head<2>();
//   const Vec2r grad = bsc.s.col(idx).head<2>() - hess * lmb;
//
//   // Setup trust region Newton QP problem.
//   const RtMatrixX<2, 2>::d Q =
//       hess + SolverSettings::kPinvTol3 * Matr<2, 2>::Identity();
//   const RtVectorX<2>::d c = grad - SolverSettings::kPinvTol3 * lmb;
//   const RtMatrixX<0, 2>::d Aeq;
//   const RtVectorX<0>::d beq(0);
//   RtMatrixX<2, 2>::d Aineq;
//   Aineq.row(0) = (bsc.s.col(idx) - bsc.s.col(Inc<3>(idx))).head<2>();
//   Aineq.row(1) = (bsc.s.col(idx) - bsc.s.col(Dec<3>(idx))).head<2>();
//   const RtVectorX<2>::d bineq = Vecr<2>::Zero();
//   RtVectorX<2>::d x = Vecr<2>::Zero();
//
//   // Solve problem.
//   {
//     using namespace eiquadprog::solvers;
//     RtEiquadprog<2, 0, 2> qp;
//     [[maybe_unused]] RtEiquadprog_status status =
//         qp.solve_quadprog(Q, c, Aeq, beq, Aineq, bineq, x);
//     assert(status == RT_EIQUADPROG_OPTIMAL);
//   }
//
//   const Vec2r lmb_opt =
//       Vec2r(static_cast<Real>(x(0)), static_cast<Real>(x(1)));
//   bsc.n.head<2>() = lmb_opt + n_cp.head<2>();
//   bsc.n(2) = Real(1.0);
// }

/// @brief Updates the normal vector for the trust region Newton method.
template <BcSolverType BST>
inline void UpdateNormalNewton(const Matr<2, 2>& hess, const Vec3r& n_cp,
                               BundleScheme3Context<BST>& bsc, int idx) {
  if constexpr (SolverSettings::kTrnLevel == TrustRegionNewtonLevel::kPartial) {
    UpdateNormalPartialNewton(hess, n_cp, bsc, idx);
  }  // else {
  //   UpdateNormalFullNewton(hess, n_cp, bsc, idx);
  // }
}

/*
 * Warm start.
 */

/**
 * @brief Warm starts the simplex and the normal vector and returns the lower
 * bound.
 */
template <BcSolverType BST, bool detect_collision>
inline Real PrimalWarmStart(const MinkowskiDiffProp<3>& mdp,
                            BundleScheme3Context<BST>& bsc, Output<3>& out) {
  const Matr<3, 3> s1 = out.s1, s2 = out.s2;
  Vec3r sp;
  Real lb = Real(0.0);

  InitializeSimplex(bsc, mdp.r, out);
  // if constexpr (detect_collision) InitializeSetSimplices(mdp, out);
  InitializeSetSimplices(mdp, out);
  for (int i = 0; i < 3; ++i) {
    // Simplex points having a small contribution are not considered.
    if (out.bc(i) > SolverSettings::kEpsMinBc) {
      sp.noalias() = mdp.rot1 * s1.col(i) - mdp.rot2 * s2.col(i);
      if (bsc.n.dot(sp - bsc.s.col(0)) > Real(0.0)) {
        lb = UpdateSimplex(sp, s1.col(i), s2.col(i), bsc, out);
        UpdateNormalCuttingPlane(bsc, bsc.n);
      }
    }
  }
  NormalizeNormal(bsc.n, out.normalize_2norm_);
  return lb;
}

/*
 * Debug printing.
 */

/// @brief Prints debugging information at an iteration of the algorithm.
inline void PrintDebugIteration(int iter, [[maybe_unused]] Real cdist, Real lb,
                                Real ub, Real rel_tol, const Matr<3, 3>& s,
                                const Vec3r& bc) {
  if constexpr (SolverSettings::kPrintGdBounds) {
    PrintDebugIteration(iter, cdist / lb, cdist / ub, rel_tol,
                        (s.topRows<2>() * bc).norm());
  } else {
    PrintDebugIteration(iter, ub, lb, rel_tol, (s.topRows<2>() * bc).norm());
  }
}

/*
 * General bundle scheme in 3D.
 */

/**
 * @brief Bundle scheme for the growth distance problem in 3D.
 *
 * @note When detecting collision, the output is 1.0 if the sets are colliding,
 * and -1.0 otherwise.
 */
template <class C1, class C2, SolverType S, BcSolverType BST,
          bool detect_collision>
Real BundleScheme(const C1* set1, const Transform3r& tf1, const C2* set2,
                  const Transform3r& tf2, const Settings& settings,
                  Output<3>& out, bool warm_start) {
  if constexpr ((S == SolverType::TrustRegionNewton) &&
                (SolverSettings::kTrnLevel == TrustRegionNewtonLevel::kFull)) {
    static_assert(always_false<S>::value,
                  "The full trust region Newton method for 3D is disabled");
  }
  if constexpr (S == SolverType::ProximalBundle) {
    static_assert(always_false<S>::value,
                  "The proximal bundle method for 3D is disabled");
  }

  if (!warm_start) InitializeOutput(set1, set2, out);

  MinkowskiDiffProp<3> mdp;
  // Check center distance.
  mdp.SetCenterDistance(tf1, tf2);
  if (mdp.cdist < settings.min_center_dist) return SetZeroOutput(tf1, tf2, out);
  // Set alignment rotation matrices.
  mdp.SetRotationMatrices(tf1, tf2);
  mdp.r = out.r1_ + out.r2_;

  // Support function output.
  SupportFunctionOutput<3, SolverOrder<S>()> sfo;
  // Local bundle scheme context.
  BundleScheme3Context<BST> bsc;
  // Growth distance bounds.
  Real lb = Real(0.0), ub = kInf, gd;
  // Other local variables.
  Matr<2, 2> Q;
  Vec3r n_cp;
  int iter = 0, idxn;
  bool update_lb;

  if (warm_start && (out.status == SolutionStatus::Optimal)) {
    // Warm start.
    if (settings.ws_type == WarmStartType::Primal) {
      lb = PrimalWarmStart<BST, detect_collision>(mdp, bsc, out);
    } else {  // Dual warm start.
      InitializeSimplex(bsc, mdp.r, out);
      InitializeSetSimplices(mdp, out);
      if ((bsc.n(2) = mdp.rot.row(2).dot(out.normal)) > Real(0.0)) {
        // Warm-started normal is dual feasible.
        bsc.n.template head<2>() = mdp.rot.topRows<2>() * out.normal;
      } else {
        bsc.n(2) = Real(1.0);
      }
    }
  } else {
    // Cold start.
    InitializeSimplex(bsc, mdp.r, out);
    // Note: There can be some edge case numerical issues with the primal
    // infeasibility error if the convex set simplices are not initialized.
    // These issues don't occur with the cutting plane method.
    // if constexpr (detect_collision) InitializeSetSimplices(mdp, out);
    InitializeSetSimplices(mdp, out);
  }
  if constexpr (S != SolverType::CuttingPlane) n_cp = Vec3r::UnitZ();

  if constexpr (SolverSettings::kEnableDebugPrinting) {
    PrintDebugHeader(SolverName<S>() + " (dim = 3)");
    PrintDebugIteration(iter, mdp.cdist, lb, ub, settings.rel_tol, bsc.s,
                        out.bc);
  }

  while (true) {
    // Evaluate the support functions at the normal.
    if constexpr (SolverOrder<S>() == 2) {
      if (iter < settings.cutting_plane_iter) {
        sfo.EvaluateFirstOrder(set1, set2, mdp, bsc.n, out);
      } else {
        sfo.Evaluate(set1, set2, mdp, bsc.n, out);
      }
    } else {
      sfo.Evaluate(set1, set2, mdp, bsc.n, out);
    }

    // Update the upper bound and the current best normal vector.
    const Real ub_new = (sfo.sv1 + sfo.sv2) / bsc.n(2);
    if (ub_new < ub) {
      ub = ub_new;
      out.normal = bsc.n;
    }

    // Update the lower bound and the simplex.
    if ((update_lb = !((iter == 0) && warm_start &&
                       (settings.ws_type == WarmStartType::Dual) &&
                       (sfo.sp(2) <= Real(0.0))))) {
      if constexpr (S == SolverType::CuttingPlane) {
        lb = UpdateSimplex(sfo.sp, sfo.sp1(), sfo.sp2(), bsc, out);
      } else {
        // Check if the lower bound can be improved; if not, skip the simplex
        // update, and update the normal to the cutting plane normal.
        if ((update_lb = ((iter < settings.cutting_plane_iter) ||
                          (n_cp.dot(sfo.sp - bsc.s.col(0)) > Real(0.0))))) {
          lb = UpdateSimplex(sfo.sp, sfo.sp1(), sfo.sp2(), bsc, out, &idxn);
        }
      }
    }

    ++iter;

    if constexpr (SolverSettings::kEnableDebugPrinting) {
      PrintDebugIteration(iter, mdp.cdist, lb, ub, settings.rel_tol, bsc.s,
                          out.bc);
    }

    // Termination criteria.
    if constexpr (detect_collision) {
      // Perform collision check.
      if (ub < mdp.cdist) {
        // No collision.
        ComputeDualSolution(mdp.rot, mdp.cdist, ub, out);
        out.status = SolutionStatus::Optimal;
        gd = Real(-1.0);
        break;
      } else if (lb >= mdp.cdist) {
        // Collision.
        ComputePrimalSolution(tf1, tf2, mdp.cdist, lb, out);
        out.status = SolutionStatus::Optimal;
        gd = Real(1.0);
        break;
      }
    } else {
      // Check primal-dual gap.
      if (ub <= lb * settings.rel_tol) {
        ComputePrimalSolution(tf1, tf2, mdp.cdist, lb, out);
        gd = ComputeDualSolution(mdp.rot, mdp.cdist, ub, out);
        out.status = SolutionStatus::Optimal;
        break;
      }
    }
    // Check number of iterations.
    if (iter >= settings.max_iter) {
      ComputePrimalSolution(tf1, tf2, mdp.cdist, lb, out);
      gd = ComputeDualSolution(mdp.rot, mdp.cdist, ub, out);
      out.status = SolutionStatus::MaxIterReached;
      break;
    }

    // Update the normal vector.
    if constexpr (S == SolverType::CuttingPlane) {
      if (update_lb) {
        UpdateNormalCuttingPlane(bsc, bsc.n);
      } else {
        // Lower bound was not updated; reset the normal vector.
        bsc.n = Vec3r::UnitZ();
        continue;
      }
    } else {
      UpdateNormalCuttingPlane(bsc, n_cp);
      if constexpr (S == SolverType::ProximalBundle) {
        if (update_lb) {
          Q = ComputeGammaProximalBundle(ub, mdp.r, iter) *
              Matr<2, 2>::Identity();
          n_cp /= n_cp(2);
          // UpdateNormalProximalPoint(Q, n_cp, bsc);
        } else {
          bsc.n = n_cp;
        }
      } else {  // TrustRegionNewton.
        if (update_lb && sfo.differentiable &&
            (bsc.n(2) * (sfo.Dsp(0, 0) + sfo.Dsp(1, 1)) >
             SolverSettings::kPinvTol3)) {
          Q = bsc.n(2) * sfo.Dsp.template block<2, 2>(0, 0);
          n_cp /= n_cp(2);
          UpdateNormalNewton(Q, n_cp, bsc, idxn);
        } else {
          bsc.n = n_cp;
        }
      }
    }
    NormalizeNormal(bsc.n, out.normalize_2norm_);
  }
  out.iter = iter;

  if constexpr (SolverSettings::kEnableDebugPrinting) PrintDebugFooter();

  // (test)
  // out.prim_infeas_err = (bsc.s.template topRows<2>() * out.bc).norm();

  return gd;
}

}  // namespace detail

}  // namespace dgd

#endif  // DGD_SOLVERS_BUNDLE_SCHEME_3D_H_
