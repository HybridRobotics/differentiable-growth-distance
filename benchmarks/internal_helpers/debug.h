#ifndef DGD_BENCHMARKS_INTERNAL_HELPERS_DEBUG_H_
#define DGD_BENCHMARKS_INTERNAL_HELPERS_DEBUG_H_

#include <iomanip>
#include <iostream>
#include <limits>

#include "dgd/data_types.h"
#include "dgd/output.h"

namespace dgd {

namespace bench {

// Prints the growth distance benchmark statistics.
inline void PrintStatistics(double avg_solve_time, double max_prim_dual_gap,
                            double max_prim_infeas_err,
                            double max_dual_infeas_err, double avg_iter,
                            int nsubopt,
                            double* max_internal_prim_infeas_err = nullptr) {
  std::cout << "Avg. solve time (us)    : " << avg_solve_time << std::endl;
  std::cout << std::scientific;
  std::cout << "Max. prim dual gap      : " << max_prim_dual_gap << std::endl;
  std::cout << "Max. prim infeas err (m): " << max_prim_infeas_err << std::endl;
  if (max_internal_prim_infeas_err) {
    std::cout << "Max. prim infeas err (m): " << *max_internal_prim_infeas_err
              << "  (internal)" << std::endl;
  }
  std::cout << "Max. dual infeas err    : " << max_dual_infeas_err << std::endl;
  std::cout.unsetf(std::ios::fixed | std::ios::scientific);
  std::cout << "Avg. iterations         : " << avg_iter << std::endl;
  std::cout << "Num. suboptimal runs    : " << nsubopt << std::endl;
  std::cout << std::endl;
}

// Prints a formatted matrix.
template <int dim, bool csv = true>
inline void PrintMatrix(const Matr<dim, dim>& m) {
  const std::string prefix = csv ? "  " : "  (";
  const std::string suffix = csv ? "," : ")";
  for (int i = 0; i < dim; ++i) {
    std::cout << prefix;
    for (int j = 0; j < dim - 1; ++j) std::cout << m(i, j) << ", ";
    std::cout << m(i, dim - 1) << suffix << std::endl;
  }
}

// Prints a formatted vector.
template <int dim>
inline void PrintVector(const Vecr<dim>& v) {
  for (int i = 0; i < dim - 1; ++i) std::cout << v(i) << ", ";
  std::cout << v(dim - 1) << std::endl;
}

// Converts the solution status to string.
inline std::string StatusToString(SolutionStatus status) {
  if (status == SolutionStatus::CoincidentCenters) {
    return "Coincident centers";
  } else if (status == SolutionStatus::MaxIterReached) {
    return "Maximum iterations reached";
  } else {
    return "Optimal solution";
  }
}

// Returns a string for the warm start type.
inline std::string WarmStartTypeString(Settings settings) {
  if (settings.ws_type == WarmStartType::Primal) {
    return "Primal";
  } else if (settings.ws_type == WarmStartType::Dual) {
    return "Dual";
  } else {
    return "Unknown";
  }
}

// Prints the growth distance problem setup.
template <int dim>
void PrintSetup(const ConvexSet<dim>* set1, const Transformr<dim>& tf1,
                const ConvexSet<dim>* set2, const Transformr<dim>& tf2,
                const Settings& settings, const Output<dim>& out,
                const SolutionError& err, const Output<dim>* out_prev = nullptr,
                bool warm_start = false) {
  std::cout << "--- Solution error ---" << std::endl;
  constexpr int max_precision = std::numeric_limits<dgd::Real>::max_digits10;
  std::cout << std::fixed << std::setprecision(max_precision);
  if (warm_start) {
    std::cout << "Warm start" << std::endl;
  } else {
    std::cout << "Cold start" << std::endl;
  }
  std::cout << "Transform 1:" << std::endl;
  PrintMatrix(tf1);
  std::cout << "Transform 2:" << std::endl;
  PrintMatrix(tf2);
  std::cout << "Set 1: ";
  set1->PrintInfo();
  std::cout << "Set 2: ";
  set2->PrintInfo();
  std::cout << "Settings: " << std::endl
            << "  Max iter: " << settings.max_iter << std::endl
            << "  Rel tol: " << settings.rel_tol << std::endl
            << "  Min center dist: " << settings.min_center_dist << std::endl
            << "  Cutting plane iter: " << settings.cutting_plane_iter
            << std::endl
            << "  Warm start type: " << WarmStartTypeString(settings)
            << std::endl;
  std::cout << "Output: " << std::endl
            << "  Status: " << StatusToString(out.status) << std::endl
            << "  GD (lower): " << out.growth_dist_lb << std::endl
            << "  GD (upper): " << out.growth_dist_ub << std::endl
            << "  #Iter: " << out.iter << std::endl;
  std::cout << "  z1: ";
  PrintVector(out.z1);
  std::cout << "  z2: ";
  PrintVector(out.z2);
  std::cout << "  normal: ";
  PrintVector(out.normal);
  std::cout << "Error: " << std::endl
            << "  Primal-dual gap: " << err.prim_dual_gap << std::endl
            << "  Primal infeas err: " << err.prim_infeas_err << std::endl
            << "  Dual infeas err: " << err.dual_infeas_err << std::endl;
  if (warm_start && out_prev) {
    std::cout << "Previous output: " << std::endl
              << "  Status: " << StatusToString(out_prev->status) << std::endl
              << "  GD (lower): " << out_prev->growth_dist_lb << std::endl
              << "  GD (upper): " << out_prev->growth_dist_ub << std::endl
              << "  r1: " << out_prev->r1_ << std::endl
              << "  r2: " << out_prev->r2_ << std::endl
              << "  Normalization: " << out_prev->normalize_2norm_ << std::endl
              << "  s1:" << std::endl;
    PrintMatrix(out_prev->s1);
    std::cout << "  s2:" << std::endl;
    PrintMatrix(out_prev->s2);
    std::cout << "  bc: ";
    PrintVector(out_prev->bc);
    std::cout << "  Normal: ";
    PrintVector(out_prev->normal);
  }
  std::cout.unsetf(std::ios_base::fixed);
  std::cout << std::setprecision(6);
}

}  // namespace bench

}  // namespace dgd

#endif  // DGD_BENCHMARKS_INTERNAL_HELPERS_DEBUG_H_
