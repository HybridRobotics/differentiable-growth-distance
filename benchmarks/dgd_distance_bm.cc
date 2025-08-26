#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "dgd/data_types.h"
#include "dgd/error_metrics.h"
#include "dgd/geometry/geometry_2d.h"
#include "dgd/geometry/geometry_3d.h"
#include "dgd/growth_distance.h"
#include "dgd/output.h"
#include "dgd/settings.h"
#include "dgd/utils/timer.h"
#include "internal_helpers/debug.h"
#include "internal_helpers/filesystem_utils.h"
#include "internal_helpers/random_utils.h"
#include "internal_helpers/set_generator.h"

// Benchmarks to run.
const bool dynamic_dispatch = false;
const bool cold_start = true;
const bool warm_start = true;
const bool two_dim_sets = true;
const bool three_dim_sets = true;
const bool trust_region_newton = true;

// Printing.
const bool print_suboptimal_run = true;
const bool exit_on_suboptimal_run = true;

// Constants.
const double position_lim = 5.0;
const double dx_max = 0.1, ang_max = dgd::kPi / 18.0;

//  Number of pairs of sets to benchmark.
const int npair = 1000;
//  Number of poses per set pair for cold-start.
const int npose_c = 100;
//  Number of poses per set pair for warm-start.
const int npose_w = 100;
//  Number of cold-start function calls for a given pair of sets and poses.
const int ncold = 100;
//  Number of warm-start function calls for a given pair of sets.
const int nwarm = 100;

template <class C>
using SetPtr = std::shared_ptr<C>;

using dgd::detail::SolverType;

template <int dim, class C1, class C2, SolverType S = SolverType::CuttingPlane>
void ColdStart(std::function<const SetPtr<C1>()> generator1,
               std::function<const SetPtr<C2>()> generator2, dgd::Rng& rng,
               int npair, int npose) {
  double avg_solve_time = 0.0;
  double max_prim_dual_gap = 0.0;
  double max_prim_infeas_err = 0.0;
  double max_dual_infeas_err = 0.0;
  double avg_iter = 0.0;
  int nsubopt = 0;

  dgd::Timer timer;
  timer.Stop();
  dgd::Transformr<dim> tf1, tf2;
  dgd::Settings settings;
  dgd::Output<dim> out;
  for (int i = 0; i < npair; ++i) {
    const SetPtr<C1> set1 = generator1();
    const SetPtr<C2> set2 = generator2();
    for (int j = 0; j < npose; ++j) {
      dgd::internal::SetRandomTransforms(rng, tf1, tf2, -position_lim,
                                         position_lim);
      // Initial call to reduce cache misses.
      dgd::GrowthDistanceImpl<dim, C1, C2, S>(set1.get(), tf1, set2.get(), tf2,
                                              settings, out, false);
      timer.Start();
      for (int k = 0; k < ncold; ++k) {
        dgd::GrowthDistanceImpl<dim, C1, C2, S>(set1.get(), tf1, set2.get(),
                                                tf2, settings, out, false);
      }
      timer.Stop();
      const auto err =
          dgd::ComputeSolutionError(set1.get(), tf1, set2.get(), tf2, out);

      if (print_suboptimal_run) {
        if (out.status == dgd::SolutionStatus::MaxIterReached) {
          dgd::internal::PrintSetup(set1.get(), tf1, set2.get(), tf2, settings,
                                    out, err);
          if (exit_on_suboptimal_run) exit(EXIT_FAILURE);
        }
      }

      avg_solve_time += timer.Elapsed() / double(ncold);
      max_prim_dual_gap = std::max(max_prim_dual_gap, err.prim_dual_gap);
      max_prim_infeas_err = std::max(max_prim_infeas_err, err.prim_infeas_err);
      max_dual_infeas_err = std::max(max_dual_infeas_err, err.dual_infeas_err);
      avg_iter += out.iter;
      nsubopt += (out.status != dgd::SolutionStatus::Optimal) &&
                 (out.status != dgd::SolutionStatus::CoincidentCenters);
    }
  }
  avg_solve_time = avg_solve_time / (npair * npose);
  avg_iter = avg_iter / (npair * npose);

  dgd::internal::PrintStatistics(avg_solve_time, max_prim_dual_gap,
                                 max_prim_infeas_err, max_dual_infeas_err,
                                 avg_iter, nsubopt);
}

template <int dim, class C1, class C2, SolverType S = SolverType::CuttingPlane>
void WarmStart(std::function<const SetPtr<C1>()> generator1,
               std::function<const SetPtr<C2>()> generator2, dgd::Rng& rng,
               int npair, int npose) {
  double avg_solve_time = 0.0;
  double max_prim_dual_gap = 0.0;
  double max_prim_infeas_err = 0.0;
  double max_dual_infeas_err = 0.0;
  double avg_iter = 0.0;
  int nsubopt = 0;

  dgd::Timer timer;
  timer.Stop();
  dgd::Transformr<dim> tf1, tf1_c, tf2;
  dgd::Vecr<dim> dx;
  dgd::Rotationr<dim> drot;
  dgd::Settings settings;
  dgd::Output<dim> out;
  std::vector<dgd::Output<dim>> outs(nwarm);
  for (int i = 0; i < npair; ++i) {
    const SetPtr<C1> set1 = generator1();
    const SetPtr<C2> set2 = generator2();
    for (int j = 0; j < npose; ++j) {
      dgd::internal::SetRandomTransforms(rng, tf1, tf2, -position_lim,
                                         position_lim);
      tf1_c = tf1;
      dgd::internal::SetRandomDisplacement(rng, dx, drot, dx_max, ang_max);
      // Initial cold-start call.
      dgd::GrowthDistanceImpl<dim, C1, C2, S>(set1.get(), tf1, set2.get(), tf2,
                                              settings, out, false);
      int total_iter = 0;
      timer.Start();
      for (int k = 0; k < nwarm; ++k) {
        dgd::internal::UpdateTransform(tf1, dx, drot);
        dgd::GrowthDistanceImpl<dim, C1, C2, S>(set1.get(), tf1, set2.get(),
                                                tf2, settings, out, true);
        outs[k] = out;
        total_iter += out.iter;
      }
      timer.Stop();
      avg_solve_time += timer.Elapsed() / double(nwarm);
      avg_iter += total_iter / double(nwarm);

      tf1 = tf1_c;
      for (int k = 0; k < nwarm; ++k) {
        dgd::internal::UpdateTransform(tf1, dx, drot);
        out = outs[k];

        const auto err =
            dgd::ComputeSolutionError(set1.get(), tf1, set2.get(), tf2, out);

        if (print_suboptimal_run) {
          if (out.status == dgd::SolutionStatus::MaxIterReached) {
            if (k == 0) {
              dgd::internal::PrintSetup<dim>(set1.get(), tf1, set2.get(), tf2,
                                             settings, out, err);
            } else {
              dgd::Output<dim> out_prev;
              out_prev = outs[k - 1];
              dgd::internal::PrintSetup(set1.get(), tf1, set2.get(), tf2,
                                        settings, out, err, &out_prev, true);
            }
            if (exit_on_suboptimal_run) exit(EXIT_FAILURE);
          }
        }

        max_prim_dual_gap = std::max(max_prim_dual_gap, err.prim_dual_gap);
        max_prim_infeas_err =
            std::max(max_prim_infeas_err, err.prim_infeas_err);
        max_dual_infeas_err =
            std::max(max_dual_infeas_err, err.dual_infeas_err);
        nsubopt += (out.status != dgd::SolutionStatus::Optimal) &&
                   (out.status != dgd::SolutionStatus::CoincidentCenters);
      }
    }
  }
  avg_solve_time = avg_solve_time / (npair * npose);
  avg_iter = avg_iter / (npair * npose);

  dgd::internal::PrintStatistics(avg_solve_time, max_prim_dual_gap,
                                 max_prim_infeas_err, max_dual_infeas_err,
                                 avg_iter, nsubopt);
}

int main(int argc, char** argv) {
  std::string path = "./";
  if (argc < 2) {
    std::cout << "No directory specified; using current directory" << std::endl;
  } else if (argc == 2) {
    path = argv[1];
    if (!dgd::internal::IsValidDirectory(path)) return EXIT_FAILURE;
  } else {
    std::cerr << "Usage: " << argv[0] << " <asset_folder_path>" << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<std::string> filenames;
  dgd::internal::GetObjFileNames(path, filenames);

  dgd::internal::ConvexSetGenerator gen(dgd::internal::ConvexSetFeatureRange{});
  gen.LoadMeshesFromObjFiles(filenames);
  if (gen.nmeshes() == 0) {
    std::cout << "No .obj files found; mesh benchmarks will not be run"
              << std::endl;
  }
  dgd::Rng rng;
  auto set_default_seed = [&gen, &rng]() -> void {
    gen.SetDefaultRngSeed();
    rng.SetDefaultSeed();
  };
  auto set_random_seed = [&gen, &rng]() -> void {
    gen.SetRandomRngSeed();
    rng.SetRandomSeed();
  };

  // Benchmarks.
  //  1a. Dynamic dispatch: (cutting plane, cold-start, frustum + ellipsoid).
  if (dynamic_dispatch) {
    set_default_seed();
    auto generator1 = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetPrimitiveSet(dgd::internal::CurvedPrimitive3D::Frustum);
    };
    auto generator2 = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetPrimitiveSet(dgd::internal::CurvedPrimitive3D::Ellipsoid);
    };
    std::cout
        << "Dynamic dispatch: (cutting plane, cold-start, frustum + ellipsoid)"
        << std::endl;
    ColdStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(generator1, generator2,
                                                       rng, npair, npose_c);
  }
  //  1b. Inline call: (cutting plane, cold-start, frustum + ellipsoid).
  if (dynamic_dispatch) {
    set_default_seed();
    auto generator1 = [&gen]() -> const SetPtr<dgd::Frustum> {
      const auto set1 =
          gen.GetPrimitiveSet(dgd::internal::CurvedPrimitive3D::Frustum);
      return std::static_pointer_cast<dgd::Frustum>(set1);
    };
    auto generator2 = [&gen]() -> const SetPtr<dgd::Ellipsoid> {
      const auto set2 =
          gen.GetPrimitiveSet(dgd::internal::CurvedPrimitive3D::Ellipsoid);
      return std::static_pointer_cast<dgd::Ellipsoid>(set2);
    };
    std::cout
        << "Inline call     : (cutting plane, cold-start, frustum + ellipsoid)"
        << std::endl;
    ColdStart<3, dgd::Frustum, dgd::Ellipsoid>(generator1, generator2, rng,
                                               npair, npose_c);
  }

  //  2a. Dynamic dispatch: (cutting plane, cold-start, mesh + mesh).
  if (dynamic_dispatch && (gen.nmeshes() > 0)) {
    set_default_seed();
    auto generator = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetRandomMeshSet();
    };
    std::cout << "Dynamic dispatch: (cutting plane, cold-start, mesh + mesh)"
              << std::endl;
    ColdStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(generator, generator,
                                                       rng, npair, npose_c);
  }
  //  2b. Inline call: (cutting plane, cold-start, mesh + mesh).
  if (dynamic_dispatch && (gen.nmeshes() > 0)) {
    set_default_seed();
    auto generator = [&gen]() -> const SetPtr<dgd::Mesh> {
      return std::static_pointer_cast<dgd::Mesh>(gen.GetRandomMeshSet());
    };
    std::cout << "Inline call     : (cutting plane, cold-start, mesh + mesh)"
              << std::endl;
    ColdStart<3, dgd::Mesh, dgd::Mesh>(generator, generator, rng, npair,
                                       npose_c);
  }

  std::cout << std::string(50, '-') << std::endl;
  set_random_seed();
  //  3a. Cold-start: (cutting plane, 2D primitive + 2D primitive).
  if (cold_start && two_dim_sets) {
    auto generator = [&gen]() -> const SetPtr<dgd::ConvexSet<2>> {
      return gen.GetRandom2DSet();
    };
    std::cout << "Cold-start: (cutting plane, 2D primitive + 2D primitive)"
              << std::endl;
    ColdStart<2, dgd::ConvexSet<2>, dgd::ConvexSet<2>>(generator, generator,
                                                       rng, npair, npose_c);
  }
  //  3b. Cold-start: (trust region Newton, 2D primitive + 2D primitive).
  if (cold_start && two_dim_sets && trust_region_newton) {
    auto generator = [&gen]() -> const SetPtr<dgd::ConvexSet<2>> {
      return gen.GetRandom2DSet();
    };
    std::cout
        << "Cold-start: (trust region Newton, 2D primitive + 2D primitive)"
        << std::endl;
    ColdStart<2, dgd::ConvexSet<2>, dgd::ConvexSet<2>,
              SolverType::TrustRegionNewton>(generator, generator, rng, npair,
                                             npose_c);
  }

  //  4a. Cold-start: (cutting plane, 3D curved primitive + 3D curved
  //  primitive).
  if (cold_start && three_dim_sets) {
    auto generator = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetRandomCurvedPrimitive3DSet();
    };
    std::cout << "Cold-start: (cutting plane, 3D curved primitive + 3D curved "
                 "primitive)"
              << std::endl;
    ColdStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(generator, generator,
                                                       rng, npair, npose_w);
  }
  //  4b. Cold-start: (trust region Newton, 3D curved primitive + 3D curved
  //  primitive).
  if (cold_start && three_dim_sets && trust_region_newton) {
    auto generator = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetRandomCurvedPrimitive3DSet();
    };
    std::cout << "Cold-start: (trust region Newton, 3D curved primitive + 3D "
                 "curved primitive)"
              << std::endl;
    ColdStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>,
              SolverType::TrustRegionNewton>(generator, generator, rng, npair,
                                             npose_w);
  }

  //  5a. Cold-start: (cutting plane, 3D primitive + 3D primitive).
  if (cold_start && three_dim_sets) {
    auto generator = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetRandomPrimitive3DSet();
    };
    std::cout << "Cold-start: (cutting plane, 3D primitive + 3D primitive)"
              << std::endl;
    ColdStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(generator, generator,
                                                       rng, npair, npose_w);
  }
  //  5b. Cold-start: (trust region Newton, 3D primitive + 3D primitive).
  if (cold_start && three_dim_sets && trust_region_newton) {
    auto generator = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetRandomPrimitive3DSet();
    };
    std::cout
        << "Cold-start: (trust region Newton, 3D primitive + 3D primitive)"
        << std::endl;
    ColdStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>,
              SolverType::TrustRegionNewton>(generator, generator, rng, npair,
                                             npose_w);
  }

  //  6a. Cold-start: (cutting plane, mesh + mesh).
  if (cold_start && three_dim_sets && (gen.nmeshes() > 0)) {
    auto generator = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetRandomMeshSet();
    };
    std::cout << "Cold-start: (cutting plane, mesh + mesh)" << std::endl;
    ColdStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(generator, generator,
                                                       rng, npair, npose_c);
  }
  //  6b. Cold-start: (trust region Newton, mesh + mesh).
  if (cold_start && three_dim_sets && trust_region_newton &&
      (gen.nmeshes() > 0)) {
    auto generator = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetRandomMeshSet();
    };
    std::cout << "Cold-start: (trust region Newton, mesh + mesh)" << std::endl;
    ColdStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>,
              SolverType::TrustRegionNewton>(generator, generator, rng, npair,
                                             npose_c);
  }

  std::cout << std::string(50, '-') << std::endl;
  //  7a. Warm-start: (cutting plane, 2D primitive + 2D primitive).
  if (warm_start && two_dim_sets) {
    auto generator = [&gen]() -> const SetPtr<dgd::ConvexSet<2>> {
      return gen.GetRandom2DSet();
    };
    std::cout << "Warm-start: (cutting plane, 2D primitive + 2D primitive)"
              << std::endl;
    WarmStart<2, dgd::ConvexSet<2>, dgd::ConvexSet<2>>(generator, generator,
                                                       rng, npair, npose_w);
  }
  //  7b. Warm-start: (trust region Newton, 2D primitive + 2D primitive).
  if (warm_start && two_dim_sets && trust_region_newton) {
    auto generator = [&gen]() -> const SetPtr<dgd::ConvexSet<2>> {
      return gen.GetRandom2DSet();
    };
    std::cout
        << "Warm-start: (trust region Newton, 2D primitive + 2D primitive)"
        << std::endl;
    WarmStart<2, dgd::ConvexSet<2>, dgd::ConvexSet<2>,
              SolverType::TrustRegionNewton>(generator, generator, rng, npair,
                                             npose_w);
  }

  //  8a. Warm-start: (cutting plane, 3D curved primitive + 3D curved
  //  primitive).
  if (warm_start && three_dim_sets) {
    auto generator = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetRandomCurvedPrimitive3DSet();
    };
    std::cout << "Warm-start: (cutting plane, 3D curved primitive + 3D curved "
                 "primitive)"
              << std::endl;
    WarmStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(generator, generator,
                                                       rng, npair, npose_w);
  }
  //  8b. Warm-start: (trust region Newton, 3D curved primitive + 3D curved
  //  primitive).
  if (warm_start && three_dim_sets && trust_region_newton) {
    auto generator = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetRandomCurvedPrimitive3DSet();
    };
    std::cout << "Warm-start: (trust region Newton, 3D curved primitive + 3D "
                 "curved primitive)"
              << std::endl;
    WarmStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>,
              SolverType::TrustRegionNewton>(generator, generator, rng, npair,
                                             npose_w);
  }

  //  9a. Warm-start: (cutting plane, mesh + mesh).
  if (warm_start && three_dim_sets && (gen.nmeshes() > 0)) {
    auto generator = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetRandomMeshSet();
    };
    std::cout << "Warm-start: (cutting plane, mesh + mesh)" << std::endl;
    WarmStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(generator, generator,
                                                       rng, npair, npose_w);
  }
  //  9b. Warm-start: (trust region Newton, mesh + mesh).
  if (warm_start && three_dim_sets && (gen.nmeshes() > 0) &&
      trust_region_newton) {
    auto generator = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetRandomMeshSet();
    };
    std::cout << "Warm-start: (trust region Newton, mesh + mesh)" << std::endl;
    WarmStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>,
              SolverType::TrustRegionNewton>(generator, generator, rng, npair,
                                             npose_w);
  }
}
