#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "dgd/dgd.h"
#include "dgd/error_metrics.h"
#include "dgd/geometry/geometry_2d.h"
#include "dgd/geometry/geometry_3d.h"
#include "dgd/utils/timer.h"
#include "internal_helpers/debug.h"
#include "internal_helpers/filesystem_utils.h"
#include "internal_helpers/math_utils.h"
#include "internal_helpers/set_generator.h"

// Constants.
const double position_lim = 5.0;
const double dx_max = 0.1, ang_max = dgd::kPi / 18.0;

// Configuration.
struct Config {
  // Mesh asset path
  std::string mesh_asset_path = "./";  // -map

  //  Number of pairs of sets to benchmark.
  int npair = 100;  // -np
  //  Number of poses per set pair for cold-start.
  int ncpose = 100;  // -nc
  //  Number of poses per set pair for warm-start.
  int nwpose = 100;  // -nw
  //  Number of cold-start function calls for a given pair of sets and poses.
  int ncold = 100;  // -ncc
  //  Number of warm-start function calls for a given pair of sets.
  int nwarm = 100;  // -nwc

  // Benchmarks to run.
  bool dynamic_dispatch = false;    // -dd
  bool cold_start = true;           // -nocs
  bool warm_start = true;           // -nows
  bool two_dim_sets = true;         // -no2d
  bool three_dim_sets = true;       // -no3d
  bool trust_region_newton = true;  // -notrn

  // Printing.
  bool print_failure = true;  // -nofp
  bool exit_failure = true;   // -nofe
};

void ParseConfig(int argc, char* argv[], Config& config) {
  std::vector<std::string_view> args(argv + 1, argv + argc);

  for (size_t i = 0; i < args.size(); ++i) {
    const auto& arg = args[i];

    if (arg == "-map" || arg == "--mesh-asset-path") {
      if (i + 1 < args.size()) {
        config.mesh_asset_path = std::string(args[++i]);
        if (!dgd::bench::IsValidDirectory(config.mesh_asset_path)) {
          std::cerr << "Invalid dir.: " << config.mesh_asset_path << std::endl;
          exit(EXIT_FAILURE);
        }
      }
    } else if (arg == "-np" || arg == "--pairs") {
      if (i + 1 < args.size()) {
        config.npair = std::stoi(std::string(args[++i]));
      }
    } else if (arg == "-nc" || arg == "--cold-poses") {
      if (i + 1 < args.size()) {
        config.ncpose = std::stoi(std::string(args[++i]));
      }
    } else if (arg == "-nw" || arg == "--warm-poses") {
      if (i + 1 < args.size()) {
        config.nwpose = std::stoi(std::string(args[++i]));
      }
    } else if (arg == "-ncc" || arg == "--cold-calls") {
      if (i + 1 < args.size()) {
        config.ncold = std::stoi(std::string(args[++i]));
      }
    } else if (arg == "-nwc" || arg == "--warm-calls") {
      if (i + 1 < args.size()) {
        config.nwarm = std::stoi(std::string(args[++i]));
      }
    } else if (arg == "-dd" || arg == "--dynamic-dispatch") {
      config.dynamic_dispatch = true;
    } else if (arg == "-nocs" || arg == "--no-cold-start") {
      config.cold_start = false;
    } else if (arg == "-nows" || arg == "--no-warm-start") {
      config.warm_start = false;
    } else if (arg == "-no2d" || arg == "--no-2d-sets") {
      config.two_dim_sets = false;
    } else if (arg == "-no3d" || arg == "--no-3d-sets") {
      config.three_dim_sets = false;
    } else if (arg == "-notrn" || arg == "--no-trust-region-newton") {
      config.trust_region_newton = false;
    } else if (arg == "-nofp" || arg == "--no-failure-print") {
      config.print_failure = false;
    } else if (arg == "-nofe" || arg == "--no-failure-exit") {
      config.exit_failure = false;
    }
  }
}

template <class C>
using SetPtr = std::shared_ptr<C>;

using dgd::BcSolverType;
using dgd::SolverType;
using dgd::WarmStartType;

template <int dim, class C1, class C2>
void ColdStart(std::function<const SetPtr<C1>()> gen1,
               std::function<const SetPtr<C2>()> gen2, dgd::Rng& rng,
               const Config& config,
               const SolverType& solver = SolverType::CuttingPlane) {
  double avg_solve_time = 0.0;
  double max_prim_dual_gap = 0.0;
  double max_prim_infeas_err = 0.0;
  double max_internal_prim_infeas_err = 0.0;
  double max_dual_infeas_err = 0.0;
  double avg_iter = 0.0;
  int nsubopt = 0;

  auto GrowthDistanceImpl =
      (solver == SolverType::CuttingPlane)
          ? dgd::GrowthDistanceCp<dim, C1, C2, BcSolverType::kCramer>
          : dgd::GrowthDistanceTrn<dim, C1, C2>;

  dgd::Timer timer;
  timer.Stop();
  dgd::Transformr<dim> tf1, tf2;
  dgd::Settings settings;
  dgd::Output<dim> out;
  for (int i = 0; i < config.npair; ++i) {
    const SetPtr<C1> set1 = gen1();
    const SetPtr<C2> set2 = gen2();
    for (int j = 0; j < config.ncpose; ++j) {
      dgd::bench::SetRandomTransforms(rng, tf1, tf2, -position_lim,
                                      position_lim);
      // Initial call to reduce cache misses.
      GrowthDistanceImpl(set1.get(), tf1, set2.get(), tf2, settings, out,
                         false);
      timer.Start();
      for (int k = 0; k < config.ncold; ++k) {
        GrowthDistanceImpl(set1.get(), tf1, set2.get(), tf2, settings, out,
                           false);
      }
      timer.Stop();
      const auto err =
          dgd::ComputeSolutionError(set1.get(), tf1, set2.get(), tf2, out);

      if (config.print_failure) {
        if (out.status == dgd::SolutionStatus::MaxIterReached) {
          dgd::bench::PrintSetup(set1.get(), tf1, set2.get(), tf2, settings,
                                 out, err);
          if (config.exit_failure) exit(EXIT_FAILURE);
        }
      }

      avg_solve_time += timer.Elapsed() / double(config.ncold);
      max_prim_dual_gap = std::max(max_prim_dual_gap, err.prim_dual_gap);
      max_prim_infeas_err = std::max(max_prim_infeas_err, err.prim_infeas_err);
      // (test)
      // max_internal_prim_infeas_err = std::max(max_internal_prim_infeas_err,
      // out.prim_infeas_err);
      max_dual_infeas_err = std::max(max_dual_infeas_err, err.dual_infeas_err);
      avg_iter += out.iter;
      nsubopt += (out.status != dgd::SolutionStatus::Optimal) &&
                 (out.status != dgd::SolutionStatus::CoincidentCenters);
    }
  }
  avg_solve_time = avg_solve_time / (config.npair * config.ncpose);
  avg_iter = avg_iter / (config.npair * config.ncpose);

  dgd::bench::PrintStatistics(avg_solve_time, max_prim_dual_gap,
                              max_prim_infeas_err, max_dual_infeas_err,
                              avg_iter, nsubopt, &max_internal_prim_infeas_err);
}

template <int dim, class C1, class C2>
void WarmStart(std::function<const SetPtr<C1>()> gen1,
               std::function<const SetPtr<C2>()> gen2, dgd::Rng& rng,
               const Config& config,
               const WarmStartType& ws_type = WarmStartType::Primal,
               const SolverType& solver = SolverType::CuttingPlane) {
  double avg_solve_time = 0.0;
  double max_prim_dual_gap = 0.0;
  double max_prim_infeas_err = 0.0;
  double max_internal_prim_infeas_err = 0.0;
  double max_dual_infeas_err = 0.0;
  double avg_iter = 0.0;
  int nsubopt = 0;

  auto GrowthDistanceImpl =
      (solver == SolverType::CuttingPlane)
          ? dgd::GrowthDistanceCp<dim, C1, C2, BcSolverType::kCramer>
          : dgd::GrowthDistanceTrn<dim, C1, C2>;

  dgd::Timer timer;
  timer.Stop();
  dgd::Transformr<dim> tf1, tf1_c, tf2;
  dgd::Vecr<dim> dx;
  dgd::Rotationr<dim> drot;
  dgd::Settings settings;
  settings.ws_type = ws_type;
  dgd::Output<dim> out;
  std::vector<dgd::Output<dim>> outs(config.nwarm);
  for (int i = 0; i < config.npair; ++i) {
    const SetPtr<C1> set1 = gen1();
    const SetPtr<C2> set2 = gen2();
    for (int j = 0; j < config.nwpose; ++j) {
      dgd::bench::SetRandomTransforms(rng, tf1, tf2, -position_lim,
                                      position_lim);
      tf1_c = tf1;
      dgd::bench::SetRandomDisplacement(rng, dx, drot, dx_max, ang_max);
      // Initial cold-start call.
      GrowthDistanceImpl(set1.get(), tf1, set2.get(), tf2, settings, out,
                         false);
      int total_iter = 0;
      timer.Start();
      for (int k = 0; k < config.nwarm; ++k) {
        dgd::bench::UpdateTransform(tf1, dx, drot);
        GrowthDistanceImpl(set1.get(), tf1, set2.get(), tf2, settings, out,
                           true);
        outs[k] = out;
        total_iter += out.iter;
      }
      timer.Stop();
      avg_solve_time += timer.Elapsed() / double(config.nwarm);
      avg_iter += total_iter / double(config.nwarm);

      tf1 = tf1_c;
      for (int k = 0; k < config.nwarm; ++k) {
        dgd::bench::UpdateTransform(tf1, dx, drot);
        out = outs[k];

        const auto err =
            dgd::ComputeSolutionError(set1.get(), tf1, set2.get(), tf2, out);

        if (config.print_failure) {
          if (out.status == dgd::SolutionStatus::MaxIterReached) {
            if (k == 0) {
              dgd::bench::PrintSetup(set1.get(), tf1, set2.get(), tf2, settings,
                                     out, err);
            } else {
              dgd::Output<dim> out_prev;
              out_prev = outs[k - 1];
              dgd::bench::PrintSetup(set1.get(), tf1, set2.get(), tf2, settings,
                                     out, err, &out_prev, true);
            }
            if (config.exit_failure) exit(EXIT_FAILURE);
          }
        }

        max_prim_dual_gap = std::max(max_prim_dual_gap, err.prim_dual_gap);
        max_prim_infeas_err =
            std::max(max_prim_infeas_err, err.prim_infeas_err);
        // (test)
        // max_internal_prim_infeas_err =
        // std::max(max_internal_prim_infeas_err, out.prim_infeas_err);
        max_dual_infeas_err =
            std::max(max_dual_infeas_err, err.dual_infeas_err);
        nsubopt += (out.status != dgd::SolutionStatus::Optimal) &&
                   (out.status != dgd::SolutionStatus::CoincidentCenters);
      }
    }
  }
  avg_solve_time = avg_solve_time / (config.npair * config.nwpose);
  avg_iter = avg_iter / (config.npair * config.nwpose);

  dgd::bench::PrintStatistics(avg_solve_time, max_prim_dual_gap,
                              max_prim_infeas_err, max_dual_infeas_err,
                              avg_iter, nsubopt, &max_internal_prim_infeas_err);
}

int main(int argc, char** argv) {
  Config config;
  ParseConfig(argc, argv, config);

  const auto filenames = dgd::bench::GetObjFileNames(config.mesh_asset_path);

  dgd::bench::ConvexSetFeatureRange fr{};
  dgd::bench::ConvexSetGenerator gen(fr);
  gen.LoadMeshesFromObjFiles(filenames);
  if (gen.nmeshes() == 0) {
    std::cout << "No .obj files found; mesh benchmarks will not be run"
              << std::endl;
  }
  dgd::Rng rng;
  auto set_default_seed = [&gen, &rng]() -> void {
    gen.SetRngSeed();
    rng.SetSeed();
  };
  auto set_random_seed = [&gen, &rng]() -> void {
    gen.SetRandomRngSeed();
    rng.SetRandomSeed();
  };

  // Benchmarks.
  //  1a. Dynamic dispatch: (cutting plane, cold-start, frustum + ellipsoid).
  if (config.dynamic_dispatch) {
    set_default_seed();
    auto gen1 = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetPrimitiveSet(dgd::bench::CurvedPrimitive3D::Frustum);
    };
    auto gen2 = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetPrimitiveSet(dgd::bench::CurvedPrimitive3D::Ellipsoid);
    };
    std::cout
        << "Dynamic dispatch: (cutting plane, cold-start, frustum + ellipsoid)"
        << std::endl;
    ColdStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(gen1, gen2, rng, config);
  }
  //  1b. Inline call: (cutting plane, cold-start, frustum + ellipsoid).
  if (config.dynamic_dispatch) {
    set_default_seed();
    auto gen1 = [&gen]() -> const SetPtr<dgd::Frustum> {
      const auto set1 =
          gen.GetPrimitiveSet(dgd::bench::CurvedPrimitive3D::Frustum);
      return std::static_pointer_cast<dgd::Frustum>(set1);
    };
    auto gen2 = [&gen]() -> const SetPtr<dgd::Ellipsoid> {
      const auto set2 =
          gen.GetPrimitiveSet(dgd::bench::CurvedPrimitive3D::Ellipsoid);
      return std::static_pointer_cast<dgd::Ellipsoid>(set2);
    };
    std::cout
        << "Inline call     : (cutting plane, cold-start, frustum + ellipsoid)"
        << std::endl;
    ColdStart<3, dgd::Frustum, dgd::Ellipsoid>(gen1, gen2, rng, config);
  }

  //  2a. Dynamic dispatch: (cutting plane, cold-start, mesh + mesh).
  if (config.dynamic_dispatch && (gen.nmeshes() > 0)) {
    set_default_seed();
    auto gen1 = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetRandomMeshSet();
    };
    std::cout << "Dynamic dispatch: (cutting plane, cold-start, mesh + mesh)"
              << std::endl;
    ColdStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(gen1, gen1, rng, config);
  }
  //  2b. Inline call: (cutting plane, cold-start, mesh + mesh).
  if (config.dynamic_dispatch && (gen.nmeshes() > 0)) {
    set_default_seed();
    auto gen1 = [&gen]() -> const SetPtr<dgd::Mesh> {
      return std::static_pointer_cast<dgd::Mesh>(gen.GetRandomMeshSet());
    };
    std::cout << "Inline call     : (cutting plane, cold-start, mesh + mesh)"
              << std::endl;
    ColdStart<3, dgd::Mesh, dgd::Mesh>(gen1, gen1, rng, config);
  }

  std::cout << std::string(50, '-') << std::endl;
  set_random_seed();
  //  3a. Cold-start: (cutting plane, 2D primitive + 2D primitive).
  if (config.cold_start && config.two_dim_sets) {
    auto gen1 = [&gen]() -> const SetPtr<dgd::ConvexSet<2>> {
      return gen.GetRandom2DSet();
    };
    std::cout << "Cold-start: (cutting plane, 2D primitive + 2D primitive)"
              << std::endl;
    ColdStart<2, dgd::ConvexSet<2>, dgd::ConvexSet<2>>(gen1, gen1, rng, config);
  }
  //  3b. Cold-start: (trust region Newton, 2D primitive + 2D primitive).
  if (config.cold_start && config.two_dim_sets && config.trust_region_newton) {
    auto gen1 = [&gen]() -> const SetPtr<dgd::ConvexSet<2>> {
      return gen.GetRandom2DSet();
    };
    std::cout
        << "Cold-start: (trust region Newton, 2D primitive + 2D primitive)"
        << std::endl;
    ColdStart<2, dgd::ConvexSet<2>, dgd::ConvexSet<2>>(
        gen1, gen1, rng, config, SolverType::TrustRegionNewton);
  }

  //  4a. Cold-start: (cutting plane, 3D curved primitive + 3D curved
  //  primitive).
  if (config.cold_start && config.three_dim_sets) {
    auto gen1 = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetRandomCurvedPrimitive3DSet();
    };
    std::cout << "Cold-start: (cutting plane, 3D curved primitive + 3D curved "
                 "primitive)"
              << std::endl;
    ColdStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(gen1, gen1, rng, config);
  }
  //  4b. Cold-start: (trust region Newton, 3D curved primitive + 3D curved
  //  primitive).
  if (config.cold_start && config.three_dim_sets &&
      config.trust_region_newton) {
    auto gen1 = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetRandomCurvedPrimitive3DSet();
    };
    std::cout << "Cold-start: (trust region Newton, 3D curved primitive + 3D "
                 "curved primitive)"
              << std::endl;
    ColdStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(
        gen1, gen1, rng, config, SolverType::TrustRegionNewton);
  }

  //  5a. Cold-start: (cutting plane, 3D primitive + 3D primitive).
  if (config.cold_start && config.three_dim_sets) {
    auto gen1 = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetRandomPrimitive3DSet();
    };
    std::cout << "Cold-start: (cutting plane, 3D primitive + 3D primitive)"
              << std::endl;
    ColdStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(gen1, gen1, rng, config);
  }
  //  5b. Cold-start: (trust region Newton, 3D primitive + 3D primitive).
  if (config.cold_start && config.three_dim_sets &&
      config.trust_region_newton) {
    auto gen1 = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetRandomPrimitive3DSet();
    };
    std::cout
        << "Cold-start: (trust region Newton, 3D primitive + 3D primitive)"
        << std::endl;
    ColdStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(
        gen1, gen1, rng, config, SolverType::TrustRegionNewton);
  }

  //  6a. Cold-start: (cutting plane, mesh + mesh).
  if (config.cold_start && config.three_dim_sets && (gen.nmeshes() > 0)) {
    auto gen1 = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetRandomMeshSet();
    };
    std::cout << "Cold-start: (cutting plane, mesh + mesh)" << std::endl;
    ColdStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(gen1, gen1, rng, config);
  }
  //  6b. Cold-start: (trust region Newton, mesh + mesh).
  if (config.cold_start && config.three_dim_sets &&
      config.trust_region_newton && (gen.nmeshes() > 0)) {
    auto gen1 = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetRandomMeshSet();
    };
    std::cout << "Cold-start: (trust region Newton, mesh + mesh)" << std::endl;
    ColdStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(
        gen1, gen1, rng, config, SolverType::TrustRegionNewton);
  }

  std::cout << std::string(50, '-') << std::endl;
  //  7a. Warm-start: (cutting plane, 2D primitive + 2D primitive).
  if (config.warm_start && config.two_dim_sets) {
    auto gen1 = [&gen]() -> const SetPtr<dgd::ConvexSet<2>> {
      return gen.GetRandom2DSet();
    };
    std::cout
        << "Warm-start: (primal, cutting plane, 2D primitive + 2D primitive)"
        << std::endl;
    WarmStart<2, dgd::ConvexSet<2>, dgd::ConvexSet<2>>(gen1, gen1, rng, config,
                                                       WarmStartType::Primal);
    std::cout
        << "Warm-start: (dual, cutting plane, 2D primitive + 2D primitive)"
        << std::endl;
    WarmStart<2, dgd::ConvexSet<2>, dgd::ConvexSet<2>>(gen1, gen1, rng, config,
                                                       WarmStartType::Dual);
  }
  //  7b. Warm-start: (trust region Newton, 2D primitive + 2D primitive).
  if (config.warm_start && config.two_dim_sets && config.trust_region_newton) {
    auto gen1 = [&gen]() -> const SetPtr<dgd::ConvexSet<2>> {
      return gen.GetRandom2DSet();
    };
    std::cout << "Warm-start: (primal, trust region Newton, 2D primitive + 2D "
                 "primitive)"
              << std::endl;
    WarmStart<2, dgd::ConvexSet<2>, dgd::ConvexSet<2>>(
        gen1, gen1, rng, config, WarmStartType::Primal,
        SolverType::TrustRegionNewton);
    std::cout << "Warm-start: (dual, trust region Newton, 2D primitive + 2D "
                 "primitive)"
              << std::endl;
    WarmStart<2, dgd::ConvexSet<2>, dgd::ConvexSet<2>>(
        gen1, gen1, rng, config, WarmStartType::Dual,
        SolverType::TrustRegionNewton);
  }

  //  8a. Warm-start: (cutting plane, 3D curved primitive + 3D curved
  //  primitive).
  if (config.warm_start && config.three_dim_sets) {
    auto gen1 = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetRandomCurvedPrimitive3DSet();
    };
    std::cout << "Warm-start: (primal, cutting plane, 3D curved primitive + 3D "
                 "curved primitive)"
              << std::endl;
    WarmStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(gen1, gen1, rng, config,
                                                       WarmStartType::Primal);
    std::cout << "Warm-start: (dual, cutting plane, 3D curved primitive + 3D "
                 "curved primitive)"
              << std::endl;
    WarmStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(gen1, gen1, rng, config,
                                                       WarmStartType::Dual);
  }
  //  8b. Warm-start: (trust region Newton, 3D curved primitive + 3D curved
  //  primitive).
  if (config.warm_start && config.three_dim_sets &&
      config.trust_region_newton) {
    auto gen1 = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetRandomCurvedPrimitive3DSet();
    };
    std::cout << "Warm-start: (primal, trust region Newton, 3D curved "
                 "primitive + 3D curved primitive)"
              << std::endl;
    WarmStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(
        gen1, gen1, rng, config, WarmStartType::Primal,
        SolverType::TrustRegionNewton);
    std::cout << "Warm-start: (dual, trust region Newton, 3D curved primitive "
                 "+ 3D curved primitive)"
              << std::endl;
    WarmStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(
        gen1, gen1, rng, config, WarmStartType::Dual,
        SolverType::TrustRegionNewton);
  }

  //  9a. Warm-start: (cutting plane, mesh + mesh).
  if (config.warm_start && config.three_dim_sets && (gen.nmeshes() > 0)) {
    auto gen1 = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetRandomMeshSet();
    };
    std::cout << "Warm-start: (primal, cutting plane, mesh + mesh)"
              << std::endl;
    WarmStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(gen1, gen1, rng, config,
                                                       WarmStartType::Primal);
    std::cout << "Warm-start: (dual, cutting plane, mesh + mesh)" << std::endl;
    WarmStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(gen1, gen1, rng, config,
                                                       WarmStartType::Dual);
  }
  //  9b. Warm-start: (trust region Newton, mesh + mesh).
  if (config.warm_start && config.three_dim_sets && (gen.nmeshes() > 0) &&
      config.trust_region_newton) {
    auto gen1 = [&gen]() -> const SetPtr<dgd::ConvexSet<3>> {
      return gen.GetRandomMeshSet();
    };
    std::cout << "Warm-start: (primal, trust region Newton, mesh + mesh)"
              << std::endl;
    WarmStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(
        gen1, gen1, rng, config, WarmStartType::Primal,
        SolverType::TrustRegionNewton);
    std::cout << "Warm-start: (dual, trust region Newton, mesh + mesh)"
              << std::endl;
    WarmStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(
        gen1, gen1, rng, config, WarmStartType::Dual,
        SolverType::TrustRegionNewton);
  }
}
