#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "dgd/data_types.h"
#include "dgd/error_metrics.h"
#include "dgd/geometry/3d/ellipsoid.h"
#include "dgd/geometry/3d/frustum.h"
#include "dgd/geometry/3d/mesh.h"
#include "dgd/geometry/convex_set.h"
#include "dgd/growth_distance.h"
#include "dgd/output.h"
#include "dgd/settings.h"
#include "dgd/utils/timer.h"
#include "internal_helpers/filesystem_utils.h"
#include "internal_helpers/random_utils.h"
#include "internal_helpers/set_generator.h"

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

template <int dim>
struct OptimalSolution {
  dgd::Vecr<dim> z1, z2, normal;
  dgd::Real gd;
  dgd::SolutionStatus status;

  void SetFromOutput(const dgd::Output<dim> out) {
    z1 = out.z1;
    z2 = out.z2;
    normal = out.normal;
    gd = out.growth_dist_ub;
    status = out.status;
  }

  void SetOutput(dgd::Output<dim>& out) const {
    out.z1 = z1;
    out.z2 = z2;
    out.normal = normal;
    out.growth_dist_ub = gd;
    out.status = status;
  }
};

void PrintStatistics(double avg_solve_time, double max_prim_dual_gap,
                     double max_prim_feas_err, double max_dual_feas_err,
                     double avg_iter, int nsubopt) {
  std::cout << "Avg. solve time (us)  : " << avg_solve_time << std::endl;
  std::cout << std::scientific;
  std::cout << "Max. prim dual gap    : " << max_prim_dual_gap << std::endl;
  std::cout << "Max. prim feas err (m): " << max_prim_feas_err << std::endl;
  std::cout << "Max. dual feas err    : " << max_dual_feas_err << std::endl;
  std::cout.unsetf(std::ios::fixed | std::ios::scientific);
  std::cout << "Avg. iterations       : " << avg_iter << std::endl;
  std::cout << "Num. suboptimal runs  : " << nsubopt << std::endl;
  std::cout << std::endl;
}

template <int dim, class C1, class C2>
void ColdStart(std::function<const C1*()> generator1,
               std::function<const C2*()> generator2, dgd::Rng& rng, int npair,
               int npose) {
  double avg_solve_time = 0.0;
  double max_prim_dual_gap = 0.0;
  double max_prim_feas_err = 0.0;
  double max_dual_feas_err = 0.0;
  double avg_iter = 0.0;
  int nsubopt = 0;

  dgd::Timer timer;
  timer.Stop();
  dgd::Transformr<dim> tf1, tf2;
  dgd::Settings settings;
  dgd::Output<dim> out;
  for (int i = 0; i < npair; ++i) {
    const C1* set1 = generator1();
    const C2* set2 = generator2();
    for (int j = 0; j < npose; ++j) {
      dgd::internal::SetRandomRigidBodyTransforms(rng, tf1, tf2, -position_lim,
                                                  position_lim);
      // Initial call to reduce cache misses.
      dgd::GrowthDistance(set1, tf1, set2, tf2, settings, out);
      timer.Start();
      for (int k = 0; k < ncold; ++k) {
        dgd::GrowthDistance(set1, tf1, set2, tf2, settings, out);
      }
      timer.Stop();
      const auto err = dgd::ComputeSolutionError(set1, tf1, set2, tf2, out);

      avg_solve_time += timer.Elapsed() / double(ncold);
      max_prim_dual_gap = std::max(max_prim_dual_gap, err.prim_dual_gap);
      max_prim_feas_err = std::max(max_prim_feas_err, err.prim_feas_err);
      max_dual_feas_err = std::max(max_dual_feas_err, err.dual_feas_err);
      avg_iter += out.iter;
      nsubopt += (out.status != dgd::SolutionStatus::Optimal) &&
                 (out.status != dgd::SolutionStatus::CoincidentCenters);
    }
  }
  avg_solve_time = avg_solve_time / (npair * npose);
  avg_iter = avg_iter / (npair * npose);

  PrintStatistics(avg_solve_time, max_prim_dual_gap, max_prim_feas_err,
                  max_dual_feas_err, avg_iter, nsubopt);
}

template <int dim, class C1, class C2>
void WarmStart(std::function<const C1*()> generator1,
               std::function<const C2*()> generator2, dgd::Rng& rng, int npair,
               int npose) {
  double avg_solve_time = 0.0;
  double max_prim_dual_gap = 0.0;
  double max_prim_feas_err = 0.0;
  double max_dual_feas_err = 0.0;
  double avg_iter = 0.0;
  int nsubopt = 0;

  dgd::Timer timer;
  timer.Stop();
  dgd::Transformr<dim> tf1, tf1_c, tf2;
  dgd::Vecr<dim> dx;
  dgd::Rotationr<dim> drot;
  dgd::Settings settings;
  dgd::Output<dim> out;
  std::vector<OptimalSolution<dim>> opt_sols(nwarm);
  for (int i = 0; i < npair; ++i) {
    const C1* set1 = generator1();
    const C2* set2 = generator2();
    for (int j = 0; j < npose; ++j) {
      dgd::internal::SetRandomRigidBodyTransforms(rng, tf1, tf2, -position_lim,
                                                  position_lim);
      tf1_c = tf1;
      dgd::internal::SetRandomDisplacement(rng, dx, drot, dx_max, ang_max);
      // Initial cold-start call.
      dgd::GrowthDistance(set1, tf1, set2, tf2, settings, out);
      int total_iter = 0;
      timer.Start();
      for (int k = 0; k < nwarm; ++k) {
        dgd::internal::UpdateRigidBodyTransform(tf1, dx, drot);
        dgd::GrowthDistance(set1, tf1, set2, tf2, settings, out, true);
        opt_sols[k].SetFromOutput(out);
        total_iter += out.iter;
      }
      timer.Stop();
      avg_solve_time += timer.Elapsed() / double(nwarm);
      avg_iter += total_iter / double(nwarm);

      tf1 = tf1_c;
      for (int k = 0; k < nwarm; ++k) {
        dgd::internal::UpdateRigidBodyTransform(tf1, dx, drot);
        opt_sols[k].SetOutput(out);
        const auto err = dgd::ComputeSolutionError(set1, tf1, set2, tf2, out);
        max_prim_dual_gap = std::max(max_prim_dual_gap, err.prim_dual_gap);
        max_prim_feas_err = std::max(max_prim_feas_err, err.prim_feas_err);
        max_dual_feas_err = std::max(max_dual_feas_err, err.dual_feas_err);
        nsubopt += (out.status != dgd::SolutionStatus::Optimal) &&
                   (out.status != dgd::SolutionStatus::CoincidentCenters);
      }
    }
  }
  avg_solve_time = avg_solve_time / (npair * npose);
  avg_iter = avg_iter / (npair * npose);

  PrintStatistics(avg_solve_time, max_prim_dual_gap, max_prim_feas_err,
                  max_dual_feas_err, avg_iter, nsubopt);
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
  auto set_seed = [&gen, &rng]() -> void {
    gen.SetDefaultRngSeed();
    rng.SetDefaultSeed();
  };

  // Benchmarks.
  //  1. Dynamic dispatch: (frustum + ellipsoid, cold-start).
  {
    set_seed();
    auto generator1 = [&gen]() -> const dgd::ConvexSet<3>* {
      return gen.GetPrimitiveSet(dgd::internal::CurvedPrimitive3D::Frustum)
          .get();
    };
    auto generator2 = [&gen]() -> const dgd::ConvexSet<3>* {
      return gen.GetPrimitiveSet(dgd::internal::CurvedPrimitive3D::Ellipsoid)
          .get();
    };
    std::cout << "Dynamic dispatch: (frustum + ellipsoid, cold-start)"
              << std::endl;
    ColdStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(generator1, generator2,
                                                       rng, npair, npose_c);
  }
  //  2. Inline call: (frustum + ellipsoid, cold-start).
  {
    set_seed();
    auto generator1 = [&gen]() -> const dgd::Frustum* {
      const auto set1 =
          gen.GetPrimitiveSet(dgd::internal::CurvedPrimitive3D::Frustum);
      return static_cast<const dgd::Frustum*>(set1.get());
    };
    auto generator2 = [&gen]() -> const dgd::Ellipsoid* {
      const auto set2 =
          gen.GetPrimitiveSet(dgd::internal::CurvedPrimitive3D::Ellipsoid);
      return static_cast<const dgd::Ellipsoid*>(set2.get());
    };
    std::cout << "Inline call     : (frustum + ellipsoid, cold-start)"
              << std::endl;
    ColdStart<3, dgd::Frustum, dgd::Ellipsoid>(generator1, generator2, rng,
                                               npair, npose_c);
  }

  //  3. Dynamic dispatch: (mesh + mesh, cold-start).
  if (gen.nmeshes() > 0) {
    set_seed();
    auto generator = [&gen]() -> const dgd::ConvexSet<3>* {
      return gen.GetRandomMeshSet().get();
    };
    std::cout << "Dynamic dispatch: (mesh + mesh, cold-start)" << std::endl;
    ColdStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(generator, generator,
                                                       rng, npair, npose_c);
  }
  //  4. Inline call: (mesh + mesh, cold-start).
  if (gen.nmeshes() > 0) {
    set_seed();
    auto generator = [&gen]() -> const dgd::Mesh* {
      return static_cast<const dgd::Mesh*>(gen.GetRandomMeshSet().get());
    };
    std::cout << "Inline call     : (mesh + mesh, cold-start)" << std::endl;
    ColdStart<3, dgd::Mesh, dgd::Mesh>(generator, generator, rng, npair,
                                       npose_c);
  }

  std::cout << std::string(50, '-') << std::endl;
  //  5. Cold-start: (2D primitive + 2D primitive).
  {
    auto generator = [&gen]() -> const dgd::ConvexSet<2>* {
      return gen.GetRandom2DSet().get();
    };
    std::cout << "Cold-start: (2D primitive + 2D primitive)" << std::endl;
    ColdStart<2, dgd::ConvexSet<2>, dgd::ConvexSet<2>>(generator, generator,
                                                       rng, npair, npose_c);
  }
  //  6. Cold-start: (3D curved primitive + 3D curved primitive).
  {
    auto generator = [&gen]() -> const dgd::ConvexSet<3>* {
      return gen.GetRandomCurvedPrimitive3DSet().get();
    };
    std::cout << "Cold-start: (3D curved primitive + 3D curved primitive)"
              << std::endl;
    ColdStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(generator, generator,
                                                       rng, npair, npose_w);
  }
  //  7. Cold-start: (3D primitive + 3D primitive).
  {
    auto generator = [&gen]() -> const dgd::ConvexSet<3>* {
      return gen.GetRandomPrimitive3DSet().get();
    };
    std::cout << "Cold-start: (3D primitive + 3D primitive)" << std::endl;
    ColdStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(generator, generator,
                                                       rng, npair, npose_w);
  }

  std::cout << std::string(50, '-') << std::endl;
  //  8. Warm-start: (2D primitive + 2D primitive).
  {
    auto generator = [&gen]() -> const dgd::ConvexSet<2>* {
      return gen.GetRandom2DSet().get();
    };
    std::cout << "Warm-start: (2D primitive + 2D primitive)" << std::endl;
    WarmStart<2, dgd::ConvexSet<2>, dgd::ConvexSet<2>>(generator, generator,
                                                       rng, npair, npose_w);
  }
  //  9. Warm-start: (3D curved primitive + 3D curved primitive).
  {
    auto generator = [&gen]() -> const dgd::ConvexSet<3>* {
      return gen.GetRandomCurvedPrimitive3DSet().get();
    };
    std::cout << "Warm-start: (3D curved primitive + 3D curved primitive)"
              << std::endl;
    WarmStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(generator, generator,
                                                       rng, npair, npose_w);
  }
  //  10. Warm-start: (mesh + mesh).
  if (gen.nmeshes() > 0) {
    auto generator = [&gen]() -> const dgd::ConvexSet<3>* {
      return gen.GetRandomMeshSet().get();
    };
    std::cout << "Warm-start: (mesh + mesh)" << std::endl;
    WarmStart<3, dgd::ConvexSet<3>, dgd::ConvexSet<3>>(generator, generator,
                                                       rng, npair, npose_w);
  }
}
