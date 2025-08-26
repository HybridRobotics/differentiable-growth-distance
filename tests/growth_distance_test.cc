#include "dgd/growth_distance.h"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <typeinfo>

#include "dgd/data_types.h"
#include "dgd/error_metrics.h"
#include "dgd/geometry/geometry_2d.h"
#include "dgd/geometry/geometry_3d.h"
#include "dgd/graham_scan.h"
#include "dgd/mesh_loader.h"
#include "dgd/utils/random.h"

namespace {

using namespace dgd;

template <int dim>
using ConvexSetPtr = std::unique_ptr<ConvexSet<dim>>;

void Set2dConvexSets(ConvexSetPtr<2>& set1, ConvexSetPtr<2>& set2,
                     const Real margin1, const Real margin2) {
  // Set 1: ellipse.
  const Real hlx = 3.0, hly = 2.0;
  set1 = std::make_unique<Ellipse>(hlx, hly, margin1);

  // Set 2: polygon.
  Rng rng;
  rng.SetDefaultSeed();
  const int npts = 100;
  const Real len = 2.0;
  std::vector<Vec2r> pts, vert;
  Vec2r vec;
  const int pnorm = 6;
  for (int i = 0; i < npts; ++i) {
    vec << rng.Random(), rng.Random();
    vec = vec * len / (vec.lpNorm<pnorm>() + kEps);
    pts.push_back(vec);
  }
  GrahamScan(pts, vert);
  Real inradius = ComputePolygonInradius(vert, Vec2r::Zero());
  set2 = std::make_unique<Polygon>(vert, inradius, margin2);
}

void Set3dConvexSets(ConvexSetPtr<3>& set1, ConvexSetPtr<3>& set2,
                     const Real margin1, const Real margin2) {
  // Set 1: cone.
  const Real ha = kPi / 6.0, radius = 1.0;
  const Real height = radius / std::tan(ha);
  set1 = std::make_unique<Cone>(radius, height, margin1);

  // Set 2: mesh.
  Rng rng;
  rng.SetDefaultSeed();
  const int npts = 500;
  const Real len = 2.0;
  MeshLoader ml{};
  std::vector<Vec3r> pts, vert;
  std::vector<int> graph;
  Vec3r vec;
  constexpr int pnorm = 4;
  for (int i = 0; i < npts; ++i) {
    vec << rng.Random(), rng.Random(), rng.Random();
    vec = vec * len / (vec.lpNorm<pnorm>() + kEps);
    pts.push_back(vec);
  }
  ml.ProcessPoints(pts);
  bool valid = ml.MakeVertexGraph(vert, graph);
  ASSERT_TRUE(valid);
  const Real inradius = ml.ComputeInradius(vec);
  for (auto& v : vert) v -= vec;
  set2 = std::make_unique<Mesh>(vert, graph, inradius, margin2);
}

const Real kTol = kSqrtEps;

TEST(GrowthDistanceTest, EllipsePolygon) {
  Rng rng;
  rng.SetDefaultSeed();
  const int nsamples_cold = 100;
  const int nsamples_warm = 100;

  ConvexSetPtr<2> set1, set2;
  Set2dConvexSets(set1, set2, Real(0.1), Real(0.1));

  // Compute growth distance for random transformations.
  Transform2r tf1, tf2;
  Settings settings;
  Output<2> out;
  const Real dt = Real(0.1);

  for (int i = 0; i < nsamples_cold; ++i) {
    rng.RandomTransform(-2.0, 2.0, tf1);
    rng.RandomTransform(-2.0, 2.0, tf2);
    const Vec2r v(rng.Random(), rng.Random());
    const Real w = rng.Random(kPi);
    const Rotation2r dR = Eigen::AngleAxis<Real>(w * dt, Vec3r::UnitZ())
                              .matrix()
                              .topLeftCorner<2, 2>();
    for (int j = 0; j < nsamples_warm; ++j) {
      GrowthDistance(set1.get(), tf1, set2.get(), tf2, settings, out, (j > 0));
      ASSERT_TRUE(out.status == SolutionStatus::Optimal ||
                  out.status == SolutionStatus::CoincidentCenters);
      const SolutionError err =
          ComputeSolutionError(set1.get(), tf1, set2.get(), tf2, out);
      ASSERT_NEAR(err.prim_infeas_err, 0.0, kTol);
      ASSERT_NEAR(err.prim_dual_gap, 0.0, kTol);

      Linear(tf1) *= dR;
      Affine(tf1) += v * dt;
    }
  }
}

TEST(DetectCollisionTest, EllipsePolygon) {
  Rng rng;
  rng.SetDefaultSeed();
  const int nsamples_cold = 100;
  const int nsamples_warm = 100;

  ConvexSetPtr<2> set1, set2;
  Set2dConvexSets(set1, set2, Real(0.1), Real(0.1));

  // Check collisions for random transformations.
  Transform2r tf1, tf2;
  Settings settings;
  Output<2> out;
  const Real dt = Real(0.1);

  for (int i = 0; i < nsamples_cold; ++i) {
    rng.RandomTransform(-5.0, 5.0, tf1);
    rng.RandomTransform(-5.0, 5.0, tf2);
    const Vec2r v(rng.Random(), rng.Random());
    const Real w = rng.Random(kPi);
    const Rotation2r dR = Eigen::AngleAxis<Real>(w * dt, Vec3r::UnitZ())
                              .matrix()
                              .topLeftCorner<2, 2>();
    for (int j = 0; j < nsamples_warm; ++j) {
      const bool collision = DetectCollision(set1.get(), tf1, set2.get(), tf2,
                                             settings, out, (j > 0));
      ASSERT_TRUE(out.status == SolutionStatus::Optimal ||
                  out.status == SolutionStatus::CoincidentCenters);
      const bool assertion = AssertCollisionStatus(set1.get(), tf1, set2.get(),
                                                   tf2, out, collision);
      ASSERT_TRUE(assertion);

      Linear(tf1) *= dR;
      Affine(tf1) += v * dt;
    }
  }
}

TEST(GrowthDistanceTest, ConeMesh) {
  // Qhull computations can be unstable with float.
  if (typeid(Real) == typeid(float)) GTEST_SKIP();

  Rng rng;
  rng.SetDefaultSeed();
  const int nsamples_cold = 100;
  const int nsamples_warm = 100;

  ConvexSetPtr<3> set1, set2;
  Set3dConvexSets(set1, set2, Real(0.1), Real(0.1));

  // Compute growth distance for random transformations.
  Transform3r tf1, tf2;
  Settings settings;
  Output<3> out;
  const Real dt = Real(0.1);

  for (int i = 0; i < nsamples_cold; ++i) {
    rng.RandomTransform(-3.0, 3.0, tf1);
    rng.RandomTransform(-3.0, 3.0, tf2);
    const Vec3r v(rng.Random(), rng.Random(), rng.Random());
    const Vec3r euler(rng.Random(kPi), rng.Random(kPi), rng.Random(kPi));
    Rotation3r dR;
    EulerToRotation(dt * euler, dR);
    for (int j = 0; j < nsamples_warm; ++j) {
      GrowthDistance(set1.get(), tf1, set2.get(), tf2, settings, out, (j > 0));
      ASSERT_TRUE(out.status == SolutionStatus::Optimal ||
                  out.status == SolutionStatus::CoincidentCenters);
      const SolutionError err =
          ComputeSolutionError(set1.get(), tf1, set2.get(), tf2, out);
      ASSERT_NEAR(err.prim_infeas_err, 0.0, kTol);
      ASSERT_NEAR(err.prim_dual_gap, 0.0, kTol);

      Linear(tf1) *= dR;
      Affine(tf1) += v * dt;
    }
  }
}

TEST(DetectCollisionTest, ConeMesh) {
  // Qhull computations can be unstable with float.
  if (typeid(Real) == typeid(float)) GTEST_SKIP();

  Rng rng;
  rng.SetDefaultSeed();
  const int nsamples_cold = 100;
  const int nsamples_warm = 100;

  ConvexSetPtr<3> set1, set2;
  Set3dConvexSets(set1, set2, Real(0.1), Real(0.1));

  // Compute growth distance for random transformations.
  Transform3r tf1, tf2;
  Settings settings;
  Output<3> out;
  const Real dt = Real(0.1);

  for (int i = 0; i < nsamples_cold; ++i) {
    rng.RandomTransform(-6.0, 6.0, tf1);
    rng.RandomTransform(-6.0, 6.0, tf2);
    const Vec3r v(rng.Random(), rng.Random(), rng.Random());
    const Vec3r euler(rng.Random(kPi), rng.Random(kPi), rng.Random(kPi));
    Rotation3r dR;
    EulerToRotation(dt * euler, dR);
    for (int j = 0; j < nsamples_warm; ++j) {
      const bool collision = DetectCollision(set1.get(), tf1, set2.get(), tf2,
                                             settings, out, (j > 0));
      ASSERT_TRUE(out.status == SolutionStatus::Optimal ||
                  out.status == SolutionStatus::CoincidentCenters);
      const bool assertion = AssertCollisionStatus(set1.get(), tf1, set2.get(),
                                                   tf2, out, collision);
      ASSERT_TRUE(assertion);

      Linear(tf1) *= dR;
      Affine(tf1) += v * dt;
    }
  }
}

TEST(GrowthDistanceTest, CuboidHalfspace) {
  // Set 1: Cuboid.
  const ConvexSetPtr<3> setc =
      std::make_unique<Cuboid>(Real(1.0), Real(2.0), Real(3.0), Real(0.1));
  // Set 2: Half-space.
  const auto seth = std::make_unique<Halfspace<3>>(Real(0.4));

  // Compute growth distance for random transformations.
  Transform3r tfc = Transform3r::Identity();
  const Transform3r tfh = Transform3r::Identity();
  Settings settings{};
  Output<3> out;

  Affine(tfc) = Vec3r(8.0, -7.0, 0.7);
  const Real gd =
      GrowthDistance(setc.get(), tfc, seth.get(), tfh, settings, out, false);
  ASSERT_TRUE(out.status == SolutionStatus::Optimal);
  ASSERT_NEAR(gd, 0.2, kTol);
  Output<3> out_swap;
  const Real gd_swap = GrowthDistance(seth.get(), tfh, setc.get(), tfc,
                                      settings, out_swap, false);
  ASSERT_TRUE(out_swap.status == SolutionStatus::Optimal);
  ASSERT_NEAR(gd_swap, gd, kTol);
  ASSERT_NEAR((out.normal + out_swap.normal).norm(), 0.0, kTol);
  ASSERT_NEAR((out.z1 - out_swap.z2).norm(), 0.0, kTol);

  Affine(tfc) = Vec3r(8.0, -7.0, -Real(0.1));
  GrowthDistance(setc.get(), tfc, seth.get(), tfh, settings, out, false);
  ASSERT_TRUE(out.status == SolutionStatus::CoincidentCenters);
}

}  // namespace
