#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <typeinfo>

#include "dgd/data_types.h"
#include "dgd/geometry/2d/ellipse.h"
#include "dgd/geometry/2d/polygon.h"
#include "dgd/geometry/3d/cone.h"
#include "dgd/geometry/3d/mesh.h"
#include "dgd/graham_scan.h"
// clang-format off
#include "dgd/growth_distance.h"
// clang-format on
#include "dgd/mesh_loader.h"

namespace {

using namespace dgd;

template <int dim>
using ConvexSetPtr = std::unique_ptr<ConvexSet<dim>>;

void Set2dConvexSets(ConvexSetPtr<2>& set1, ConvexSetPtr<2>& set2,
                     const Real margin1, const Real margin2) {
  // Set 1: ellipse.
  const Real hlx{3.0}, hly{2.0};
  set1 = std::make_unique<Ellipse>(hlx, hly, margin1);

  // Set 2: polygon.
  SetDefaultSeed();
  const int npts{100};
  const Real len{2.0};
  std::vector<Vec2f> pts, vert;
  Vec2f vec;
  const int pnorm{6};
  for (int i = 0; i < npts; ++i) {
    vec = Vec2f(Random(1.0), Random(1.0));
    vec = vec * len / (vec.lpNorm<pnorm>() + kEps);
    pts.push_back(vec);
  }
  GrahamScan(pts, vert);
  Real inradius{ComputePolygonInradius(vert, Vec2f::Zero())};
  set2 = std::make_unique<Polygon>(vert, margin2, inradius);
}

void Set3dConvexSets(ConvexSetPtr<3>& set1, ConvexSetPtr<3>& set2,
                     const Real margin1, const Real margin2) {
  // Set 1: cone.
  const Real ha{kPi / 6.0}, radius{1.0};
  const Real height{radius / std::tan(ha)};
  set1 = std::make_unique<Cone>(radius, height, margin1);

  // Set 2: mesh.
  SetDefaultSeed();
  const int npts{500};
  const Real len{2.0};
  MeshLoader ml{};
  std::vector<Vec3f> pts, vert;
  std::vector<int> graph;
  Vec3f vec;
  const int pnorm{4};
  for (int i = 0; i < npts; ++i) {
    vec = Vec3f(Random(1.0), Random(1.0), Random(1.0));
    vec = vec * len / (vec.lpNorm<pnorm>() + kEps);
    pts.push_back(vec);
  }
  ml.ProcessPoints(pts);
  bool valid{ml.MakeVertexGraph(vert, graph)};
  ASSERT_TRUE(valid);
  const Real inradius{ml.ComputeInradius(vec)};
  for (auto& v : vert) v -= vec;
  set2 = std::make_unique<Mesh>(vert, graph, margin2, inradius);
}

const Real kTol{kEpsSqrt};

TEST(GrowthDistanceTest, EllipsePolygon) {
  const int nsamples_cold{100};
  const int nsamples_warm{100};

  ConvexSetPtr<2> set1, set2;
  Set2dConvexSets(set1, set2, 0.0, 0.0);

  // Compute growth distance for random transformations.
  Transform2f tf1, tf2;
  SolverSettings settings;
  SolverOutput<2> out;
  const Real dt{Real(0.1)};

  for (int i = 0; i < nsamples_cold; ++i) {
    RandomRigidBodyTransform<2>(-2.0, 2.0, tf1);
    RandomRigidBodyTransform<2>(-2.0, 2.0, tf2);
    const Vec2f v(Random(1.0), Random(1.0));
    const Real w{Random(kPi)};
    const Rot2f dR{Eigen::AngleAxis<Real>(w * dt, Vec3f::UnitZ())
                       .matrix()
                       .topLeftCorner<2, 2>()};
    for (int j = 0; j < nsamples_warm; ++j) {
      GrowthDistance(set1.get(), tf1, set2.get(), tf2, settings, out, (j > 0));
      ASSERT_TRUE(out.status == SolutionStatus::kOptimal ||
                  out.status == SolutionStatus::kCoincidentCenters);
      const SolutionError err{
          GetSolutionError(set1.get(), tf1, set2.get(), tf2, out)};
      ASSERT_NEAR(err.prim_feas_err, 0.0, kTol);
      ASSERT_NEAR(err.prim_dual_gap, 0.0, kTol);

      tf1.topLeftCorner<2, 2>() *= dR;
      tf1.topRightCorner<2, 1>() += v * dt;
    }
  }
}

TEST(CollisionCheckTest, EllipsePolygon) {
  const int nsamples_cold{100};
  const int nsamples_warm{100};

  ConvexSetPtr<2> set1, set2;
  Set2dConvexSets(set1, set2, 0.0, 0.0);

  // Check collisions for random transformations.
  Transform2f tf1, tf2;
  SolverSettings settings;
  SolverOutput<2> out;
  const Real dt{Real(0.1)};

  for (int i = 0; i < nsamples_cold; ++i) {
    RandomRigidBodyTransform<2>(-5.0, 5.0, tf1);
    RandomRigidBodyTransform<2>(-5.0, 5.0, tf2);
    const Vec2f v(Random(1.0), Random(1.0));
    const Real w{Random(kPi)};
    const Rot2f dR{Eigen::AngleAxis<Real>(w * dt, Vec3f::UnitZ())
                       .matrix()
                       .topLeftCorner<2, 2>()};
    for (int j = 0; j < nsamples_warm; ++j) {
      const bool collision{CollisionCheck(set1.get(), tf1, set2.get(), tf2,
                                          settings, out, (j > 0))};
      ASSERT_TRUE(out.status == SolutionStatus::kOptimal ||
                  out.status == SolutionStatus::kCoincidentCenters);
      const bool assertion{
          AssertCollision(set1.get(), tf1, set2.get(), tf2, out, collision)};
      ASSERT_TRUE(assertion);

      tf1.topLeftCorner<2, 2>() *= dR;
      tf1.topRightCorner<2, 1>() += v * dt;
    }
  }
}

TEST(GrowthDistanceTest, ConeMesh) {
  // Qhull computations can be unstable with float.
  if (typeid(Real) == typeid(float)) GTEST_SKIP();

  const int nsamples_cold{100};
  const int nsamples_warm{100};

  ConvexSetPtr<3> set1, set2;
  Set3dConvexSets(set1, set2, 0.0, 0.0);

  // Compute growth distance for random transformations.
  Transform3f tf1, tf2;
  SolverSettings settings;
  SolverOutput<3> out;
  const Real dt{Real(0.1)};

  for (int i = 0; i < nsamples_cold; ++i) {
    RandomRigidBodyTransform<3>(-3.0, 3.0, tf1);
    RandomRigidBodyTransform<3>(-3.0, 3.0, tf2);
    const Vec3f v(Random(1.0), Random(1.0), Random(1.0));
    const Vec3f euler(Random(kPi), Random(kPi), Random(kPi));
    Rot3f dR;
    EulerToRotation(dt * euler, dR);
    for (int j = 0; j < nsamples_warm; ++j) {
      GrowthDistance(set1.get(), tf1, set2.get(), tf2, settings, out, (j > 0));
      ASSERT_TRUE(out.status == SolutionStatus::kOptimal ||
                  out.status == SolutionStatus::kCoincidentCenters);
      const SolutionError err{
          GetSolutionError(set1.get(), tf1, set2.get(), tf2, out)};
      ASSERT_NEAR(err.prim_feas_err, 0.0, kTol);
      ASSERT_NEAR(err.prim_dual_gap, 0.0, kTol);

      tf1.topLeftCorner<3, 3>() *= dR;
      tf1.topRightCorner<3, 1>() += v * dt;
    }
  }
}

TEST(CollisionCheckTest, ConeMesh) {
  // Qhull computations can be unstable with float.
  if (typeid(Real) == typeid(float)) GTEST_SKIP();

  const int nsamples_cold{100};
  const int nsamples_warm{100};

  ConvexSetPtr<3> set1, set2;
  Set3dConvexSets(set1, set2, 0.0, 0.0);

  // Compute growth distance for random transformations.
  Transform3f tf1, tf2;
  SolverSettings settings;
  SolverOutput<3> out;
  const Real dt{Real(0.1)};

  for (int i = 0; i < nsamples_cold; ++i) {
    RandomRigidBodyTransform<3>(-6.0, 6.0, tf1);
    RandomRigidBodyTransform<3>(-6.0, 6.0, tf2);
    const Vec3f v(Random(1.0), Random(1.0), Random(1.0));
    const Vec3f euler(Random(kPi), Random(kPi), Random(kPi));
    Rot3f dR;
    EulerToRotation(dt * euler, dR);
    for (int j = 0; j < nsamples_warm; ++j) {
      const bool collision{CollisionCheck(set1.get(), tf1, set2.get(), tf2,
                                          settings, out, (j > 0))};
      ASSERT_TRUE(out.status == SolutionStatus::kOptimal ||
                  out.status == SolutionStatus::kCoincidentCenters);
      const bool assertion{
          AssertCollision(set1.get(), tf1, set2.get(), tf2, out, collision)};
      ASSERT_TRUE(assertion);

      tf1.topLeftCorner<3, 3>() *= dR;
      tf1.topRightCorner<3, 1>() += v * dt;
    }
  }
}

}  // namespace
