#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <typeinfo>

#include "dgd/data_types.h"
#include "dgd/dgd.h"
#include "dgd/error_metrics.h"
#include "dgd/geometry/geometry_2d.h"
#include "dgd/geometry/geometry_3d.h"
#include "dgd/geometry/halfspace.h"
#include "dgd/graham_scan.h"
#include "dgd/mesh_loader.h"
#include "dgd/output.h"
#include "dgd/settings.h"
#include "dgd/utils/random.h"
#include "dgd/utils/transformations.h"
#include "test_utils.h"

namespace {

using namespace dgd;
using dgd::test::ConvexSetPtr;
using dgd::test::MakeConvexSetPair;

const Real kTol = std::pow(kEps, Real(0.49));

// ---------------------------------------------------------------------------
// Growth distance tests
// ---------------------------------------------------------------------------

template <int dim>
using GrowthDistanceType = Real (*)(const ConvexSet<dim>*,
                                    const Transformr<dim>&,
                                    const ConvexSet<dim>*,
                                    const Transformr<dim>&, const Settings&,
                                    Output<dim>&, bool);

template <int dim, GrowthDistanceType<dim> gd>
bool WarmStartFunctionality() {
  return (gd == static_cast<GrowthDistanceType<dim>>(GrowthDistanceCp<dim>));
}

template <int dim, GrowthDistanceType<dim> gd>
void BundleSchemeGrowthDistanceTest() {
  const bool warm_start = WarmStartFunctionality<dim, gd>();

  // Qhull computations can be unstable with float.
  if ((dim == 3) && (typeid(Real) == typeid(float))) GTEST_SKIP();

  Rng rng;
  rng.SetSeed();
  const int nsamples_cold = 100;
  const int nsamples_warm = 200;

  ConvexSetPtr<dim> set1, set2;
  MakeConvexSetPair<dim>(set1, set2, Real(0.1), Real(0.1));

  // Compute growth distance for random transformations.
  Settings settings;
  Output<dim> out;
  const Real dt = Real(0.1);

  for (int i = 0; i < nsamples_cold; ++i) {
    auto tf1 = rng.RandomTransform<dim>(Real(-2.0), Real(2.0));
    auto tf2 = rng.RandomTransform<dim>(Real(-2.0), Real(2.0));
    Vecr<dim> v;
    for (int k = 0; k < dim; ++k) v(k) = rng.Random();
    const Rotationr<dim> dR = rng.RandomRotation<dim>(kPi * dt);
    for (int j = 0; j < nsamples_warm; ++j) {
      const bool ws = (j > 0) && warm_start;
      if (2 * j > nsamples_warm) settings.ws_type = WarmStartType::Dual;
      gd(set1.get(), tf1, set2.get(), tf2, settings, out, ws);
      ASSERT_TRUE(out.status == SolutionStatus::Optimal ||
                  out.status == SolutionStatus::CoincidentCenters);
      const SolutionError err =
          ComputeSolutionError(set1.get(), tf1, set2.get(), tf2, out);
      ASSERT_NEAR(err.prim_infeas_err, Real(0.0), kTol);
      ASSERT_NEAR(err.prim_dual_gap, Real(0.0), kTol);

      Linear(tf1) *= dR;
      Affine(tf1) += v * dt;
    }
  }
}

TEST(GrowthDistanceTest, CuttingPlane_EllipsePolygon) {
  BundleSchemeGrowthDistanceTest<2, GrowthDistanceCp<2>>();
}

TEST(GrowthDistanceTest, CuttingPlane_ConeMesh) {
  BundleSchemeGrowthDistanceTest<3, GrowthDistanceCp<3>>();
}

TEST(GrowthDistanceTest, TrustRegionNewton_EllipsePolygon) {
  BundleSchemeGrowthDistanceTest<2, GrowthDistanceTrn<2>>();
}

TEST(GrowthDistanceTest, TrustRegionNewton_ConeMesh) {
  BundleSchemeGrowthDistanceTest<3, GrowthDistanceTrn<3>>();
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

  Affine(tfc) = Vec3r(Real(8.0), Real(-7.0), Real(0.7));
  const Real gd =
      GrowthDistance(setc.get(), tfc, seth.get(), tfh, settings, out, false);
  ASSERT_TRUE(out.status == SolutionStatus::Optimal);
  ASSERT_NEAR(gd, Real(0.2), kTol);

  Affine(tfc) = Vec3r(Real(8.0), Real(-7.0), Real(-0.1));
  GrowthDistance(setc.get(), tfc, seth.get(), tfh, settings, out, false);
  ASSERT_TRUE(out.status == SolutionStatus::CoincidentCenters);
}

// ---------------------------------------------------------------------------
// Collision detection tests
// ---------------------------------------------------------------------------

template <int dim>
void CuttingPlaneDetectCollisionTest() {
  // Qhull computations can be unstable with float.
  if ((dim == 3) && (typeid(Real) == typeid(float))) GTEST_SKIP();

  Rng rng;
  rng.SetSeed();
  const int nsamples_cold = 100;
  const int nsamples_warm = 200;

  ConvexSetPtr<dim> set1, set2;
  MakeConvexSetPair<dim>(set1, set2, Real(0.1), Real(0.1));

  // Check collisions for random transformations.
  Settings settings;
  Output<dim> out;
  const Real dt = Real(0.1);

  for (int i = 0; i < nsamples_cold; ++i) {
    auto tf1 = rng.RandomTransform<dim>(Real(-5.0), Real(5.0));
    auto tf2 = rng.RandomTransform<dim>(Real(-5.0), Real(5.0));
    Vecr<dim> v;
    for (int k = 0; k < dim; ++k) v(k) = rng.Random();
    const Rotationr<dim> dR = rng.RandomRotation<dim>(kPi * dt);
    for (int j = 0; j < nsamples_warm; ++j) {
      if (2 * j > nsamples_warm) settings.ws_type = WarmStartType::Dual;
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

TEST(DetectCollisionTest, CuttingPlane_EllipsePolygon) {
  CuttingPlaneDetectCollisionTest<2>();
}

TEST(DetectCollisionTest, CuttingPlane_ConeMesh) {
  CuttingPlaneDetectCollisionTest<3>();
}

}  // namespace
