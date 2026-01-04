#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <typeinfo>

#include "dgd/data_types.h"
#include "dgd/dgd.h"
#include "dgd/error_metrics.h"
#include "dgd/geometry/geometry_2d.h"
#include "dgd/geometry/geometry_3d.h"
#include "dgd/graham_scan.h"
#include "dgd/mesh_loader.h"
#include "dgd/utils/random.h"
#include "dgd/utils/transformations.h"

namespace {

using namespace dgd;

template <int dim>
using C = ConvexSet<dim>;
template <int dim>
using ConvexSetPtr = std::unique_ptr<ConvexSet<dim>>;

template <int dim>
void SetConvexSets(ConvexSetPtr<dim>& set1, ConvexSetPtr<dim>& set2,
                   const Real margin1, const Real margin2);

// Ellipse-Polygon
template <>
void SetConvexSets<2>(ConvexSetPtr<2>& set1, ConvexSetPtr<2>& set2,
                      const Real margin1, const Real margin2) {
  // Set 1: ellipse.
  const Real hlx = Real(3.0), hly = Real(2.0);
  set1 = std::make_unique<Ellipse>(hlx, hly, margin1);

  // Set 2: polygon.
  Rng rng;
  rng.SetSeed();
  const int npts = 100;
  const Real len = Real(2.0);
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

// Cone-Mesh
template <>
void SetConvexSets<3>(ConvexSetPtr<3>& set1, ConvexSetPtr<3>& set2,
                      const Real margin1, const Real margin2) {
  // Set 1: cone.
  const Real ha = kPi / Real(6.0), radius = Real(1.0);
  const Real height = radius / std::tan(ha);
  set1 = std::make_unique<Cone>(radius, height, margin1);

  // Set 2: mesh.
  Rng rng;
  rng.SetSeed();
  const int npts = 500;
  const Real len = Real(2.0);
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

const Real kTol = std::pow(kEps, Real(0.49));

// Growth distance tests
template <int dim>
using GrowthDistanceType = Real (*)(const ConvexSet<dim>*,
                                    const Transformr<dim>&,
                                    const ConvexSet<dim>*,
                                    const Transformr<dim>&, const Settings&,
                                    Output<dim>&, bool);

template <int d, GrowthDistanceType<d> GD>
struct GrowthDistanceConfig {
  static constexpr int dim = d;
  static constexpr GrowthDistanceType<dim> growth_distance = GD;
  static inline const bool warm_start =
      (GD == static_cast<GrowthDistanceType<dim>>(
                 GrowthDistanceCp<dim, C<dim>, C<dim>>));
};

struct GrowthDistanceCp_2D
    : GrowthDistanceConfig<2, GrowthDistanceCp<2, C<2>, C<2>>> {};
struct GrowthDistanceTrn_2D
    : GrowthDistanceConfig<2, GrowthDistanceTrn<2, C<2>, C<2>>> {};
struct GrowthDistanceCp_3D
    : GrowthDistanceConfig<3, GrowthDistanceCp<3, C<3>, C<3>>> {};
struct GrowthDistanceTrn_3D
    : GrowthDistanceConfig<3, GrowthDistanceTrn<3, C<3>, C<3>>> {};

struct GrowthDistanceNameGenerator {
  template <typename T>
  static std::string GetName(int) {
    if (T::growth_distance ==
        static_cast<GrowthDistanceType<T::dim>>(
            GrowthDistanceCp<T::dim, C<T::dim>, C<T::dim>>)) {
      return "CuttingPlane_" + std::to_string(T::dim) + "D";
    } else if (T::growth_distance ==
               static_cast<GrowthDistanceType<T::dim>>(
                   GrowthDistanceTrn<T::dim, C<T::dim>, C<T::dim>>)) {
      return "TrustRegionNewton_" + std::to_string(T::dim) + "D";
    } else {
      return "Unknown_" + std::to_string(T::dim) + "D";
    }
  }
};

template <typename T>
class GrowthDistanceTest : public testing::Test {
 protected:
  GrowthDistanceTest() {}

  ~GrowthDistanceTest() {}
};

using GrowthDistanceTypes =
    testing::Types<GrowthDistanceCp_2D, GrowthDistanceCp_3D,
                   GrowthDistanceTrn_2D, GrowthDistanceTrn_3D>;
TYPED_TEST_SUITE(GrowthDistanceTest, GrowthDistanceTypes,
                 GrowthDistanceNameGenerator);

TYPED_TEST(GrowthDistanceTest, SolutionConvergence) {
  constexpr int dim = TypeParam::dim;
  auto growth_distance = TypeParam::growth_distance;
  const bool warm_start = TypeParam::warm_start;

  // Qhull computations can be unstable with float.
  if ((dim == 3) && (typeid(Real) == typeid(float))) GTEST_SKIP();

  Rng rng;
  rng.SetSeed();
  const int nsamples_cold = 100;
  const int nsamples_warm = 200;

  ConvexSetPtr<dim> set1, set2;
  SetConvexSets<dim>(set1, set2, Real(0.1), Real(0.1));

  // Compute growth distance for random transformations.
  Transformr<dim> tf1, tf2;
  Settings settings;
  Output<dim> out;
  const Real dt = Real(0.1);

  for (int i = 0; i < nsamples_cold; ++i) {
    rng.RandomTransform(Real(-2.0), Real(2.0), tf1);
    rng.RandomTransform(Real(-2.0), Real(2.0), tf2);
    Vecr<dim> v;
    for (int k = 0; k < dim; ++k) v(k) = rng.Random();
    const Rotationr<dim> dR = rng.RandomRotation<dim>(kPi * dt);
    for (int j = 0; j < nsamples_warm; ++j) {
      const bool ws = (j > 0) && warm_start;
      if (2 * j > nsamples_warm) settings.ws_type = WarmStartType::Dual;
      growth_distance(set1.get(), tf1, set2.get(), tf2, settings, out, ws);
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

// Collision detection tests
struct DetectCollision_2D {
  static constexpr int dim = 2;
};
struct DetectCollision_3D {
  static constexpr int dim = 3;
};

struct DetectCollisionNameGenerator {
  template <typename T>
  static std::string GetName(int) {
    return std::to_string(T::dim) + "D";
  }
};

template <typename T>
class DetectCollisionTest : public testing::Test {
 protected:
  DetectCollisionTest() {}

  ~DetectCollisionTest() {}
};

using DetectCollisionTypes =
    testing::Types<DetectCollision_2D, DetectCollision_3D>;
TYPED_TEST_SUITE(DetectCollisionTest, DetectCollisionTypes,
                 DetectCollisionNameGenerator);

TYPED_TEST(DetectCollisionTest, CollisionAssertion) {
  constexpr int dim = TypeParam::dim;

  // Qhull computations can be unstable with float.
  if ((dim == 3) && (typeid(Real) == typeid(float))) GTEST_SKIP();

  Rng rng;
  rng.SetSeed();
  const int nsamples_cold = 100;
  const int nsamples_warm = 200;

  ConvexSetPtr<dim> set1, set2;
  SetConvexSets<dim>(set1, set2, Real(0.1), Real(0.1));

  // Check collisions for random transformations.
  Transformr<dim> tf1, tf2;
  Settings settings;
  Output<dim> out;
  const Real dt = Real(0.1);

  for (int i = 0; i < nsamples_cold; ++i) {
    rng.RandomTransform(Real(-5.0), Real(5.0), tf1);
    rng.RandomTransform(Real(-5.0), Real(5.0), tf2);
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
  const Real gd = GrowthDistanceHalfspace(setc.get(), tfc, seth.get(), tfh,
                                          settings, out, false);
  ASSERT_TRUE(out.status == SolutionStatus::Optimal);
  ASSERT_NEAR(gd, Real(0.2), kTol);

  Affine(tfc) = Vec3r(Real(8.0), Real(-7.0), Real(-0.1));
  GrowthDistanceHalfspace(setc.get(), tfc, seth.get(), tfh, settings, out,
                          false);
  ASSERT_TRUE(out.status == SolutionStatus::CoincidentCenters);
}

}  // namespace
