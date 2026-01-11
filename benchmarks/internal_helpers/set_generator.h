#ifndef DGD_BENCHMARKS_INTERNAL_HELPERS_SET_GENERATOR_H_
#define DGD_BENCHMARKS_INTERNAL_HELPERS_SET_GENERATOR_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "dgd/data_types.h"
#include "dgd/utils/random.h"

namespace dgd {

// Forward declarations.
template <int dim>
class ConvexSet;

class Mesh;

namespace bench {

// Type definitions for convenience.
template <int dim>
using ConvexSetPtr = std::shared_ptr<ConvexSet<dim>>;
using MeshPtr = std::shared_ptr<Mesh>;

// Primitive enumerations.
enum class Primitive2D {
  Ellipse,
  Polygon,
  Rectangle,
  Stadium,
  Circle,
  Count_,
};

enum class CurvedPrimitive3D {
  Capsule,
  Cone,
  Cylinder,
  Ellipsoid,
  Frustum,
  Sphere,
  Count_,
};

enum class FlatPrimitive3D {
  Cuboid,
  Polytope,
  Count_,
};

// Feature dimension ranges for convex sets.
struct ConvexSetFeatureRange {
  // 2D convex sets.
  struct {
    Real hlx = 0.25;
    Real hly = 0.25;
  } ellipse;

  struct {
    Real hlx = 0.25;
    Real hly = 0.25;
  } rectangle;

  struct {
    std::array<int, 2> nvert{6, 16};
    Real size = 0.5;
  } polygon;

  // 3D convex sets.
  struct {
    Real radius = 0.25;
    Real height = 0.5;
  } cone;

  struct {
    Real hlx = 0.4;
    Real radius = 0.25;
  } cylinder;

  struct {
    Real hlx = 0.25;
    Real hly = 0.25;
    Real hlz = 0.25;
  } ellipsoid;

  struct {
    Real base_radius = 0.25;
    Real top_radius = 0.25;
    Real height = 0.5;
  } frustum;

  struct {
    Real hlx = 0.25;
    Real hly = 0.25;
    Real hlz = 0.25;
  } cuboid;

  struct {
    std::array<int, 2> nvert{8, 32};
    Real size = 0.4;
  } polytope;

  // XD convex sets.
  struct {
    Real hlx = 0.25;
    Real radius = 0.25;
  } capsule;

  struct {
    Real radius = 0.25;
  } sphere;

  // Spherical polytopes.
  struct {
    Real radius = 0.25;
  } sph_polytope;

  Real margin = 0.25;
  // Probability of positive margin.
  Real pos_margin_prob = 0.5;

  // Valid values for the scale are in the interval [1e-3, 1.0]. If a value
  // outside this range is provided, the call is a no-op and the existing value
  // of the scale is left unchanged.
  void SetScale(Real scale) {
    if ((scale <= 1.0) && (scale >= 1e-3)) scale_ = scale;
  }

  Real scale() const { return scale_; }

  std::array<Real, 2> Range(Real ub) const { return {scale_ * ub, ub}; }

 private:
  Real scale_ = 1e-2;
};

// Convex set generator.
class ConvexSetGenerator {
 public:
  explicit ConvexSetGenerator(ConvexSetFeatureRange fr);

  ~ConvexSetGenerator() = default;

  // Sets an RNG seed.
  void SetRngSeed(unsigned int seed = 5489u) { rng_.SetSeed(seed); };

  // Sets a true random seed.
  void SetRandomRngSeed() { rng_.SetRandomSeed(); };

  // Loads meshes from .obj files.
  void LoadMeshesFromObjFiles(const std::vector<std::string>& filenames);

  // Returns a random margin value.
  Real GetMargin();

  // Generates a random convex set of a primitive type.
  ConvexSetPtr<2> GetPrimitiveSet(Primitive2D type);

  ConvexSetPtr<3> GetPrimitiveSet(CurvedPrimitive3D type);

  ConvexSetPtr<3> GetPrimitiveSet(FlatPrimitive3D type);

  // Generates random primitive convex sets.
  ConvexSetPtr<3> GetRandomCurvedPrimitive3DSet();

  ConvexSetPtr<3> GetRandomPrimitive3DSet();

  // Retrieves a random Mesh set using an index.
  ConvexSetPtr<3> GetRandomMeshSet(int* idx = nullptr);

  // Generates a random 2D convex set.
  ConvexSetPtr<2> GetRandom2DSet();

  // Generates a random 3D convex set.
  ConvexSetPtr<3> GetRandom3DSet();

  // Generates random vertices on an ellipsoid.
  std::vector<Vec3r> GetRandomEllipsoidVertices(int nvert,
                                                const Vec3r& half_lengths);

  // Generates random ellipsoidal polytope/mesh.
  template <class T>
  ConvexSetPtr<3> GetRandomEllipsoidalPolytope(int nvert, Real skew);

  ConvexSetFeatureRange& feature_range() { return fr_; }

  const std::vector<MeshPtr>& meshes() const { return meshes_; }

  int nmeshes() const { return nmeshes_; }

 private:
  std::vector<MeshPtr> meshes_;
  std::vector<Vec2r> polygon_vert_;   // Temporary.
  std::vector<Vec3r> polytope_vert_;  // Temporary.
  ConvexSetFeatureRange fr_;
  Rng rng_;
  const int count2_, ccount3_, fcount3_;
  int nmeshes_;
};

}  // namespace bench

}  // namespace dgd

#endif  // DGD_BENCHMARK_CONVEX_SET_GENERATOR_H_
