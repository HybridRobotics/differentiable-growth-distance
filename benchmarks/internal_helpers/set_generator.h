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

namespace internal {

// Type definitions for convenience.
template <int dim>
using ConvexSetPtr = std::shared_ptr<ConvexSet<dim>>;
using MeshPtr = std::shared_ptr<Mesh>;

// Primitive enumerations.
enum class Primitive2D {
  Ellipse,
  // Polygon,
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
  // Polytope,
  Count_,
};

struct ConvexSetFeatureRange {
  // 2D convex sets.
  struct {
    std::array<Real, 2> hlx{0.25 * 1e-2, 0.25};
    std::array<Real, 2> hly{0.25 * 1e-2, 0.25};
  } ellipse;

  struct {
    std::array<Real, 2> hlx{0.25 * 1e-2, 0.25};
    std::array<Real, 2> hly{0.25 * 1e-2, 0.25};
  } rectangle;

  struct {
    std::array<int, 2> nvert{6, 16};
    Real size = 0.4;
  } polygon;

  // 3D convex sets.
  struct {
    std::array<Real, 2> radius{0.25 * 1e-2, 0.25};
    std::array<Real, 2> height{0.5 * 1e-2, 0.5};
  } cone;

  struct {
    std::array<Real, 2> hlx{0.4 * 1e-2, 0.4};
    std::array<Real, 2> radius{0.25 * 1e-2, 0.25};
  } cylinder;

  struct {
    std::array<Real, 2> hlx{0.25 * 1e-2, 0.25};
    std::array<Real, 2> hly{0.25 * 1e-2, 0.25};
    std::array<Real, 2> hlz{0.25 * 1e-2, 0.25};
  } ellipsoid;

  struct {
    std::array<Real, 2> base_radius{0.25 * 1e-2, 0.25};
    std::array<Real, 2> top_radius{0.25 * 1e-2, 0.25};
    std::array<Real, 2> height{0.5 * 1e-2, 0.5};
  } frustum;

  struct {
    std::array<Real, 2> hlx{0.25 * 1e-2, 0.25};
    std::array<Real, 2> hly{0.25 * 1e-2, 0.25};
    std::array<Real, 2> hlz{0.25 * 1e-2, 0.25};
  } cuboid;

  struct {
    std::array<int, 2> nvert{8, 32};
    Real size = 0.4;
  } polytope;

  // XD convex sets.
  struct {
    std::array<Real, 2> hlx{0.25 * 1e-2, 0.25};
    std::array<Real, 2> radius{0.25 * 1e-2, 0.25};
  } capsule;

  struct {
    std::array<Real, 2> radius{0.25 * 1e-2, 0.25};
  } sphere;
};

// Convex set generator.
class ConvexSetGenerator {
 public:
  explicit ConvexSetGenerator(const ConvexSetFeatureRange& fr);

  ~ConvexSetGenerator() = default;

  // Sets default RNG seed.
  void SetDefaultRngSeed() { rng_.SetDefaultSeed(); };

  // Load meshes from .obj files.
  void LoadMeshesFromObjFiles(const std::vector<std::string>& filenames);

  // Generate a random convex set of a primitive type.
  ConvexSetPtr<2> GetPrimitiveSet(Primitive2D type);

  ConvexSetPtr<3> GetPrimitiveSet(CurvedPrimitive3D type);

  ConvexSetPtr<3> GetPrimitiveSet(FlatPrimitive3D type);

  // Get random primitive convex sets.
  ConvexSetPtr<3> GetRandomCurvedPrimitive3DSet();

  ConvexSetPtr<3> GetRandomPrimitive3DSet();

  // Retrieve a random Mesh set with index.
  ConvexSetPtr<3> GetRandomMeshSet(int* idx = nullptr);

  // Get a random convex set.
  ConvexSetPtr<2> GetRandom2DSet();

  ConvexSetPtr<3> GetRandom3DSet();

  const std::vector<MeshPtr>& meshes() const { return meshes_; }

  int nmeshes() const { return nmeshes_; }

 private:
  std::vector<MeshPtr> meshes_;
  /*
  std::vector<Vec2r> polygon_vert_; // Temporary.
  std::vector<Vec3r> polytope_vert_; // Temporary.
  */
  const ConvexSetFeatureRange fr_;
  Rng rng_;
  const int count2_, ccount3_, fcount3_;
  int nmeshes_;
};

}  // namespace internal

}  // namespace dgd

#endif  // DGD_BENCHMARK_CONVEX_SET_GENERATOR_H_
