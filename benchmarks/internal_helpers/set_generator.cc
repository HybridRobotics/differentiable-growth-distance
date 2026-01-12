#include "internal_helpers/set_generator.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "dgd/geometry/geometry_2d.h"
#include "dgd/geometry/geometry_3d.h"
#include "dgd/graham_scan.h"
#include "dgd/utils/random.h"
#include "internal_helpers/mesh_loader.h"

namespace dgd {

namespace bench {

namespace {

template <int dim>
inline void CenterVertices(std::vector<Vecr<dim>>& vert) {
  Vecr<dim> center = Vecr<dim>::Zero();
  for (const auto& v : vert) center += v;
  center /= static_cast<int>(vert.size());
  for (auto& v : vert) v -= center;
}

}  // namespace

ConvexSetGenerator::ConvexSetGenerator(ConvexSetFeatureRange fr)
    : fr_(std::move(fr)),
      count2_(static_cast<int>(Primitive2D::Count_)),
      ccount3_(static_cast<int>(CurvedPrimitive3D::Count_)),
      fcount3_(static_cast<int>(FlatPrimitive3D::Count_)) {
  SetRngSeed();
  if ((fr_.polytope.nvert[0] < 4) || (fr_.polygon.nvert[0] < 3)) {
    throw std::invalid_argument(
        "Polytopes (polygons) must have at least 4 (3) vertices.");
  }
  meshes_.clear();
  nmeshes_ = 0;
}

void ConvexSetGenerator::LoadMeshesFromObjFiles(
    const std::vector<std::string>& filenames) {
  meshes_.clear();

  MeshProperties mp{};
  for (const auto& filename : filenames) {
    try {
      mp.SetVertexMeshFromObjFile(filename);
    } catch (const std::runtime_error& e) {
      continue;
    }
    mp.SetZeroVertexCenter();

    meshes_.push_back(std::make_shared<Mesh>(std::move(mp.vert),
                                             std::move(mp.vgraph), mp.inradius,
                                             0.0, 0.9, 1, mp.name));
  }
  nmeshes_ = static_cast<int>(meshes_.size());
}

inline Real ConvexSetGenerator::GetMargin() {
  return rng_.CoinFlip(fr_.pos_margin_prob) ? rng_.Random({0.0, fr_.margin})
                                            : 0.0;
}

ConvexSetPtr<2> ConvexSetGenerator::GetPrimitiveSet(Primitive2D type) {
  const Real margin = GetMargin();

  switch (type) {
    case Primitive2D::Ellipse: {
      const Real hlx = rng_.Random(fr_.Range(fr_.ellipse.hlx));
      const Real hly = rng_.Random(fr_.Range(fr_.ellipse.hly));
      return std::make_shared<Ellipse>(hlx, hly, margin);
    }

    case Primitive2D::Polygon: {
      polygon_vert_.clear();
      polygon_vert_.reserve(fr_.polygon.nvert[1]);

      const int nvert = rng_.RandomInt(fr_.polygon.nvert);
      Vec2r v;
      for (int i = 0; i < nvert; ++i) {
        v = Vec2r(rng_.Random(), rng_.Random());
        v = v * fr_.polygon.size / (v.lpNorm<4>() + kEps);
        polygon_vert_.push_back(v);
      }
      CenterVertices(polygon_vert_);

      return std::make_shared<Polygon>(std::move(polygon_vert_), kEps, margin);
    }

    case Primitive2D::Rectangle: {
      const Real hlx = rng_.Random(fr_.Range(fr_.rectangle.hlx));
      const Real hly = rng_.Random(fr_.Range(fr_.rectangle.hly));
      return std::make_shared<Rectangle>(hlx, hly, margin);
    }

    case Primitive2D::Stadium: {
      const Real hlx = rng_.Random(fr_.Range(fr_.capsule.hlx));
      const Real radius = rng_.Random(fr_.Range(fr_.capsule.radius));
      return std::make_shared<Stadium>(hlx, radius, margin);
    }

    case Primitive2D::Circle: {
      const Real radius = rng_.Random(fr_.Range(fr_.sphere.radius));
      return std::make_shared<Circle>(radius);
    }

    default:
      throw std::invalid_argument("Invalid 2D primitive type");
  }
}

ConvexSetPtr<3> ConvexSetGenerator::GetPrimitiveSet(CurvedPrimitive3D type) {
  const Real margin = GetMargin();

  switch (type) {
    case CurvedPrimitive3D::Capsule: {
      const Real hlx = rng_.Random(fr_.Range(fr_.capsule.hlx));
      const Real radius = rng_.Random(fr_.Range(fr_.capsule.radius));
      return std::make_shared<Capsule>(hlx, radius, margin);
    }

    case CurvedPrimitive3D::Cone: {
      const Real radius = rng_.Random(fr_.Range(fr_.cone.radius));
      const Real height = rng_.Random(fr_.Range(fr_.cone.height));
      return std::make_shared<Cone>(radius, height, margin);
    }

    case CurvedPrimitive3D::Cylinder: {
      const Real hlx = rng_.Random(fr_.Range(fr_.cylinder.hlx));
      const Real radius = rng_.Random(fr_.Range(fr_.cylinder.radius));
      return std::make_shared<Cylinder>(hlx, radius, margin);
    }

    case CurvedPrimitive3D::Ellipsoid: {
      const Real hlx = rng_.Random(fr_.Range(fr_.ellipsoid.hlx));
      const Real hly = rng_.Random(fr_.Range(fr_.ellipsoid.hly));
      const Real hlz = rng_.Random(fr_.Range(fr_.ellipsoid.hlz));
      return std::make_shared<Ellipsoid>(hlx, hly, hlz, margin);
    }

    case CurvedPrimitive3D::Frustum: {
      const Real base_radius = rng_.Random(fr_.Range(fr_.frustum.base_radius));
      const Real top_radius = rng_.Random(fr_.Range(fr_.frustum.top_radius));
      const Real height = rng_.Random(fr_.Range(fr_.frustum.height));
      return std::make_shared<Frustum>(base_radius, top_radius, height, margin);
    }

    case CurvedPrimitive3D::Sphere: {
      const Real radius = rng_.Random(fr_.Range(fr_.sphere.radius));
      return std::make_shared<Sphere>(radius);
    }

    default:
      throw std::invalid_argument("Invalid curved 3D primitive type");
  }
}

ConvexSetPtr<3> ConvexSetGenerator::GetPrimitiveSet(FlatPrimitive3D type) {
  Real margin = GetMargin();

  switch (type) {
    case FlatPrimitive3D::Cuboid: {
      const Real hlx = rng_.Random(fr_.Range(fr_.cuboid.hlx));
      const Real hly = rng_.Random(fr_.Range(fr_.cuboid.hly));
      const Real hlz = rng_.Random(fr_.Range(fr_.cuboid.hlz));
      return std::make_shared<Cuboid>(hlx, hly, hlz, margin);
    }

    case FlatPrimitive3D::Polytope: {
      polytope_vert_.clear();
      polytope_vert_.reserve(fr_.polytope.nvert[1]);

      if (meshes_.empty()) {
        const int nvert = rng_.RandomInt(fr_.polytope.nvert);
        Vec3r v;
        for (int i = 0; i < nvert; ++i) {
          v = Vec3r(rng_.Random(), rng_.Random(), rng_.Random());
          v = v * fr_.polytope.size / (v.lpNorm<4>() + kEps);
          polytope_vert_.push_back(v);
        }
      } else {
        const int mesh_idx = rng_.RandomInt({0, nmeshes_ - 1});
        const int nvert_m = meshes_[mesh_idx]->nvertices();
        for (int i = 0; i < 4; ++i) {
          polytope_vert_.push_back(
              meshes_[mesh_idx]->vertices()[(i * (nvert_m - 1)) / 3]);
        }
        for (int i = 4; i < fr_.polytope.nvert[1]; ++i) {
          const int idx = rng_.RandomInt({0, nvert_m - 1});
          polytope_vert_.push_back(meshes_[mesh_idx]->vertices()[idx]);
        }
      }
      CenterVertices(polytope_vert_);

      // Add a small margin to ensure that the set is solid.
      margin += 1e-2;
      return std::make_shared<Polytope>(std::move(polytope_vert_), kSqrtEps,
                                        margin);
    }

    default:
      throw std::invalid_argument("Invalid flat 3D primitive type");
  }
}

ConvexSetPtr<3> ConvexSetGenerator::GetRandomCurvedPrimitive3DSet() {
  const int idx = rng_.RandomInt({0, ccount3_ - 1});
  return GetPrimitiveSet(static_cast<CurvedPrimitive3D>(idx));
}

ConvexSetPtr<3> ConvexSetGenerator::GetRandomPrimitive3DSet() {
  const int idx = rng_.RandomInt({0, ccount3_ + fcount3_ - 1});
  if (idx < ccount3_) {
    return GetPrimitiveSet(static_cast<CurvedPrimitive3D>(idx));
  } else {
    return GetPrimitiveSet(static_cast<FlatPrimitive3D>(idx - ccount3_));
  }
}

ConvexSetPtr<3> ConvexSetGenerator::GetRandomMeshSet(int* idx) {
  if (meshes_.empty()) {
    throw std::runtime_error(
        "No meshes available to retrieve a random mesh set");
  }

  const int mesh_idx = rng_.RandomInt({0, nmeshes_ - 1});
  if (idx) *idx = mesh_idx;
  return meshes_[mesh_idx];
}

ConvexSetPtr<2> ConvexSetGenerator::GetRandom2DSet() {
  const int idx = rng_.RandomInt({0, count2_ - 1});
  return GetPrimitiveSet(static_cast<Primitive2D>(idx));
}

ConvexSetPtr<3> ConvexSetGenerator::GetRandom3DSet() {
  const int idx = rng_.RandomInt({0, ccount3_ + fcount3_ + nmeshes_ - 1});
  if (idx < ccount3_ + fcount3_) {
    return GetRandomPrimitive3DSet();
  } else {
    return GetRandomMeshSet();
  }
}

inline std::vector<Vec3r> ConvexSetGenerator::GetRandomEllipsoidVertices(
    int nvert, const Vec3r& half_lengths) {
  if (half_lengths.minCoeff() <= 0.0) {
    throw std::domain_error("Half-lengths are not positive.");
  }

  std::vector<Vec3r> vert(nvert);
  for (int i = 0; i < nvert; ++i) {
    vert[i] = rng_.RandomUnitVector<3>().cwiseProduct(half_lengths);
  }
  return vert;
}

template <class T>
ConvexSetPtr<3> ConvexSetGenerator::GetRandomEllipsoidalPolytope(int nvert,
                                                                 Real skew) {
  if ((skew < 1e-3) || (skew > 1.0)) throw std::domain_error("Invalid skew");

  Vec3r half_lengths = Vec3r::Constant(fr_.sph_polytope.radius);
  half_lengths[0] *= skew;
  half_lengths[1] *= std::exp2(rng_.Random({std::log2(skew), 0.0}));
  const auto vert = GetRandomEllipsoidVertices(nvert, half_lengths);

  MeshProperties mp{};
  try {
    mp.SetVertexMeshFromVertices(vert);
  } catch (const std::runtime_error& e) {
    throw std::runtime_error(std::string("Failed to set vertex mesh: ") +
                             e.what());
  }
  mp.SetZeroVertexCenter();

  const Real margin = 0.0;
  if constexpr (std::is_same<T, Mesh>::value) {
    return std::make_shared<Mesh>(std::move(mp.vert), std::move(mp.vgraph),
                                  mp.inradius, margin);
  } else if constexpr (std::is_same<T, Polytope>::value) {
    return std::make_shared<Polytope>(std::move(mp.vert), mp.inradius, margin);
  } else {
    throw std::invalid_argument("Invalid polytopic type");
  }
}

template ConvexSetPtr<3> ConvexSetGenerator::GetRandomEllipsoidalPolytope<Mesh>(
    int nvert, Real skew);

template ConvexSetPtr<3>
ConvexSetGenerator::GetRandomEllipsoidalPolytope<Polytope>(int nvert,
                                                           Real skew);

}  // namespace bench

}  // namespace dgd
