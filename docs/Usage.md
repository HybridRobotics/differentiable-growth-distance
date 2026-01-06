# Usage Guide

The public interface (headers under `include/dgd/`) is documented and contains the full API details. This brief guide outlines common workflows and highlights the headers to include for each task.

---

## 1) Convex sets

Headers to include:
- Minimal: `#include "dgd/geometry/convex_set.h"`
- Concrete types (pick as needed):
  - 2D: `#include "dgd/geometry/2d/ellipse.h"`, `.../polygon.h`, `.../rectangle.h`, etc.
  - 3D: `#include "dgd/geometry/3d/cone.h"`, `.../mesh.h`, `.../ellipsoid.h`, etc.
  - 2D/3D (for capsules and spheres): `#include "dgd/geometry/xd/capsule.h"`, `.../sphere.h`.
- Halfspace: `#include "dgd/geometry/halfspace.h"`

Example (2D ellipse and polygon):
```cpp
#include "dgd/geometry/2d/ellipse.h"
#include "dgd/geometry/2d/polygon.h"
using namespace dgd;

auto a = std::make_unique<Ellipse>(/*hlx=*/1.0, /*hly=*/0.5, /*margin=*/0.05);
std::vector<Vec2r> verts = /* build or load vertices */;
auto b = std::make_unique<Polygon>(verts, /*inradius=*/0.1, /*margin=*/0.0);
```

### Notes

- **Base transforms:** Concrete types store their geometry in their own local frame. Check the documentation for each concrete type to determine its base frame and center point.

- **User-defined convex sets:** To add a custom set, derive from `ConvexSet<dim>` and implement the required virtual function:
 ```cpp
  Real SupportFunction(const Vecr<dim>&, Vecr<dim>&, SupportFunctionHint<dim>*);
 ```
 and, optionally, other virtual functions. See the existing implementations (e.g., `Ellipse`, `Polygon`, `Cone`, `Mesh`) for examples.

---

## 2) Convex hulls, mesh import, and mesh graphs

Convex hull of 2D/3D points can be computed as follows:

- Graham scan (convex hull in 2D):
  - Header: `#include "dgd/graham_scan.h"`
  - Use `GrahamScan(input_points, output_hull)` to compute a 2D convex hull (in CCW orientation) from a point cloud.

- Mesh import (from .obj files), convex hull in 3D, and adjacency graph computation:
  - Header: `#include "dgd/mesh_loader.h"`
  - Use `MeshLoader` to import points (or to process an in-memory vertex list).
  - Example workflow:
    1. Fill `std::vector<Vec3r> pts` (or call `MeshLoader::LoadObj` if .obj file is available, and skip step 2).
    2. Call `ml.ProcessPoints(pts)` (removes duplicates).
    3. Build convex hull and vertex adjacency graph: </br>
      `ml.MakeVertexGraph(vert, graph)` → returns the vertex list and adjacency graph. </br>
 Build convex hull and facet adjacency graph: </br>
      `ml.MakeFacetGraph(normal, offset, graph, interior-point)` → returns the facet list (normals and offsets), adjacency graph, and an interior point. </br>
    4. Compute inradius at the center point: `ml.ComputeInradius(center)`, etc.

**Important:** The center point returned by `MeshLoader` (or used internally) is not guaranteed to be the world origin — check the loader's returned center point and adjust accordingly.

---

## 3) Growth distance computation & collision detection

Public runtime-dispatch API (cheap to include):
- Header: `#include "dgd/dgd.h"`
- Runtime (virtual-dispatch) functions:
  - `dgd::GrowthDistance<dim>(const ConvexSet<dim>*, const Transformr<dim>&, const ConvexSet<dim>*, const Transformr<dim>&, const Settings&, Output<dim>&, bool warm_start)`
  - `dgd::DetectCollision<dim>(...)`

Header-only, zero-virtual-cost templates (opt-in):
- Header: `#include "dgd/solvers/bundle_scheme_impl.h"`
- Call:
  - `dgd::GrowthDistanceCpTpl<dim, Concrete1, Concrete2>(...)`
  - `dgd::DetectCollisionTpl<dim, Concrete1, Concrete2>(...)`
- **Note:** These templates are heavy; include `bundle_scheme_impl.h` only in translation units where you need peak performance. The library does not provide pre-instantiated `Tpl` functions — including the header is required.

Simple example (runtime API):
```cpp
#include "dgd/dgd.h"
#include "dgd/geometry/2d/ellipse.h"
#include "dgd/geometry/2d/polygon.h"
using namespace dgd;

Ellipse a(...);
Polygon b(...);
Transform2r ta = Transform2r::Identity();
Transform2r tb = Transform2r::Identity();
Linear(tb) = /* set rotation matrix */;
Affine(tb) = /* set translation vector */;
Settings settings;
Output<2> out;
Real gd = GrowthDistance(&a, ta, &b, tb, settings, out, /*warm_start=*/false);

bool collision = DetectCollision(&a, ta, &b, tb, settings, out, false);
```

Example (header-only templated API):
```cpp
#include "dgd/solvers/bundle_scheme_impl.h" // heavy; opt-in
#include "dgd/geometry/2d/ellipse.h"
#include "dgd/geometry/2d/polygon.h"
using namespace dgd;

Ellipse a(...);
Polygon b(...);
Transform2r ta = Transform2r::Identity();
Transform2r tb = Transform2r::Identity();
Linear(tb) = /* set rotation matrix */;
Affine(tb) = /* set translation vector */;
Settings settings;
Output<2> out;
Real gd = GrowthDistanceCpTpl(&a, ta, &b, tb, settings, out, /*warm_start=*/false);

bool collision = DetectCollisionTpl(&a, ta, &b, tb, settings, out, false);
```

Two available solver algorithms (currently):
- *Cutting plane* (`GrowthDistanceCp`):
  - Best for polytopic sets (with zero margin), and robust.
- *Trust region Newton* (`GrowthDistanceTrn`):
  - Second-order solver best for sets with curved surfaces.

### Notes

Solver / warm-start guidance (broad categories):

| Set pair category         | Preferred solver | Warm-start type |
|:--------------------------|:-----------------|:-------------------------|
| Polytope + Polytope       | Cutting plane | Warm start: Primal |
| Curved set + Any set   | Trust region Newton | Warm start: Dual or problem-dependent |

- The code exposes both solver entry points (see the `dgd/dgd.h` and `dgd/solvers/bundle_scheme_impl.h` headers).
- Warm-start effectiveness depends on the solver, warm start type, and problem instance.

**For the runtime API:**

- Warm start is turned off for the trust region Newton solver, as it can be slower than cold start.
- The function `GrowthDistance` automatically selects a solver based on the problem instance. For more granular control, use the runtime solver entry points or the header-only, templated API.

**For the header-only templated API:**

- Warm start is not disabled for the trust region Newton solver, but note that it can result in a significant primal infeasibility error (~10e-7).

---

## 4) Solution error computation

Header: `#include "dgd/error_metrics.h"`

Use the provided helpers to validate solver outputs (primal/dual infeasibility and primal-dual gap). Example:
```cpp
#include "dgd/error_metrics.h"

SolutionError err = ComputeSolutionError(&a, ta, &b, tb, out);
std::cout << "primal infeas: " << err.prim_infeas_err
 << " dual gap: " << err.prim_dual_gap << "\n";
```

Note that, due to the nature of the algorithms, the dual infeasibility error is always zero.

---

## Examples

- See `tests/` and `benchmarks/` for example usages (both runtime and header-only usages are present).

Full example:

```cpp
#include <iostream>
#include <memory>

#include "dgd/dgd.h"                     // runtime API (small header)
#include "dgd/error_metrics.h"
#include "dgd/utils/transformations.h"
#include "dgd/geometry/2d/ellipse.h"
#include "dgd/geometry/2d/polygon.h"
#include "dgd/geometry/3d/cone.h"
#include "dgd/geometry/3d/ellipsoid.h"

// OPTIONAL: include this only if you want the header-only, non-virtual API.
// It is heavy; include in translation units where you need max performance and the concrete types are known.
// #include "dgd/solvers/bundle_scheme_impl.h"

using namespace dgd;

int main() {
  try {
    // 2D example (runtime API)
    {
      Ellipse a(3.0, 1.5, /*margin=*/0.05);
      std::vector<Vec2r> verts = {Vec2r(1.0, 0.0), Vec2r(0.0, 1.0),
                                  Vec2r(-1.0, 0.0), Vec2r(0.0, -1.0)};
      Polygon b(verts, /*inradius=*/0.8, /*margin=*/0.0);

      Transform2r ta = Transform2r::Identity();
      Transform2r tb = Transform2r::Identity();
      Affine(tb) += Vec2r::Constant(4.0);
      Settings settings;
      Output<2> out;

      // runtime-dispatch call (small header, virtual dispatch)
      Real gd = GrowthDistance(&a, ta, &b, tb, settings, out, false);
      std::cout << "[2D] growth distance = " << gd << '\n';

      // solution error metrics
      SolutionError err = ComputeSolutionError(&a, ta, &b, tb, out);
      std::cout << "[2D] primal infeas = " << err.prim_infeas_err
                << "     prim-dual gap = " << err.prim_dual_gap << '\n';

      // collision detection example
      bool coll = DetectCollision(&a, ta, &b, tb, settings, out, false);
      std::cout << "[2D] collision? " << (coll ? "yes" : "no") << '\n';
    }

    // 3D example (runtime API)
    {
      Cone a(/*radius=*/1.0, /*height=*/2.0, /*margin=*/0.05);
      Ellipsoid b(/*rx=*/0.8, /*ry=*/0.6, /*rz=*/1.2, /*margin=*/0.05);

      Transform3r ta = Transform3r::Identity();
      Transform3r tb = Transform3r::Identity();
      Affine(tb) += Vec3r::Constant(0.1);
      Settings settings;
      Output<3> out;

      // runtime-dispatch
      Real gd = GrowthDistance(&a, ta, &b, tb, settings, out, false);
      std::cout << "[3D] growth distance = " << gd << '\n';

      // solution error metrics
      SolutionError err = ComputeSolutionError(&a, ta, &b, tb, out);
      std::cout << "[3D] primal infeas = " << err.prim_infeas_err
                << "     prim dual gap = " << err.prim_dual_gap << '\n';

      // collision detection example
      bool coll = DetectCollision(&a, ta, &b, tb, settings, out, false);
      std::cout << "[3D] collision? " << (coll ? "yes" : "no") << '\n';
    }
  } catch (const std::exception& e) {
    std::cerr << "example failed: " << e.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
```
