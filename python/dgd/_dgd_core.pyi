"""
Differentiable growth distance algorithm for convex sets.
"""
from __future__ import annotations
import numpy
from numpy.typing import NDArray
import typing
__all__: list[str] = ['BasePointHint2', 'BasePointHint3', 'Capsule', 'Circle', 'Cone', 'ConvexSet2', 'ConvexSet3', 'Cuboid', 'Cylinder', 'DirectionalDerivative2', 'DirectionalDerivative3', 'EPS', 'Ellipse', 'Ellipsoid', 'Frustum', 'Halfspace2', 'Halfspace3', 'INF', 'KinematicState2', 'KinematicState3', 'Mesh', 'MeshLoader', 'NormalConeSpan2', 'NormalConeSpan3', 'NormalPair2', 'NormalPair3', 'Output2', 'Output3', 'PI', 'Polygon', 'Polytope', 'Rectangle', 'SQRT_EPS', 'Settings', 'SolutionError', 'SolutionStatus', 'Sphere', 'Stadium', 'SupportFunctionHint2', 'SupportFunctionHint3', 'SupportPatchHull2', 'SupportPatchHull3', 'TotalDerivative2', 'TotalDerivative3', 'TwistFrame', 'WarmStartType', 'assert_collision_status', 'compute_kkt_nullspace', 'compute_polygon_inradius', 'compute_solution_error', 'detect_collision', 'factorize_kkt_system', 'gd_derivative', 'gd_gradient', 'gd_jacobian', 'gd_solution_derivative', 'graham_scan', 'growth_distance', 'growth_distance_cp', 'growth_distance_trn']
class BasePointHint2:
    """
    Base point hint.
    """
    def __init__(self) -> None:
        ...
    @property
    def bc(self) -> NDArray[numpy.float64]:
        """
        Barycentric coordinates for the base point corresponding to the simplex.
        """
    @bc.setter
    def bc(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def idx(self) -> NDArray[numpy.int32]:
        """
        Index hints for the base point corresponding to the simplex.
        """
    @idx.setter
    def idx(self, arg0: NDArray[numpy.int32]) -> None:
        ...
class BasePointHint3:
    """
    Base point hint.
    """
    def __init__(self) -> None:
        ...
    @property
    def bc(self) -> NDArray[numpy.float64]:
        """
        Barycentric coordinates for the base point corresponding to the simplex.
        """
    @bc.setter
    def bc(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def idx(self) -> NDArray[numpy.int32]:
        """
        Index hints for the base point corresponding to the simplex.
        """
    @idx.setter
    def idx(self, arg0: NDArray[numpy.int32]) -> None:
        ...
class Capsule(ConvexSet3):
    """
    3D capsule.
    """
    def __init__(self, hlx: float, radius: float, margin: float = 0.0) -> None:
        ...
class Circle(ConvexSet2):
    """
    2D circle.
    """
    def __init__(self, radius: float) -> None:
        ...
class Cone(ConvexSet3):
    """
    3D cone.
    """
    def __init__(self, radius: float, height: float, margin: float = 0.0) -> None:
        ...
    def offset(self) -> float:
        """
        Returns the base z-offset.
        """
class ConvexSet2:
    """
    Abstract convex set interface.
    """
    @staticmethod
    def dimension() -> int:
        """
        Returns the convex set dimension.
        """
    @staticmethod
    def eps_sp() -> float:
        """
        Returns the support point differentiability tolerance.
        """
    def bounds(self) -> tuple:
        """
        Computes the AABB bounds. Returns (diagonal, min, max).
        """
    def compute_local_geometry(self, z: NDArray[numpy.float64], n: NDArray[numpy.float64], hint: BasePointHint2 = None) -> tuple:
        """
        Computes the local geometry at a base point-normal vector pair. Returns (support_patch_hull, normal_cone_span).
        """
    def is_polytopic(self) -> bool:
        """
        Returns True if the set is polytopic.
        """
    def print_info(self) -> None:
        """
        Prints information about the convex set.
        """
    def projection_derivative(self, p: NDArray[numpy.float64], pi: NDArray[numpy.float64], hint: BasePointHint2 = None) -> tuple:
        """
        Computes the derivative of the projection map at an exterior point. Returns (differentiable, d_pi_p).
        """
    def require_unit_normal(self) -> bool:
        """
        Returns the normalization requirement for support normals.
        """
    def support_function(self, n: NDArray[numpy.float64], hint: SupportFunctionHint2 = None) -> tuple:
        """
        Computes the support function value and a support point at n. Returns (sv, sp).
        """
    def support_function_hess(self, n: NDArray[numpy.float64], hint: SupportFunctionHint2 = None) -> tuple:
        """
        Computes the support function value and its derivatives at n. Returns (sv, sp, differentiable, d_sp_n).
        """
    @property
    def eps_d(self) -> float:
        """
        Dual solution geometry tolerance.
        """
    @eps_d.setter
    def eps_d(self, arg1: float) -> None:
        ...
    @property
    def eps_p(self) -> float:
        """
        Primal solution geometry tolerance.
        """
    @eps_p.setter
    def eps_p(self, arg1: float) -> None:
        ...
    @property
    def inradius(self) -> float:
        """
        Inradius lower bound at the origin.
        """
    @inradius.setter
    def inradius(self, arg1: float) -> None:
        ...
class ConvexSet3:
    """
    Abstract convex set interface.
    """
    @staticmethod
    def dimension() -> int:
        """
        Returns the convex set dimension.
        """
    @staticmethod
    def eps_sp() -> float:
        """
        Returns the support point differentiability tolerance.
        """
    def bounds(self) -> tuple:
        """
        Computes the AABB bounds. Returns (diagonal, min, max).
        """
    def compute_local_geometry(self, z: NDArray[numpy.float64], n: NDArray[numpy.float64], hint: BasePointHint3 = None) -> tuple:
        """
        Computes the local geometry at a base point-normal vector pair. Returns (support_patch_hull, normal_cone_span).
        """
    def is_polytopic(self) -> bool:
        """
        Returns True if the set is polytopic.
        """
    def print_info(self) -> None:
        """
        Prints information about the convex set.
        """
    def projection_derivative(self, p: NDArray[numpy.float64], pi: NDArray[numpy.float64], hint: BasePointHint3 = None) -> tuple:
        """
        Computes the derivative of the projection map at an exterior point. Returns (differentiable, d_pi_p).
        """
    def require_unit_normal(self) -> bool:
        """
        Returns the normalization requirement for support normals.
        """
    def support_function(self, n: NDArray[numpy.float64], hint: SupportFunctionHint3 = None) -> tuple:
        """
        Computes the support function value and a support point at n. Returns (sv, sp).
        """
    def support_function_hess(self, n: NDArray[numpy.float64], hint: SupportFunctionHint3 = None) -> tuple:
        """
        Computes the support function value and its derivatives at n. Returns (sv, sp, differentiable, d_sp_n).
        """
    @property
    def eps_d(self) -> float:
        """
        Dual solution geometry tolerance.
        """
    @eps_d.setter
    def eps_d(self, arg1: float) -> None:
        ...
    @property
    def eps_p(self) -> float:
        """
        Primal solution geometry tolerance.
        """
    @eps_p.setter
    def eps_p(self, arg1: float) -> None:
        ...
    @property
    def inradius(self) -> float:
        """
        Inradius lower bound at the origin.
        """
    @inradius.setter
    def inradius(self, arg1: float) -> None:
        ...
class Cuboid(ConvexSet3):
    """
    3D axis-aligned cuboid.
    """
    def __init__(self, hlx: float, hly: float, hlz: float, margin: float = 0.0) -> None:
        ...
class Cylinder(ConvexSet3):
    """
    3D axis-aligned cylinder.
    """
    def __init__(self, hlx: float, radius: float, margin: float = 0.0) -> None:
        ...
class DirectionalDerivative2:
    """

    Directional (Gateaux) derivatives of the growth distance for 2D sets.

    Attributes
    ----------
    z_nullspace : numpy.ndarray, shape (2,)
        Orthonormal basis for the primal solution set affine hull (world frame).
    n_nullspace : numpy.ndarray, shape (2, 2)
        Orthonormal basis for the dual solution set span (world frame, first
        column is the optimal normal vector).
    d_normal : numpy.ndarray, shape (2,)
        Derivative of the dual solution.
    d_z1, d_z2 : numpy.ndarray, shape (2,)
        Derivatives of the primal solutions.
    d_gd : float
        Derivative of the growth distance.
    z_nullity : int
        Dimension of the primal solution set null space.
    n_nullity : int
        Dimension of the dual solution set null space.
    value_differentiable : bool
        Whether the growth distance value is differentiable.
    differentiable : bool
        Whether the full solution is differentiable.
    """
    def __init__(self) -> None:
        ...
    @property
    def d_gd(self) -> float:
        """
        Derivative of the growth distance.
        """
    @d_gd.setter
    def d_gd(self, arg0: float) -> None:
        ...
    @property
    def d_normal(self) -> NDArray[numpy.float64]:
        """
        Derivative of the dual solution.
        """
    @d_normal.setter
    def d_normal(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def d_z1(self) -> NDArray[numpy.float64]:
        """
        Derivative of the primal solution for set 1.
        """
    @d_z1.setter
    def d_z1(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def d_z2(self) -> NDArray[numpy.float64]:
        """
        Derivative of the primal solution for set 2.
        """
    @d_z2.setter
    def d_z2(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def differentiable(self) -> bool:
        """
        Whether the full solution is differentiable.
        """
    @differentiable.setter
    def differentiable(self, arg0: bool) -> None:
        ...
    @property
    def n_nullity(self) -> int:
        """
        Dimension of the dual null space.
        """
    @n_nullity.setter
    def n_nullity(self, arg0: int) -> None:
        ...
    @property
    def n_nullspace(self) -> NDArray[numpy.float64]:
        """
        Dual null space basis (world frame).
        """
    @n_nullspace.setter
    def n_nullspace(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def value_differentiable(self) -> bool:
        """
        Whether the growth distance value is differentiable.
        """
    @value_differentiable.setter
    def value_differentiable(self, arg0: bool) -> None:
        ...
    @property
    def z_nullity(self) -> int:
        """
        Dimension of the primal null space.
        """
    @z_nullity.setter
    def z_nullity(self, arg0: int) -> None:
        ...
    @property
    def z_nullspace(self) -> NDArray[numpy.float64]:
        """
        Primal null space basis (world frame).
        """
    @z_nullspace.setter
    def z_nullspace(self, arg0: NDArray[numpy.float64]) -> None:
        ...
class DirectionalDerivative3:
    """

    Directional (Gateaux) derivatives of the growth distance for 3D sets.

    Attributes
    ----------
    z_nullspace : numpy.ndarray, shape (3, 2)
        Orthonormal basis for the primal solution set affine hull (world frame).
    n_nullspace : numpy.ndarray, shape (3, 3)
        Orthonormal basis for the dual solution set span (world frame, first
        column is the optimal normal vector).
    d_normal : numpy.ndarray, shape (3,)
        Derivative of the dual solution.
    d_z1, d_z2 : numpy.ndarray, shape (3,)
        Derivatives of the primal solutions.
    d_gd : float
        Derivative of the growth distance.
    z_nullity : int
        Dimension of the primal solution set null space.
    n_nullity : int
        Dimension of the dual solution set null space.
    value_differentiable : bool
        Whether the growth distance value is differentiable.
    differentiable : bool
        Whether the full solution is differentiable.
    """
    def __init__(self) -> None:
        ...
    @property
    def d_gd(self) -> float:
        """
        Derivative of the growth distance.
        """
    @d_gd.setter
    def d_gd(self, arg0: float) -> None:
        ...
    @property
    def d_normal(self) -> NDArray[numpy.float64]:
        """
        Derivative of the dual solution.
        """
    @d_normal.setter
    def d_normal(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def d_z1(self) -> NDArray[numpy.float64]:
        """
        Derivative of the primal solution for set 1.
        """
    @d_z1.setter
    def d_z1(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def d_z2(self) -> NDArray[numpy.float64]:
        """
        Derivative of the primal solution for set 2.
        """
    @d_z2.setter
    def d_z2(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def differentiable(self) -> bool:
        """
        Whether the full solution is differentiable.
        """
    @differentiable.setter
    def differentiable(self, arg0: bool) -> None:
        ...
    @property
    def n_nullity(self) -> int:
        """
        Dimension of the dual null space.
        """
    @n_nullity.setter
    def n_nullity(self, arg0: int) -> None:
        ...
    @property
    def n_nullspace(self) -> NDArray[numpy.float64]:
        """
        Dual null space basis (world frame).
        """
    @n_nullspace.setter
    def n_nullspace(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def value_differentiable(self) -> bool:
        """
        Whether the growth distance value is differentiable.
        """
    @value_differentiable.setter
    def value_differentiable(self, arg0: bool) -> None:
        ...
    @property
    def z_nullity(self) -> int:
        """
        Dimension of the primal null space.
        """
    @z_nullity.setter
    def z_nullity(self, arg0: int) -> None:
        ...
    @property
    def z_nullspace(self) -> NDArray[numpy.float64]:
        """
        Primal null space basis (world frame).
        """
    @z_nullspace.setter
    def z_nullspace(self, arg0: NDArray[numpy.float64]) -> None:
        ...
class Ellipse(ConvexSet2):
    """
    2D axis-aligned ellipse.
    """
    def __init__(self, hlx: float, hly: float, margin: float = 0.0) -> None:
        ...
class Ellipsoid(ConvexSet3):
    """
    3D axis-aligned ellipsoid.
    """
    def __init__(self, hlx: float, hly: float, hlz: float, margin: float = 0.0) -> None:
        ...
class Frustum(ConvexSet3):
    """
    3D frustum.
    """
    def __init__(self, base_radius: float, top_radius: float, height: float, margin: float = 0.0) -> None:
        ...
    def offset(self) -> float:
        """
        Returns the base z-offset.
        """
class Halfspace2:
    """
    2D half-space.
    """
    @staticmethod
    def dimension() -> int:
        """
        Returns dimension.
        """
    def __init__(self, margin: float = 0.0) -> None:
        ...
    def print_info(self) -> None:
        """
        Prints half-space info.
        """
    @property
    def margin(self) -> float:
        """
        Safety margin.
        """
    @margin.setter
    def margin(self, arg0: float) -> None:
        ...
class Halfspace3:
    """
    3D half-space.
    """
    @staticmethod
    def dimension() -> int:
        """
        Returns dimension.
        """
    def __init__(self, margin: float = 0.0) -> None:
        ...
    def print_info(self) -> None:
        """
        Prints half-space info.
        """
    @property
    def margin(self) -> float:
        """
        Safety margin.
        """
    @margin.setter
    def margin(self, arg0: float) -> None:
        ...
class KinematicState2:
    """

    Kinematic state of a 2D rigid body.

    Attributes
    ----------
    tf : numpy.ndarray, shape (3, 3)
        Rigid body transformation (rotation + translation) in homogeneous
        coordinates.  Initialized to identity.
    tw : numpy.ndarray, shape (3,)
        Rigid body twist ``(v_x, v_y, omega_z)``.  Initialized to zero.
    """
    tf: NDArray[numpy.float64]
    tw: NDArray[numpy.float64]
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, tf: NDArray[numpy.float64], tw: NDArray[numpy.float64]) -> None:
        ...
    def __repr__(self) -> str:
        ...
class KinematicState3:
    """

    Kinematic state of a 3D rigid body.

    Attributes
    ----------
    tf : numpy.ndarray, shape (4, 4)
        Rigid body transformation (rotation + translation) in homogeneous
        coordinates.  Initialized to identity.
    tw : numpy.ndarray, shape (6,)
        Rigid body twist ``(v_x, v_y, v_z, omega_x, omega_y, omega_z)``.
        Initialized to zero.
    """
    tf: NDArray[numpy.float64]
    tw: NDArray[numpy.float64]
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, tf: NDArray[numpy.float64], tw: NDArray[numpy.float64]) -> None:
        ...
    def __repr__(self) -> str:
        ...
class Mesh(ConvexSet3):
    """
    3D mesh.
    """
    def __init__(self, vertices: NDArray[numpy.float64], graph: NDArray[numpy.int32], inradius: float, margin: float = 0.0, thresh: float = 0.9, guess_level: int = 1, name: str = '__Mesh__') -> None:
        ...
    def graph(self) -> NDArray[numpy.int32]:
        """
        Returns mesh graph as an integer array.
        """
    def nvertices(self) -> int:
        """
        Returns number of vertices.
        """
    def vertices(self) -> NDArray[numpy.float64]:
        """
        Returns mesh hull vertices as an (n, 3) array.
        """
class MeshLoader:
    """

    Loads 3D meshes and computes convex-hull graph representations.
    """
    def __init__(self, maxhullvert: int = 10000) -> None:
        """
        Constructs a MeshLoader.
        """
    def compute_inradius(self, interior_point: typing.Any = None, use_given_ip: bool = False) -> tuple:
        """
        Computes inradius from internally stored points.

        Parameters
        ----------
        interior_point : numpy.ndarray, shape (3,), optional
        		Interior point used when `use_given_ip` is True.
        use_given_ip : bool, default False
        		If False, a new interior point is computed.

        Returns
        -------
        tuple[float, numpy.ndarray]
        		(inradius, interior_point_used)
        """
    def compute_inradius_from_halfspaces(self, normals: NDArray[numpy.float64], offsets: NDArray[numpy.float64], interior_point: NDArray[numpy.float64]) -> float:
        """
        Computes inradius from half-space representation and an interior point.
        """
    def load_obj(self, input: str, is_file: bool = True) -> None:
        """
        Loads from an OBJ filename or an OBJ string.
        """
    def make_facet_graph(self) -> tuple:
        """
        Builds the convex-hull facet graph and returns an interior point.

        Returns
        -------
        tuple[bool, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        		(valid, normals, offsets, graph, interior_point) where:
        		- normals has shape (nfacet, 3)
        		- offsets has shape (nfacet,)
        		- graph has shape (k,)
        		- interior_point has shape (3,)

        Graph layout
        ------------
        graph[0] = nfacet
        graph[1] = nridge
        graph[2 : 2 + nfacet] = facet_ridgeadr
        graph[2 + nfacet : ] = ridge_localid records terminated by -1
        """
    def make_vertex_graph(self) -> tuple:
        """
        Builds the convex-hull vertex graph.

        Returns
        -------
        tuple[bool, numpy.ndarray, numpy.ndarray]
        		(valid, vertices, graph) where:
        		- vertices has shape (nvert, 3)
        		- graph has shape (k,)

        Graph layout
        ------------
        graph[0] = nvert
        graph[1] = nface
        graph[2 : 2 + nvert] = vert_edgeadr
        graph[2 + nvert : ] = edge_localid records terminated by -1
        """
    def npts(self) -> int:
        """
        Returns number of stored points.
        """
    def process_points(self, points: NDArray[numpy.float64]) -> None:
        """
        Converts points to internal format and removes duplicates.
        """
class NormalConeSpan2:
    """
    Normal cone span.
    """
    def __init__(self) -> None:
        ...
    @property
    def span_dim(self) -> int:
        """
        Dimension of the normal cone span.
        """
    @span_dim.setter
    def span_dim(self, arg0: int) -> None:
        ...
class NormalConeSpan3:
    """
    Normal cone span.
    """
    def __init__(self) -> None:
        ...
    @property
    def basis(self) -> NDArray[numpy.float64]:
        """
        Basis for the normal cone span (excluding n).
        """
    @basis.setter
    def basis(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def span_dim(self) -> int:
        """
        Dimension of the normal cone span.
        """
    @span_dim.setter
    def span_dim(self, arg0: int) -> None:
        ...
class NormalPair2:
    """
    Base point-normal vector pair.
    """
    def __init__(self) -> None:
        ...
    @property
    def n(self) -> NDArray[numpy.float64]:
        """
        Normal vector.
        """
    @n.setter
    def n(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def z(self) -> NDArray[numpy.float64]:
        """
        Base point.
        """
    @z.setter
    def z(self, arg0: NDArray[numpy.float64]) -> None:
        ...
class NormalPair3:
    """
    Base point-normal vector pair.
    """
    def __init__(self) -> None:
        ...
    @property
    def n(self) -> NDArray[numpy.float64]:
        """
        Normal vector.
        """
    @n.setter
    def n(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def z(self) -> NDArray[numpy.float64]:
        """
        Base point.
        """
    @z.setter
    def z(self, arg0: NDArray[numpy.float64]) -> None:
        ...
class Output2:
    """

    Growth distance algorithm output for 2D convex sets.

    Attributes
    ----------
    s1, s2 : numpy.ndarray, shape (2, 2)
        Simplex vertices for each convex set (local frame), corresponding to the
        optimal inner polyhedral approximation.
    bc : numpy.ndarray, shape (2,)
        Barycentric coordinates of the optimal simplex.
    normal : numpy.ndarray, shape (2,)
        Unit normal vector of the optimal separating hyperplane (world frame),
        pointing from set 1 towards set 2.
    z1, z2 : numpy.ndarray, shape (2,)
        Primal optimal solutions in the world frame.
    idx_s1, idx_s2 : numpy.ndarray of int, shape (2,)
        Simplex vertex index hints.
    growth_dist_ub : float
        Upper bound on the growth distance (primal solution).
    growth_dist_lb : float
        Lower bound on the growth distance (dual solution).
    iter : int
        Number of solver iterations taken.
    status : SolutionStatus
        Solution status at termination.
    """
    def __init__(self) -> None:
        ...
    @property
    def bc(self) -> NDArray[numpy.float64]:
        """
        Barycentric coordinates.
        """
    @bc.setter
    def bc(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def growth_dist_lb(self) -> float:
        """
        Lower bound on the growth distance.
        """
    @growth_dist_lb.setter
    def growth_dist_lb(self, arg0: float) -> None:
        ...
    @property
    def growth_dist_ub(self) -> float:
        """
        Upper bound on the growth distance.
        """
    @growth_dist_ub.setter
    def growth_dist_ub(self, arg0: float) -> None:
        ...
    @property
    def idx_s1(self) -> NDArray[numpy.int32]:
        """
        Simplex vertex indices for set 1.
        """
    @idx_s1.setter
    def idx_s1(self, arg0: NDArray[numpy.int32]) -> None:
        ...
    @property
    def idx_s2(self) -> NDArray[numpy.int32]:
        """
        Simplex vertex indices for set 2.
        """
    @idx_s2.setter
    def idx_s2(self, arg0: NDArray[numpy.int32]) -> None:
        ...
    @property
    def iter(self) -> int:
        """
        Number of solver iterations taken.
        """
    @iter.setter
    def iter(self, arg0: int) -> None:
        ...
    @property
    def normal(self) -> NDArray[numpy.float64]:
        """
        Unit normal dual solution (world frame).
        """
    @normal.setter
    def normal(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def s1(self) -> NDArray[numpy.float64]:
        """
        Simplex vertices for set 1.
        """
    @s1.setter
    def s1(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def s2(self) -> NDArray[numpy.float64]:
        """
        Simplex vertices for set 2.
        """
    @s2.setter
    def s2(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def status(self) -> SolutionStatus:
        """
        Solution status.
        """
    @status.setter
    def status(self, arg0: SolutionStatus) -> None:
        ...
    @property
    def z1(self) -> NDArray[numpy.float64]:
        """
        Primal solution for set 1.
        """
    @z1.setter
    def z1(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def z2(self) -> NDArray[numpy.float64]:
        """
        Primal solution for set 2.
        """
    @z2.setter
    def z2(self, arg0: NDArray[numpy.float64]) -> None:
        ...
class Output3:
    """

    Growth distance algorithm output for 3D convex sets.

    Attributes
    ----------
    s1, s2 : numpy.ndarray, shape (3, 3)
        Simplex vertices for each convex set (local frame).
    bc : numpy.ndarray, shape (3,)
        Barycentric coordinates of the optimal simplex.
    normal : numpy.ndarray, shape (3,)
        Unit normal vector of the optimal separating hyperplane (world frame),
        pointing from set 1 towards set 2.
    z1, z2 : numpy.ndarray, shape (3,)
        Primal optimal solutions in the world frame.
    idx_s1, idx_s2 : numpy.ndarray of int, shape (3,)
        Simplex vertex index hints.
    growth_dist_ub : float
        Upper bound on the growth distance (primal solution).
    growth_dist_lb : float
        Lower bound on the growth distance (dual solution).
    iter : int
        Number of solver iterations taken.
    status : SolutionStatus
        Solution status at termination.
    """
    def __init__(self) -> None:
        ...
    @property
    def bc(self) -> NDArray[numpy.float64]:
        """
        Barycentric coordinates.
        """
    @bc.setter
    def bc(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def growth_dist_lb(self) -> float:
        """
        Lower bound on the growth distance.
        """
    @growth_dist_lb.setter
    def growth_dist_lb(self, arg0: float) -> None:
        ...
    @property
    def growth_dist_ub(self) -> float:
        """
        Upper bound on the growth distance.
        """
    @growth_dist_ub.setter
    def growth_dist_ub(self, arg0: float) -> None:
        ...
    @property
    def idx_s1(self) -> NDArray[numpy.int32]:
        """
        Simplex vertex indices for set 1.
        """
    @idx_s1.setter
    def idx_s1(self, arg0: NDArray[numpy.int32]) -> None:
        ...
    @property
    def idx_s2(self) -> NDArray[numpy.int32]:
        """
        Simplex vertex indices for set 2.
        """
    @idx_s2.setter
    def idx_s2(self, arg0: NDArray[numpy.int32]) -> None:
        ...
    @property
    def iter(self) -> int:
        """
        Number of solver iterations taken.
        """
    @iter.setter
    def iter(self, arg0: int) -> None:
        ...
    @property
    def normal(self) -> NDArray[numpy.float64]:
        """
        Unit normal dual solution (world frame).
        """
    @normal.setter
    def normal(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def s1(self) -> NDArray[numpy.float64]:
        """
        Simplex vertices for set 1.
        """
    @s1.setter
    def s1(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def s2(self) -> NDArray[numpy.float64]:
        """
        Simplex vertices for set 2.
        """
    @s2.setter
    def s2(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def status(self) -> SolutionStatus:
        """
        Solution status.
        """
    @status.setter
    def status(self, arg0: SolutionStatus) -> None:
        ...
    @property
    def z1(self) -> NDArray[numpy.float64]:
        """
        Primal solution for set 1.
        """
    @z1.setter
    def z1(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def z2(self) -> NDArray[numpy.float64]:
        """
        Primal solution for set 2.
        """
    @z2.setter
    def z2(self, arg0: NDArray[numpy.float64]) -> None:
        ...
class Polygon(ConvexSet2):
    """
    2D convex polygon.
    """
    def __init__(self, vertices: NDArray[numpy.float64], inradius: float, margin: float = 0.0) -> None:
        ...
    def nvertices(self) -> int:
        """
        Returns number of vertices.
        """
    def vertices(self) -> NDArray[numpy.float64]:
        """
        Returns polygon vertices as an (n, 2) array.
        """
class Polytope(ConvexSet3):
    """
    3D convex polytope.
    """
    def __init__(self, vertices: NDArray[numpy.float64], inradius: float, margin: float = 0.0, thresh: float = 0.75) -> None:
        ...
    def nvertices(self) -> int:
        """
        Returns number of vertices.
        """
    def vertices(self) -> NDArray[numpy.float64]:
        """
        Returns polytope vertices as an (n, 3) array.
        """
class Rectangle(ConvexSet2):
    """
    2D axis-aligned rectangle.
    """
    def __init__(self, hlx: float, hly: float, margin: float = 0.0) -> None:
        ...
class Settings:
    """

    Settings for the differentiable growth distance algorithm.

    All fields default to well-chosen values and can be overridden in-place.

    Attributes
    ----------
    min_center_dist : float
        Minimum distance between the center points of the convex sets.
        Growth distance is set to zero when the centers are closer than this.
    rel_tol : float
        Relative tolerance for the primal–dual gap
        (convergence criterion: ``lb <= ub <= rel_tol * lb``).
    nullspace_tol : float
        Tolerance for primal/dual null-space computations in 3D.
    jac_tol : float
        Tolerance for the solution derivative computation.
    max_iter : int
        Maximum number of solver iterations.
    ws_type : WarmStartType
        Warm start type.
    twist_frame : TwistFrame
        Reference frame for input twist vectors.
    """
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def jac_tol(self) -> float:
        """
        Tolerance for solution derivative computation.
        """
    @jac_tol.setter
    def jac_tol(self, arg0: float) -> None:
        ...
    @property
    def max_iter(self) -> int:
        """
        Maximum number of solver iterations.
        """
    @max_iter.setter
    def max_iter(self, arg0: int) -> None:
        ...
    @property
    def min_center_dist(self) -> float:
        """
        Minimum distance between center points of the sets.
        """
    @min_center_dist.setter
    def min_center_dist(self, arg0: float) -> None:
        ...
    @property
    def nullspace_tol(self) -> float:
        """
        Tolerance for null-space computations (3D).
        """
    @nullspace_tol.setter
    def nullspace_tol(self, arg0: float) -> None:
        ...
    @property
    def rel_tol(self) -> float:
        """
        Relative primal–dual gap tolerance.
        """
    @rel_tol.setter
    def rel_tol(self, arg0: float) -> None:
        ...
    @property
    def twist_frame(self) -> TwistFrame:
        """
        Reference frame for input twist vectors.
        """
    @twist_frame.setter
    def twist_frame(self, arg0: TwistFrame) -> None:
        ...
    @property
    def ws_type(self) -> WarmStartType:
        """
        Warm start type.
        """
    @ws_type.setter
    def ws_type(self, arg0: WarmStartType) -> None:
        ...
class SolutionError:
    """
    Growth-distance error metrics.
    """
    def __init__(self) -> None:
        ...
    @property
    def dual_infeas_err(self) -> float:
        """
        Dual infeasibility error.
        """
    @dual_infeas_err.setter
    def dual_infeas_err(self, arg0: float) -> None:
        ...
    @property
    def prim_dual_gap(self) -> float:
        """
        Relative primal-dual gap.
        """
    @prim_dual_gap.setter
    def prim_dual_gap(self, arg0: float) -> None:
        ...
    @property
    def prim_infeas_err(self) -> float:
        """
        Normalized primal infeasibility error.
        """
    @prim_infeas_err.setter
    def prim_infeas_err(self, arg0: float) -> None:
        ...
class SolutionStatus:
    """

    Solution status at the termination of the growth distance algorithm.

    Values
    ------
    Optimal              : converged within the relative tolerance.
    MaxIterReached       : maximum iteration count was reached.
    CoincidentCenters    : center points of the two sets are too close.
    IllConditionedInputs : input sets are ill-conditioned (inradii too small).


    Members:

      Optimal : Optimal solution reached.

      MaxIterReached : Maximum number of iterations reached.

      CoincidentCenters : Coincident center positions of the convex sets.

      IllConditionedInputs : Ill-conditioned input sets.
    """
    CoincidentCenters: typing.ClassVar[SolutionStatus]  # value = <SolutionStatus.CoincidentCenters: 2>
    IllConditionedInputs: typing.ClassVar[SolutionStatus]  # value = <SolutionStatus.IllConditionedInputs: 3>
    MaxIterReached: typing.ClassVar[SolutionStatus]  # value = <SolutionStatus.MaxIterReached: 1>
    Optimal: typing.ClassVar[SolutionStatus]  # value = <SolutionStatus.Optimal: 0>
    __members__: typing.ClassVar[dict[str, SolutionStatus]]  # value = {'Optimal': <SolutionStatus.Optimal: 0>, 'MaxIterReached': <SolutionStatus.MaxIterReached: 1>, 'CoincidentCenters': <SolutionStatus.CoincidentCenters: 2>, 'IllConditionedInputs': <SolutionStatus.IllConditionedInputs: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Sphere(ConvexSet3):
    """
    3D sphere.
    """
    def __init__(self, radius: float) -> None:
        ...
class Stadium(ConvexSet2):
    """
    2D stadium capsule.
    """
    def __init__(self, hlx: float, radius: float, margin: float = 0.0) -> None:
        ...
class SupportFunctionHint2:
    """
    Support function hint.
    """
    def __init__(self) -> None:
        ...
    @property
    def idx_ws(self) -> int:
        """
        Integer warm-start hint.
        """
    @idx_ws.setter
    def idx_ws(self, arg0: int) -> None:
        ...
    @property
    def n_prev(self) -> NDArray[numpy.float64]:
        """
        Normal vector at the previous iteration.
        """
    @n_prev.setter
    def n_prev(self, arg0: NDArray[numpy.float64]) -> None:
        ...
class SupportFunctionHint3:
    """
    Support function hint.
    """
    def __init__(self) -> None:
        ...
    @property
    def idx_ws(self) -> int:
        """
        Integer warm-start hint.
        """
    @idx_ws.setter
    def idx_ws(self, arg0: int) -> None:
        ...
    @property
    def n_prev(self) -> NDArray[numpy.float64]:
        """
        Normal vector at the previous iteration.
        """
    @n_prev.setter
    def n_prev(self, arg0: NDArray[numpy.float64]) -> None:
        ...
class SupportPatchHull2:
    """
    Support patch hull.
    """
    def __init__(self) -> None:
        ...
    @property
    def aff_dim(self) -> int:
        """
        Affine dimension of the support patch.
        """
    @aff_dim.setter
    def aff_dim(self, arg0: int) -> None:
        ...
class SupportPatchHull3:
    """
    Support patch hull.
    """
    def __init__(self) -> None:
        ...
    @property
    def aff_dim(self) -> int:
        """
        Affine dimension of the support patch.
        """
    @aff_dim.setter
    def aff_dim(self, arg0: int) -> None:
        ...
    @property
    def basis(self) -> NDArray[numpy.float64]:
        """
        Basis for the support patch affine hull.
        """
    @basis.setter
    def basis(self, arg0: NDArray[numpy.float64]) -> None:
        ...
class TotalDerivative2:
    """

    Total (Frechet) derivatives of the growth distance for 2-D sets.

    All Jacobians are with respect to the SE(2) rigid body motion (3-vectors).

    Attributes
    ----------
    d_normal_tf1, d_normal_tf2 : numpy.ndarray, shape (2, 3)
        Jacobians of the dual solution w.r.t. motion of sets 1 and 2.
    d_z1_tf1, d_z1_tf2 : numpy.ndarray, shape (2, 3)
        Jacobians of the primal solution z1 w.r.t. motion of sets 1 and 2.
    d_z2_tf1, d_z2_tf2 : numpy.ndarray, shape (2, 3)
        Jacobians of the primal solution z2 w.r.t. motion of sets 1 and 2.
    d_gd_tf1, d_gd_tf2 : numpy.ndarray, shape (3,)
        Gradients of the growth distance w.r.t. motion of sets 1 and 2.
    """
    def __init__(self) -> None:
        ...
    @property
    def d_gd_tf1(self) -> NDArray[numpy.float64]:
        """
        Gradient of growth distance w.r.t. motion of set 1.
        """
    @d_gd_tf1.setter
    def d_gd_tf1(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def d_gd_tf2(self) -> NDArray[numpy.float64]:
        """
        Gradient of growth distance w.r.t. motion of set 2.
        """
    @d_gd_tf2.setter
    def d_gd_tf2(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def d_normal_tf1(self) -> NDArray[numpy.float64]:
        """
        Jacobian of dual solution w.r.t. motion of set 1.
        """
    @d_normal_tf1.setter
    def d_normal_tf1(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def d_normal_tf2(self) -> NDArray[numpy.float64]:
        """
        Jacobian of dual solution w.r.t. motion of set 2.
        """
    @d_normal_tf2.setter
    def d_normal_tf2(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def d_z1_tf1(self) -> NDArray[numpy.float64]:
        """
        Jacobian of primal solution z1 w.r.t. motion of set 1.
        """
    @d_z1_tf1.setter
    def d_z1_tf1(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def d_z1_tf2(self) -> NDArray[numpy.float64]:
        """
        Jacobian of primal solution z1 w.r.t. motion of set 2.
        """
    @d_z1_tf2.setter
    def d_z1_tf2(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def d_z2_tf1(self) -> NDArray[numpy.float64]:
        """
        Jacobian of primal solution z2 w.r.t. motion of set 1.
        """
    @d_z2_tf1.setter
    def d_z2_tf1(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def d_z2_tf2(self) -> NDArray[numpy.float64]:
        """
        Jacobian of primal solution z2 w.r.t. motion of set 2.
        """
    @d_z2_tf2.setter
    def d_z2_tf2(self, arg0: NDArray[numpy.float64]) -> None:
        ...
class TotalDerivative3:
    """

    Total (Frechet) derivatives of the growth distance for 3-D sets.

    All Jacobians are with respect to the SE(3) rigid body motion (6-vectors).

    Attributes
    ----------
    d_normal_tf1, d_normal_tf2 : numpy.ndarray, shape (3, 6)
        Jacobians of the dual solution w.r.t. motion of sets 1 and 2.
    d_z1_tf1, d_z1_tf2 : numpy.ndarray, shape (3, 6)
        Jacobians of the primal solution z1 w.r.t. motion of sets 1 and 2.
    d_z2_tf1, d_z2_tf2 : numpy.ndarray, shape (3, 6)
        Jacobians of the primal solution z2 w.r.t. motion of sets 1 and 2.
    d_gd_tf1, d_gd_tf2 : numpy.ndarray, shape (6,)
        Gradients of the growth distance w.r.t. motion of sets 1 and 2.
    """
    def __init__(self) -> None:
        ...
    @property
    def d_gd_tf1(self) -> NDArray[numpy.float64]:
        """
        Gradient of growth distance w.r.t. motion of set 1.
        """
    @d_gd_tf1.setter
    def d_gd_tf1(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def d_gd_tf2(self) -> NDArray[numpy.float64]:
        """
        Gradient of growth distance w.r.t. motion of set 2.
        """
    @d_gd_tf2.setter
    def d_gd_tf2(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def d_normal_tf1(self) -> NDArray[numpy.float64]:
        """
        Jacobian of dual solution w.r.t. motion of set 1.
        """
    @d_normal_tf1.setter
    def d_normal_tf1(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def d_normal_tf2(self) -> NDArray[numpy.float64]:
        """
        Jacobian of dual solution w.r.t. motion of set 2.
        """
    @d_normal_tf2.setter
    def d_normal_tf2(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def d_z1_tf1(self) -> NDArray[numpy.float64]:
        """
        Jacobian of primal solution z1 w.r.t. motion of set 1.
        """
    @d_z1_tf1.setter
    def d_z1_tf1(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def d_z1_tf2(self) -> NDArray[numpy.float64]:
        """
        Jacobian of primal solution z1 w.r.t. motion of set 2.
        """
    @d_z1_tf2.setter
    def d_z1_tf2(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def d_z2_tf1(self) -> NDArray[numpy.float64]:
        """
        Jacobian of primal solution z2 w.r.t. motion of set 1.
        """
    @d_z2_tf1.setter
    def d_z2_tf1(self, arg0: NDArray[numpy.float64]) -> None:
        ...
    @property
    def d_z2_tf2(self) -> NDArray[numpy.float64]:
        """
        Jacobian of primal solution z2 w.r.t. motion of set 2.
        """
    @d_z2_tf2.setter
    def d_z2_tf2(self, arg0: NDArray[numpy.float64]) -> None:
        ...
class TwistFrame:
    """

    Rigid body twist reference frame.

    Values
    ------
    Spatial : spatial twist in the world frame.
    Hybrid  : hybrid twist in the world frame (translational velocity is the
              velocity of the origin of the local frame).
    Body    : body twist in the local frame.


    Members:

      Spatial : Spatial twist in the world frame.

      Hybrid : Hybrid twist in the world frame.

      Body : Body twist in the local frame.
    """
    Body: typing.ClassVar[TwistFrame]  # value = <TwistFrame.Body: 2>
    Hybrid: typing.ClassVar[TwistFrame]  # value = <TwistFrame.Hybrid: 1>
    Spatial: typing.ClassVar[TwistFrame]  # value = <TwistFrame.Spatial: 0>
    __members__: typing.ClassVar[dict[str, TwistFrame]]  # value = {'Spatial': <TwistFrame.Spatial: 0>, 'Hybrid': <TwistFrame.Hybrid: 1>, 'Body': <TwistFrame.Body: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class WarmStartType:
    """

    Warm start type for the growth distance algorithm.

    Values
    ------
    Primal : initialize from the previous primal solution.
    Dual   : initialize from the previous dual solution.


    Members:

      Primal : Primal solution warm start.

      Dual : Dual solution warm start.
    """
    Dual: typing.ClassVar[WarmStartType]  # value = <WarmStartType.Dual: 1>
    Primal: typing.ClassVar[WarmStartType]  # value = <WarmStartType.Primal: 0>
    __members__: typing.ClassVar[dict[str, WarmStartType]]  # value = {'Primal': <WarmStartType.Primal: 0>, 'Dual': <WarmStartType.Dual: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
@typing.overload
def assert_collision_status(set1: ConvexSet2, tf1: NDArray[numpy.float64], set2: ConvexSet2, tf2: NDArray[numpy.float64], out: Output2, collision: bool, max_prim_infeas_err: float = 1.4901161193847656e-08) -> bool:
    """
    Asserts collision status for ConvexSet2 x ConvexSet2.
    """
@typing.overload
def assert_collision_status(set1: ConvexSet3, tf1: NDArray[numpy.float64], set2: ConvexSet3, tf2: NDArray[numpy.float64], out: Output3, collision: bool, max_prim_infeas_err: float = 1.4901161193847656e-08) -> bool:
    """
    Asserts collision status for ConvexSet3 x ConvexSet3.
    """
@typing.overload
def compute_kkt_nullspace(set1: ConvexSet2, tf1: NDArray[numpy.float64], set2: ConvexSet2, tf2: NDArray[numpy.float64], settings: Settings, out: Output2, dd: DirectionalDerivative2) -> int:
    """
    Computes KKT null space for ConvexSet2 x ConvexSet2.
    """
@typing.overload
def compute_kkt_nullspace(set1: ConvexSet2, tf1: NDArray[numpy.float64], set2: Halfspace2, tf2: NDArray[numpy.float64], settings: Settings, out: Output2, dd: DirectionalDerivative2) -> int:
    """
    Computes KKT null space for ConvexSet2 x Halfspace2.
    """
@typing.overload
def compute_kkt_nullspace(set1: ConvexSet3, tf1: NDArray[numpy.float64], set2: ConvexSet3, tf2: NDArray[numpy.float64], settings: Settings, out: Output3, dd: DirectionalDerivative3) -> int:
    """
    Computes KKT null space for ConvexSet3 x ConvexSet3.
    """
@typing.overload
def compute_kkt_nullspace(set1: ConvexSet3, tf1: NDArray[numpy.float64], set2: Halfspace3, tf2: NDArray[numpy.float64], settings: Settings, out: Output3, dd: DirectionalDerivative3) -> int:
    """
    Computes KKT null space for ConvexSet3 x Halfspace3.
    """
def compute_polygon_inradius(vertices: NDArray[numpy.float64], interior_point: NDArray[numpy.float64]) -> float:
    """
    Computes polygon inradius at a given interior point.

    Parameters
    ----------
    vertices : numpy.ndarray, shape (n, 2)
    		Polygon vertices in CCW order.
    interior_point : numpy.ndarray, shape (2,)
    		A point in the polygon interior.

    Returns
    -------
    float
    		Inradius at the interior point (negative if the point is outside).
    """
@typing.overload
def compute_solution_error(set1: ConvexSet2, tf1: NDArray[numpy.float64], set2: ConvexSet2, tf2: NDArray[numpy.float64], out: Output2) -> SolutionError:
    """
    Computes primal/dual error metrics for ConvexSet2 x ConvexSet2.
    """
@typing.overload
def compute_solution_error(set1: ConvexSet3, tf1: NDArray[numpy.float64], set2: ConvexSet3, tf2: NDArray[numpy.float64], out: Output3) -> SolutionError:
    """
    Computes primal/dual error metrics for ConvexSet3 x ConvexSet3.
    """
@typing.overload
def detect_collision(set1: ConvexSet2, tf1: NDArray[numpy.float64], set2: ConvexSet2, tf2: NDArray[numpy.float64], settings: Settings, out: Output2, warm_start: bool = False) -> bool:
    """
    Collision detection for ConvexSet2 x ConvexSet2.
    """
@typing.overload
def detect_collision(set1: ConvexSet2, tf1: NDArray[numpy.float64], set2: Halfspace2, tf2: NDArray[numpy.float64], settings: Settings, out: Output2, warm_start: bool = False) -> bool:
    """
    Collision detection for ConvexSet2 x Halfspace2.
    """
@typing.overload
def detect_collision(set1: ConvexSet3, tf1: NDArray[numpy.float64], set2: ConvexSet3, tf2: NDArray[numpy.float64], settings: Settings, out: Output3, warm_start: bool = False) -> bool:
    """
    Collision detection for ConvexSet3 x ConvexSet3.
    """
@typing.overload
def detect_collision(set1: ConvexSet3, tf1: NDArray[numpy.float64], set2: Halfspace3, tf2: NDArray[numpy.float64], settings: Settings, out: Output3, warm_start: bool = False) -> bool:
    """
    Collision detection for ConvexSet3 x Halfspace3.
    """
@typing.overload
def factorize_kkt_system(set1: ConvexSet2, tf1: NDArray[numpy.float64], set2: ConvexSet2, tf2: NDArray[numpy.float64], settings: Settings, out: Output2, dd: DirectionalDerivative2) -> bool:
    """
    Factorizes KKT system for ConvexSet2 x ConvexSet2.
    """
@typing.overload
def factorize_kkt_system(set1: ConvexSet2, tf1: NDArray[numpy.float64], set2: Halfspace2, tf2: NDArray[numpy.float64], settings: Settings, out: Output2, dd: DirectionalDerivative2) -> bool:
    """
    Factorizes KKT system for ConvexSet2 x Halfspace2.
    """
@typing.overload
def factorize_kkt_system(set1: ConvexSet3, tf1: NDArray[numpy.float64], set2: ConvexSet3, tf2: NDArray[numpy.float64], settings: Settings, out: Output3, dd: DirectionalDerivative3) -> bool:
    """
    Factorizes KKT system for ConvexSet3 x ConvexSet3.
    """
@typing.overload
def factorize_kkt_system(set1: ConvexSet3, tf1: NDArray[numpy.float64], set2: Halfspace3, tf2: NDArray[numpy.float64], settings: Settings, out: Output3, dd: DirectionalDerivative3) -> bool:
    """
    Factorizes KKT system for ConvexSet3 x Halfspace3.
    """
@typing.overload
def gd_derivative(state1: KinematicState2, state2: KinematicState2, settings: Settings, out: Output2, dd: DirectionalDerivative2 = None) -> float:
    """
    Directional derivative of growth distance for 2D convex sets.
    """
@typing.overload
def gd_derivative(state1: KinematicState3, state2: KinematicState3, settings: Settings, out: Output3, dd: DirectionalDerivative3 = None) -> float:
    """
    Directional derivative of growth distance for 3D convex sets.
    """
@typing.overload
def gd_gradient(tf1: NDArray[numpy.float64], tf2: NDArray[numpy.float64], settings: Settings, out: Output2, td: TotalDerivative2) -> None:
    """
    Gradient of growth distance for 2D convex sets.
    """
@typing.overload
def gd_gradient(tf1: NDArray[numpy.float64], tf2: NDArray[numpy.float64], settings: Settings, out: Output3, td: TotalDerivative3) -> None:
    """
    Gradient of growth distance for 3D convex sets.
    """
@typing.overload
def gd_jacobian(tf1: NDArray[numpy.float64], tf2: NDArray[numpy.float64], settings: Settings, out: Output2, dd: DirectionalDerivative2, td: TotalDerivative2) -> None:
    """
    Jacobian of growth distance optimal solution for 2D convex sets.
    """
@typing.overload
def gd_jacobian(tf1: NDArray[numpy.float64], tf2: NDArray[numpy.float64], settings: Settings, out: Output3, dd: DirectionalDerivative3, td: TotalDerivative3) -> None:
    """
    Jacobian of growth distance optimal solution for 3D convex sets.
    """
@typing.overload
def gd_solution_derivative(state1: KinematicState2, state2: KinematicState2, settings: Settings, out: Output2, dd: DirectionalDerivative2) -> None:
    """
    Directional derivative of growth distance optimal solution for 2D convex sets.
    """
@typing.overload
def gd_solution_derivative(state1: KinematicState3, state2: KinematicState3, settings: Settings, out: Output3, dd: DirectionalDerivative3) -> None:
    """
    Directional derivative of growth distance optimal solution for 3D convex sets.
    """
def graham_scan(points: NDArray[numpy.float64]) -> NDArray[numpy.float64]:
    """
    Computes the 2D convex hull using Graham scan.

    Parameters
    ----------
    points : numpy.ndarray, shape (n, 2)
    		Input 2D points.

    Returns
    -------
    numpy.ndarray, shape (m, 2)
    		Convex hull vertices in CCW order with collinear duplicates removed.
    """
@typing.overload
def growth_distance(set1: ConvexSet2, tf1: NDArray[numpy.float64], set2: ConvexSet2, tf2: NDArray[numpy.float64], settings: Settings, out: Output2, warm_start: bool = False) -> float:
    """
    Growth distance for ConvexSet2 x ConvexSet2.
    """
@typing.overload
def growth_distance(set1: ConvexSet2, tf1: NDArray[numpy.float64], set2: Halfspace2, tf2: NDArray[numpy.float64], settings: Settings, out: Output2, warm_start: bool = False) -> float:
    """
    Growth distance for ConvexSet2 x Halfspace2.
    """
@typing.overload
def growth_distance(set1: ConvexSet3, tf1: NDArray[numpy.float64], set2: ConvexSet3, tf2: NDArray[numpy.float64], settings: Settings, out: Output3, warm_start: bool = False) -> float:
    """
    Growth distance for ConvexSet3 x ConvexSet3.
    """
@typing.overload
def growth_distance(set1: ConvexSet3, tf1: NDArray[numpy.float64], set2: Halfspace3, tf2: NDArray[numpy.float64], settings: Settings, out: Output3, warm_start: bool = False) -> float:
    """
    Growth distance for ConvexSet3 x Halfspace3.
    """
@typing.overload
def growth_distance_cp(set1: ConvexSet2, tf1: NDArray[numpy.float64], set2: ConvexSet2, tf2: NDArray[numpy.float64], settings: Settings, out: Output2, warm_start: bool = False) -> float:
    """
    Growth distance using cutting-plane method for ConvexSet2 x ConvexSet2.
    """
@typing.overload
def growth_distance_cp(set1: ConvexSet3, tf1: NDArray[numpy.float64], set2: ConvexSet3, tf2: NDArray[numpy.float64], settings: Settings, out: Output3, warm_start: bool = False) -> float:
    """
    Growth distance using cutting-plane method for ConvexSet3 x ConvexSet3.
    """
@typing.overload
def growth_distance_trn(set1: ConvexSet2, tf1: NDArray[numpy.float64], set2: ConvexSet2, tf2: NDArray[numpy.float64], settings: Settings, out: Output2, warm_start: bool = False) -> float:
    """
    Growth distance using trust-region Newton method for ConvexSet2 x ConvexSet2.
    """
@typing.overload
def growth_distance_trn(set1: ConvexSet3, tf1: NDArray[numpy.float64], set2: ConvexSet3, tf2: NDArray[numpy.float64], settings: Settings, out: Output3, warm_start: bool = False) -> float:
    """
    Growth distance using trust-region Newton method for ConvexSet3 x ConvexSet3.
    """
EPS: float = 2.220446049250313e-16
INF: float  # value = inf
PI: float = 3.141592653589793
SQRT_EPS: float = 1.4901161193847656e-08
