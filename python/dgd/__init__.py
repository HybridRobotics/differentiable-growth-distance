"""Python bindings for the Differentiable Growth Distance library.

This package exposes pybind11 bindings compiled into ``dgd._dgd_core``.
"""

from ._dgd_core import (  # noqa: F401
    # Constants
    EPS,
    INF,
    PI,
    SQRT_EPS,
    BasePointHint2,
    BasePointHint3,
    # Geometry base classes
    ConvexSet2,
    ConvexSet3,
    DirectionalDerivative2,
    DirectionalDerivative3,
    # Types
    KinematicState2,
    KinematicState3,
    MeshLoader,
    NormalConeSpan2,
    NormalConeSpan3,
    NormalPair2,
    NormalPair3,
    Output2,
    Output3,
    # Settings
    Settings,
    # Output
    SolutionStatus,
    # Geometry base helpers
    SupportFunctionHint2,
    SupportFunctionHint3,
    SupportPatchHull2,
    SupportPatchHull3,
    TotalDerivative2,
    TotalDerivative3,
    TwistFrame,
    WarmStartType,
    compute_polygon_inradius,
    # Utilities
    graham_scan,
)

__all__ = [
    # Constants
    "EPS",
    "INF",
    "PI",
    "SQRT_EPS",
    # Types
    "KinematicState2",
    "KinematicState3",
    # Geometry base helpers
    "SupportFunctionHint2",
    "SupportFunctionHint3",
    "NormalPair2",
    "NormalPair3",
    "SupportPatchHull2",
    "SupportPatchHull3",
    "NormalConeSpan2",
    "NormalConeSpan3",
    "BasePointHint2",
    "BasePointHint3",
    # Geometry base classes
    "ConvexSet2",
    "ConvexSet3",
    # Settings
    "Settings",
    "TwistFrame",
    "WarmStartType",
    # Output
    "SolutionStatus",
    "Output2",
    "Output3",
    "DirectionalDerivative2",
    "DirectionalDerivative3",
    "TotalDerivative2",
    "TotalDerivative3",
    # Utilities
    "graham_scan",
    "compute_polygon_inradius",
    "MeshLoader",
]
