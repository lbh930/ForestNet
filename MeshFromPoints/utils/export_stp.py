"""export_stp.py
Build solid trees (cylinder + ellipsoid) and export one STEP file.
Needs pythonocc‑core.
"""
from pathlib import Path
from typing import List, Tuple
import numpy as np

from OCC.Core.gp import gp_Pnt, gp_Trsf
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Extend.DataExchange import write_step_file

TreeParam = Tuple[np.ndarray, float, float, np.ndarray]  # (base, r, h, (rx,ry,rz))


def _make_tree(base: np.ndarray, r: float, h: float, er: np.ndarray):
    """Return OCC solid for one tree."""
    # trunk
    cyl = BRepPrimAPI_MakeCylinder(r, h).Solid()
    t1 = gp_Trsf(); t1.SetTranslation(gp_Pnt(0, 0, 0), gp_Pnt(*base))
    cyl = BRepBuilderAPI_Transform(cyl, t1).Shape()

    # crown
    sph = BRepPrimAPI_MakeSphere(1.0).Shape()
    s = gp_Trsf(); s.SetScaleFactor(1)
    s.SetValues(er[0], 0, 0, 0,
                0, er[1], 0, 0,
                0, 0, er[2], 0)
    sph = BRepBuilderAPI_Transform(sph, s).Shape()
    t2 = gp_Trsf(); t2.SetTranslation(gp_Pnt(0, 0, 0), gp_Pnt(base[0], base[1], base[2] + h))
    sph = BRepBuilderAPI_Transform(sph, t2).Shape()

    return BRepAlgoAPI_Fuse(cyl, sph).Shape()


def export_trees_to_step(trees: List[TreeParam], step_path: str):
    """trees: list of params → STEP solid forest."""
    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)

    for p in trees:
        solid = _make_tree(*p)
        builder.Add(comp, solid)

    Path(step_path).parent.mkdir(parents=True, exist_ok=True)
    status = write_step_file(comp, str(step_path))