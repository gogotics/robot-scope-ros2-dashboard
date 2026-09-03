"""Geometry for a four-marker ArUco parking bay."""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ParkingPose:
    center_x: float
    center_z: float
    yaw: float
    entry_x: float
    entry_z: float
    width: float
    depth: float

def diagonal_intersection(p0,p3,p1,p2):
    """Return the pixel intersection of diagonals 0-3 and 1-2."""
    a=np.asarray(p0,dtype=float).reshape(2); b=np.asarray(p3,dtype=float).reshape(2)
    c=np.asarray(p1,dtype=float).reshape(2); d=np.asarray(p2,dtype=float).reshape(2)
    direction0=b-a; direction1=d-c
    cross=direction0[0]*direction1[1]-direction0[1]*direction1[0]
    if not math.isfinite(cross) or abs(cross)<1e-6: raise ValueError("parking diagonals are parallel")
    delta=c-a; t=(delta[0]*direction1[1]-delta[1]*direction1[0])/cross
    if not 0.0<=t<=1.0: raise ValueError("parking diagonals do not cross inside the bay")
    point=a+t*direction0
    if not np.isfinite(point).all(): raise ValueError("parking center is not finite")
    return float(point[0]),float(point[1])

def parking_pose(points, entry_offset=.70):
    """Compute bay pose from IDs 0=near-left, 1=near-right, 2=far-left, 3=far-right."""
    if set(points) != {0,1,2,3}: raise ValueError("parking pose requires marker IDs 0, 1, 2, and 3")
    p={i:np.asarray(points[i],dtype=float).reshape(2) for i in range(4)}
    if not all(np.isfinite(v).all() for v in p.values()): raise ValueError("marker positions must be finite")
    center=sum(p.values())/4.0
    lateral=((p[1]-p[0])+(p[3]-p[2]))/2.0
    forward=((p[2]-p[0])+(p[3]-p[1]))/2.0
    width=float(np.linalg.norm(lateral)); depth=float(np.linalg.norm(forward))
    if width<.05 or depth<.05: raise ValueError("parking bay geometry is degenerate")
    unit=forward/depth
    # Entry lies before the near edge, toward the camera/robot.
    near=(p[0]+p[1])/2.0
    entry=near-unit*float(entry_offset)
    return ParkingPose(float(center[0]),float(center[1]),float(math.atan2(unit[0],unit[1])),float(entry[0]),float(entry[1]),width,depth)
