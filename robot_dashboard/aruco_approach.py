"""Fail-closed ArUco marker approach controller."""
from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum

class State(str, Enum):
    DISARMED="disarmed"; SEARCHING="searching"; ALIGNING="aligning"; APPROACHING="approaching"; ARRIVED="arrived"

@dataclass(frozen=True)
class Observation:
    visible: bool; marker_id: int|None=None; x: float=0.; y: float=0.; z: float=0.; yaw: float=0.; error: float=math.inf

@dataclass(frozen=True)
class Command:
    state: State; linear_x: float; angular_z: float; reason: str

@dataclass
class Config:
    target_id:int=0; target_distance:float=.55; distance_tolerance:float=.05
    lateral_tolerance:float=.035; yaw_tolerance:float=math.radians(8); stable_frames:int=3
    max_error:float=3.; min_distance:float=.15; max_distance:float=4.
    linear_gain:float=.45; lateral_gain:float=1.6; yaw_gain:float=.35
    max_linear:float=.18; max_angular:float=.45; drive_angle:float=math.radians(14)

class Controller:
    def __init__(self, config=None): self.config=config or Config(); self.reset()
    def reset(self): self.streak=0; self.arrived=False
    def update(self, o, armed):
        c=self.config
        if not armed: self.reset(); return Command(State.DISARMED,0.,0.,"motion not armed")
        valid=(o.visible and o.marker_id==c.target_id and all(math.isfinite(v) for v in (o.x,o.z,o.yaw,o.error)) and c.min_distance<=o.z<=c.max_distance and o.error<=c.max_error)
        if not valid: self.streak=0; return Command(State.SEARCHING,0.,0.,"target missing or pose rejected")
        self.streak+=1
        if self.streak<c.stable_frames: return Command(State.SEARCHING,0.,0.,"waiting for stable target")
        dz=o.z-c.target_distance
        if abs(dz)<=c.distance_tolerance and abs(o.x)<=c.lateral_tolerance and abs(o.yaw)<=c.yaw_tolerance: self.arrived=True
        if self.arrived: return Command(State.ARRIVED,0.,0.,"target pose reached")
        wz=max(-c.max_angular,min(c.max_angular,-(c.lateral_gain*o.x+c.yaw_gain*o.yaw)))
        vx=min(c.max_linear,max(0.,c.linear_gain*dz)) if abs(math.atan2(o.x,o.z))<=c.drive_angle else 0.
        return Command(State.APPROACHING if vx else State.ALIGNING,vx,wz,"tracking target")

def estimate_pose(corners, size, camera_matrix, distortion):
    import cv2, numpy as np
    h=float(size)/2
    if not math.isfinite(h) or h<=0: raise ValueError("marker size must be positive")
    obj=np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]],np.float64)
    img=np.asarray(corners,np.float64).reshape(4,2); k=np.asarray(camera_matrix,np.float64).reshape(3,3); d=np.asarray(distortion,np.float64)
    ok,r,t=cv2.solvePnP(obj,img,k,d,flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok: raise ValueError("solvePnP failed")
    projected,_=cv2.projectPoints(obj,r,t,k,d)
    error=float(np.linalg.norm(projected.reshape(4,2)-img,axis=1).mean())
    rotation,_=cv2.Rodrigues(r); normal=rotation[:,2]
    return r.reshape(3),t.reshape(3),float(math.atan2(normal[0],normal[2])),error
