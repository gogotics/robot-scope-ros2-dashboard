#!/usr/bin/env python3
"""Run ArUco pose/approach observation directly from a USB camera."""
import argparse, json, math, time
from pathlib import Path
import cv2
import numpy as np
from robot_dashboard.aruco_approach import Config, Controller, Observation, estimate_pose

def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",default="/dev/video0"); p.add_argument("--target-id",type=int,default=0); p.add_argument("--marker-size",type=float,default=.16); p.add_argument("--target-distance",type=float,default=.55); p.add_argument("--horizontal-fov",type=float,default=70.42); p.add_argument("--calibration",type=Path); p.add_argument("--seconds",type=float,default=30.); p.add_argument("--output",type=Path,default=Path("/tmp/aruco_logitech.jpg")); a=p.parse_args()
    cap=cv2.VideoCapture(a.device,cv2.CAP_V4L2)
    if not cap.isOpened(): raise SystemExit("camera open failed: "+a.device)
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),cv2.aruco.DetectorParameters()); ctl=Controller(Config(target_id=a.target_id,target_distance=a.target_distance)); start=time.monotonic(); frames=0; hits=0; last=None; k=d=None
    while time.monotonic()-start<a.seconds:
        ok,frame=cap.read()
        if not ok: continue
        frames+=1; h,w=frame.shape[:2]
        if k is None:
            if a.calibration:
                data=np.load(a.calibration); k=np.asarray(data["camera_matrix"]); d=np.asarray(data["distortion"])
            else:
                fx=w/(2*math.tan(math.radians(a.horizontal_fov)/2)); k=np.array([[fx,0,w/2],[0,fx,h/2],[0,0,1]],float); d=np.zeros(5)
        corners,ids,_=detector.detectMarkers(frame); obs=Observation(False)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame,corners,ids)
            for cs,mid in zip(corners,ids.reshape(-1)):
                if int(mid)==a.target_id:
                    try:
                        r,t,yaw,error=estimate_pose(cs,a.marker_size,k,d); obs=Observation(True,int(mid),float(t[0]),float(t[1]),float(t[2]),yaw,error); hits+=1; cv2.drawFrameAxes(frame,k,d,r,t,a.marker_size*.5)
                    except (ValueError,cv2.error): pass
                    break
        command=ctl.update(obs,False)
        label=f"ID {a.target_id}: {'FOUND' if obs.visible else 'SEARCHING'}"
        if obs.visible: label+=f" x={obs.x:+.2f}m z={obs.z:.2f}m yaw={math.degrees(obs.yaw):+.1f}deg"
        cv2.putText(frame,label,(12,32),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,0) if obs.visible else (0,180,255),2); last=frame
    cap.release()
    if last is not None: a.output.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(a.output),last)
    print(json.dumps({"camera":a.device,"frames":frames,"target_id":a.target_id,"detections":hits,"output":str(a.output),"calibration":"file" if a.calibration else "approximate-c920-fov"}))
if __name__=="__main__": main()
