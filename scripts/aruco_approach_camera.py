#!/usr/bin/env python3
"""Run ArUco pose/approach observation directly from a USB camera."""
import argparse, json, math, time
from pathlib import Path
import cv2
import numpy as np
from robot_dashboard.aruco_approach import Config, Controller, Observation, estimate_pose

def main():
    names={n:getattr(cv2.aruco,n) for n in dir(cv2.aruco) if n.startswith("DICT_") and isinstance(getattr(cv2.aruco,n),int)}
    p=argparse.ArgumentParser(); p.add_argument("--device",default="/dev/video0"); p.add_argument("--dictionary",choices=sorted(names),default="DICT_4X4_50"); p.add_argument("--target-id",type=int,default=-1,help="-1 detects any ID"); p.add_argument("--marker-size",type=float,default=.16); p.add_argument("--target-distance",type=float,default=.55); p.add_argument("--horizontal-fov",type=float,default=70.42); p.add_argument("--calibration",type=Path); p.add_argument("--width",type=int,default=640); p.add_argument("--height",type=int,default=480); p.add_argument("--fps",type=int,default=30); p.add_argument("--seconds",type=float,default=300.); p.add_argument("--display",action="store_true"); p.add_argument("--output",type=Path,default=Path("/tmp/aruco_logitech.jpg")); a=p.parse_args()
    cap=cv2.VideoCapture(a.device,cv2.CAP_V4L2)
    if not cap.isOpened(): raise SystemExit("camera open failed: "+a.device)
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*"MJPG")); cap.set(cv2.CAP_PROP_FRAME_WIDTH,a.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,a.height); cap.set(cv2.CAP_PROP_FPS,a.fps); cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(names[a.dictionary]),cv2.aruco.DetectorParameters()); ctl=Controller(Config(target_id=a.target_id,target_distance=a.target_distance)); start=time.monotonic(); fps_start=start; fps_value=0.; frames=0; fps_frames=0; hits=0; last=None; k=d=None
    while time.monotonic()-start<a.seconds:
        ok,frame=cap.read()
        if not ok: continue
        frames+=1; fps_frames+=1; now=time.monotonic(); h,w=frame.shape[:2]
        if now-fps_start>=.5: fps_value=fps_frames/(now-fps_start); fps_frames=0; fps_start=now
        if k is None:
            if a.calibration:
                data=np.load(a.calibration); k=np.asarray(data["camera_matrix"]); d=np.asarray(data["distortion"])
            else:
                fx=w/(2*math.tan(math.radians(a.horizontal_fov)/2)); k=np.array([[fx,0,w/2],[0,fx,h/2],[0,0,1]],float); d=np.zeros(5)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); corners,ids,_=detector.detectMarkers(gray); obs=Observation(False)
        if ids is not None:
            for cs,mid in zip(corners,ids.reshape(-1)):
                points=np.rint(np.asarray(cs).reshape(4,2)).astype(np.int32).reshape(-1,1,2); cv2.polylines(frame,[points],True,(255,0,0),3,cv2.LINE_AA); x,y=points.reshape(4,2).min(axis=0); cv2.putText(frame,f"ID {int(mid)}",(int(x),max(22,int(y)-7)),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,0,0),2)
                if a.target_id<0 or int(mid)==a.target_id:
                    try:
                        r,t,yaw,error=estimate_pose(cs,a.marker_size,k,d); obs=Observation(True,int(mid),float(t[0]),float(t[1]),float(t[2]),yaw,error); hits+=1; cv2.drawFrameAxes(frame,k,d,r,t,a.marker_size*.5)
                    except (ValueError,cv2.error): pass
                    break
        command=ctl.update(obs,False)
        label=f"{a.dictionary} {'DETECTED' if obs.visible else 'NO ARUCO'}  FPS {fps_value:.1f}"
        if obs.visible: label+=f" x={obs.x:+.2f}m z={obs.z:.2f}m yaw={math.degrees(obs.yaw):+.1f}deg"
        cv2.rectangle(frame,(0,0),(w,45),(0,0,0),-1); cv2.putText(frame,label,(12,31),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,0,0) if obs.visible else (0,0,255),2); last=frame
        if a.display:
            cv2.imshow("Low-latency ArUco",frame)
            if cv2.waitKey(1)&0xFF in (27,ord("q")): break
    cap.release()
    if a.display: cv2.destroyAllWindows()
    if last is not None: a.output.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(a.output),last)
    print(json.dumps({"camera":a.device,"frames":frames,"fps":fps_value,"dictionary":a.dictionary,"target_id":a.target_id,"detections":hits,"output":str(a.output),"calibration":"file" if a.calibration else "approximate-c920-fov"}))
if __name__=="__main__": main()
