#!/usr/bin/env python3
"""Visualize a four-marker parking bay from a low-latency USB camera."""
import argparse,math,time
from pathlib import Path
import cv2,numpy as np
from robot_dashboard.aruco_approach import estimate_pose
from robot_dashboard.aruco_parking import parking_pose

def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",default="/dev/video0"); p.add_argument("--dictionary",default="DICT_6X6_50"); p.add_argument("--marker-size",type=float,default=.16); p.add_argument("--entry-offset",type=float,default=.70); p.add_argument("--horizontal-fov",type=float,default=70.42); p.add_argument("--calibration",type=Path); p.add_argument("--width",type=int,default=640); p.add_argument("--height",type=int,default=480); p.add_argument("--fps",type=int,default=30); p.add_argument("--seconds",type=float,default=300.); p.add_argument("--display",action="store_true"); a=p.parse_args()
    if not hasattr(cv2.aruco,a.dictionary): p.error("unknown dictionary")
    cap=cv2.VideoCapture(a.device,cv2.CAP_V4L2)
    if not cap.isOpened(): p.error("cannot open camera: "+a.device)
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*"MJPG")); cap.set(cv2.CAP_PROP_FRAME_WIDTH,a.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,a.height); cap.set(cv2.CAP_PROP_FPS,a.fps); cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco,a.dictionary)),cv2.aruco.DetectorParameters()); k=d=None; start=time.monotonic(); fps_start=start; count=0; shown_fps=0.; smooth=None
    while time.monotonic()-start<a.seconds:
        ok,frame=cap.read()
        if not ok: continue
        count+=1; now=time.monotonic(); h,w=frame.shape[:2]
        if now-fps_start>=.5: shown_fps=count/(now-start); fps_start=now
        if k is None:
            if a.calibration:
                data=np.load(a.calibration); k=np.asarray(data["camera_matrix"]); d=np.asarray(data["distortion"])
            else:
                fx=w/(2*math.tan(math.radians(a.horizontal_fov)/2)); k=np.array([[fx,0,w/2],[0,fx,h/2],[0,0,1]],float); d=np.zeros(5)
        corners,ids,_=detector.detectMarkers(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)); positions={}; pixels={}
        if ids is not None:
            for cs,mid in zip(corners,ids.reshape(-1)):
                marker_id=int(mid)
                if marker_id not in (0,1,2,3): continue
                pts=np.asarray(cs).reshape(4,2); pixels[marker_id]=pts.mean(axis=0)
                cv2.polylines(frame,[np.rint(pts).astype(np.int32).reshape(-1,1,2)],True,(255,0,0),3,cv2.LINE_AA)
                x,y=pts.min(axis=0); cv2.putText(frame,f"ID {marker_id}",(int(x),max(22,int(y)-6)),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,0,0),2)
                try:
                    _,t,_,error=estimate_pose(cs,a.marker_size,k,d)
                    if error<=3.: positions[marker_id]=(float(t[0]),float(t[2]))
                except (ValueError,cv2.error): pass
        status=f"MARKERS {len(positions)}/4  FPS {shown_fps:.1f}"; color=(0,0,255)
        if len(positions)==4:
            try:
                pose=parking_pose(positions,a.entry_offset); values=np.array([pose.center_x,pose.center_z,pose.yaw,pose.entry_x,pose.entry_z]); smooth=values if smooth is None else .25*values+.75*smooth
                order=[0,1,3,2]; polygon=np.rint([pixels[i] for i in order]).astype(np.int32).reshape(-1,1,2); cv2.polylines(frame,[polygon],True,(255,0,0),4,cv2.LINE_AA)
                center=np.mean([pixels[i] for i in range(4)],axis=0); near=(pixels[0]+pixels[1])/2; far=(pixels[2]+pixels[3])/2; direction=far-near; length=np.linalg.norm(direction)
                if length>1: direction/=length
                c=tuple(np.rint(center).astype(int)); cv2.drawMarker(frame,c,(0,255,255),cv2.MARKER_TILTED_CROSS,28,4); cv2.arrowedLine(frame,c,tuple(np.rint(center+direction*70).astype(int)),(0,255,255),4,tipLength=.25)
                status=f"PARKING FOUND  x={smooth[0]:+.2f}m z={smooth[1]:.2f}m yaw={math.degrees(smooth[2]):+.1f}deg"; color=(0,255,0)
            except ValueError: status="INVALID PARKING GEOMETRY"
        else: smooth=None
        cv2.rectangle(frame,(0,0),(w,46),(0,0,0),-1); cv2.putText(frame,status,(10,31),cv2.FONT_HERSHEY_SIMPLEX,.62,color,2)
        if a.display:
            cv2.imshow("ArUco parking bay",frame)
            if cv2.waitKey(1)&0xFF in (27,ord('q')): break
    cap.release()
    if a.display: cv2.destroyAllWindows()
if __name__=="__main__": main()
