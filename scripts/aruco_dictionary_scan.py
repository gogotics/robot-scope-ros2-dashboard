#!/usr/bin/env python3
"""Identify the OpenCV ArUco dictionary and IDs in an image or USB camera."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

def dictionaries():
    seen={}
    for name in sorted(n for n in dir(cv2.aruco) if n.startswith("DICT_")):
        value=getattr(cv2.aruco,name)
        if isinstance(value,int): seen.setdefault(value,[]).append(name)
    return seen

def variants(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if frame.ndim==3 else frame
    clahe=cv2.createCLAHE(2.0,(8,8)).apply(gray)
    yield "gray",gray
    yield "clahe",clahe
    yield "gray_2x",cv2.resize(gray,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    yield "clahe_2x",cv2.resize(clahe,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)

def scan(frame):
    found=defaultdict(lambda:{"ids":set(),"detections":0,"variants":set(),"boxes":[]})
    for value,names in dictionaries().items():
        detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(value),cv2.aruco.DetectorParameters())
        for variant,image in variants(frame):
            corners,ids,_=detector.detectMarkers(image)
            if ids is not None:
                row=found[value]; row["ids"].update(map(int,ids.reshape(-1))); row["detections"]+=len(ids); row["variants"].add(variant)
                scale=2.0 if variant.endswith("_2x") else 1.0
                if not row["boxes"]:
                    row["boxes"]=[(np.asarray(c).reshape(4,2)/scale,int(marker_id)) for c,marker_id in zip(corners,ids.reshape(-1))]
    return found

def report(found):
    if not found: print("NO_ARUCO_DETECTED"); return
    for value,row in sorted(found.items(),key=lambda item:(-len(item[1]["ids"]),item[0])):
        print(f"{','.join(dictionaries()[value])}: ids={sorted(row['ids'])} unique={len(row['ids'])} hits={row['detections']} variants={sorted(row['variants'])}")

def main():
    p=argparse.ArgumentParser(); group=p.add_mutually_exclusive_group(required=True); group.add_argument("--image",type=Path); group.add_argument("--device"); p.add_argument("--seconds",type=float,default=15); p.add_argument("--display",action="store_true",help="show a live preview; q or Esc exits"); a=p.parse_args()
    if a.image:
        frame=cv2.imread(str(a.image));
        if frame is None: p.error(f"cannot read image: {a.image}")
        print(f"source={a.image} shape={frame.shape}"); report(scan(frame)); return
    cap=cv2.VideoCapture(a.device,cv2.CAP_V4L2)
    if not cap.isOpened(): p.error(f"cannot open camera: {a.device}")
    import time
    aggregate=defaultdict(lambda:{"ids":set(),"detections":0,"variants":set(),"boxes":[]}); frames=0; start=time.monotonic()
    while time.monotonic()-start<a.seconds:
        ok,frame=cap.read()
        if not ok: continue
        frames+=1
        current=scan(frame)
        for value,row in current.items():
            aggregate[value]["ids"].update(row["ids"]); aggregate[value]["detections"]+=row["detections"]; aggregate[value]["variants"].update(row["variants"])
        if a.display:
            preview=frame.copy(); lines=[]
            if current:
                value,row=max(current.items(),key=lambda item:(len(item[1]["boxes"]),len(item[1]["ids"])))
                dictionary_name=dictionaries()[value][0]
                for corners,marker_id in row["boxes"]:
                    points=np.rint(corners).astype(np.int32).reshape((-1,1,2))
                    cv2.polylines(preview,[points],True,(255,0,0),4,cv2.LINE_AA)
                    x,y=points.reshape(4,2).min(axis=0)
                    cv2.putText(preview,f"{dictionary_name} ID {marker_id}",(int(x),max(24,int(y)-8)),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,0,0),2)
                lines=[f"DETECTED: {dictionary_name} IDs {sorted(row['ids'])}"]
            else:
                lines=["NO ARUCO DETECTED"]
            color=(255,0,0) if current else (0,0,255)
            cv2.rectangle(preview,(0,0),(preview.shape[1],48),(0,0,0),-1)
            for index,text in enumerate(lines): cv2.putText(preview,text,(12,32+28*index),cv2.FONT_HERSHEY_SIMPLEX,.72,color,2)
            cv2.imshow("ArUco dictionary scanner",preview)
            if cv2.waitKey(1)&0xFF in (27,ord("q")): break
    cap.release()
    if a.display: cv2.destroyAllWindows()
    print(f"source={a.device} frames={frames}"); report(aggregate)
if __name__=="__main__": main()
