#!/usr/bin/env python3
"""Identify the OpenCV ArUco dictionary and IDs in an image or USB camera."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import cv2

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
    found=defaultdict(lambda:{"ids":set(),"detections":0,"variants":set()})
    for value,names in dictionaries().items():
        detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(value),cv2.aruco.DetectorParameters())
        for variant,image in variants(frame):
            _,ids,_=detector.detectMarkers(image)
            if ids is not None:
                row=found[value]; row["ids"].update(map(int,ids.reshape(-1))); row["detections"]+=len(ids); row["variants"].add(variant)
    return found

def report(found):
    if not found: print("NO_ARUCO_DETECTED"); return
    for value,row in sorted(found.items(),key=lambda item:(-len(item[1]["ids"]),item[0])):
        print(f"{','.join(dictionaries()[value])}: ids={sorted(row['ids'])} unique={len(row['ids'])} hits={row['detections']} variants={sorted(row['variants'])}")

def main():
    p=argparse.ArgumentParser(); group=p.add_mutually_exclusive_group(required=True); group.add_argument("--image",type=Path); group.add_argument("--device"); p.add_argument("--seconds",type=float,default=15); a=p.parse_args()
    if a.image:
        frame=cv2.imread(str(a.image));
        if frame is None: p.error(f"cannot read image: {a.image}")
        print(f"source={a.image} shape={frame.shape}"); report(scan(frame)); return
    cap=cv2.VideoCapture(a.device,cv2.CAP_V4L2)
    if not cap.isOpened(): p.error(f"cannot open camera: {a.device}")
    import time
    aggregate=defaultdict(lambda:{"ids":set(),"detections":0,"variants":set()}); frames=0; start=time.monotonic()
    while time.monotonic()-start<a.seconds:
        ok,frame=cap.read()
        if not ok: continue
        frames+=1
        for value,row in scan(frame).items():
            aggregate[value]["ids"].update(row["ids"]); aggregate[value]["detections"]+=row["detections"]; aggregate[value]["variants"].update(row["variants"])
    cap.release(); print(f"source={a.device} frames={frames}"); report(aggregate)
if __name__=="__main__": main()
