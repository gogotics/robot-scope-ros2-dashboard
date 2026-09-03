#!/usr/bin/env python3
"""Generate and verify exact OpenCV parking markers (DICT_6X6_50 IDs 0-3)."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("aruco_markers"))
    parser.add_argument("--marker-pixels",type=int,default=500)
    parser.add_argument("--margin-pixels",type=int,default=100)
    args=parser.parse_args()
    if args.marker_pixels<80 or args.margin_pixels<10: parser.error("marker must be >=80px and margin >=10px")
    args.output.mkdir(parents=True,exist_ok=True)
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    detector=cv2.aruco.ArucoDetector(dictionary,cv2.aruco.DetectorParameters())
    for marker_id in range(4):
        marker=cv2.aruco.generateImageMarker(dictionary,marker_id,args.marker_pixels)
        image=cv2.copyMakeBorder(marker,args.margin_pixels,args.margin_pixels,args.margin_pixels,args.margin_pixels,cv2.BORDER_CONSTANT,value=255)
        path=args.output/f"aruco_id_{marker_id}.png"
        if not cv2.imwrite(str(path),image): raise SystemExit(f"failed to write {path}")
        check=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        _,ids,_=detector.detectMarkers(check)
        detected=[] if ids is None else list(map(int,ids.reshape(-1)))
        if detected!=[marker_id]: raise SystemExit(f"verification failed for {path}: {detected}")
        print(f"verified {path}: DICT_6X6_50 ID {marker_id} shape={image.shape[1]}x{image.shape[0]}")
    # A simple reference image for placement; print the four individual files.
    files=[cv2.imread(str(args.output/f"aruco_id_{i}.png"),cv2.IMREAD_GRAYSCALE) for i in range(4)]
    gap=np.full((files[0].shape[0],args.margin_pixels),255,np.uint8)
    row0=np.hstack((files[0],gap,files[1])); row1=np.hstack((files[2],gap,files[3]))
    vertical=np.full((args.margin_pixels,row0.shape[1]),255,np.uint8)
    cv2.imwrite(str(args.output/"parking_layout_reference.png"),np.vstack((row0,vertical,row1)))
    print("layout: ID 0=near-left, ID 1=near-right, ID 2=far-left, ID 3=far-right")
if __name__=="__main__": main()
