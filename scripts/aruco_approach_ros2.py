#!/usr/bin/env python3
"""D435i ArUco detection and safe Go2 approach via the private command gate."""
import argparse,json,math,time
import cv2,numpy as np
from robot_dashboard.aruco_approach import Config,Controller,Observation,estimate_pose

def main():
    p=argparse.ArgumentParser(); p.add_argument("--image-topic",default="/camera/camera/color/image_raw"); p.add_argument("--camera-info-topic",default="/camera/camera/color/camera_info"); p.add_argument("--cmd-topic",default="/robot_scope/nav/cmd_vel_raw"); p.add_argument("--target-id",type=int,default=0); p.add_argument("--marker-size",type=float,default=.16); p.add_argument("--target-distance",type=float,default=.55); p.add_argument("--arm",action="store_true"); p.add_argument("--timeout",type=float,default=.35); a=p.parse_args()
    if a.marker_size<=0 or a.timeout<=0: p.error("marker-size and timeout must be positive")
    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped,Twist
    from sensor_msgs.msg import Image,CameraInfo
    from std_msgs.msg import String
    from rclpy.qos import qos_profile_sensor_data
    rclpy.init(); node=rclpy.create_node("aruco_go2_approach"); bridge=CvBridge(); ctl=Controller(Config(target_id=a.target_id,target_distance=a.target_distance)); detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),cv2.aruco.DetectorParameters()); state={"k":None,"d":None,"last":0.}
    cmdpub=node.create_publisher(Twist,a.cmd_topic,10); posepub=node.create_publisher(PoseStamped,"/aruco/target_pose",10); statpub=node.create_publisher(String,"/aruco/approach_status",10)
    def stop(): cmdpub.publish(Twist())
    def info(m): state["k"]=np.asarray(m.k).reshape(3,3); state["d"]=np.asarray(m.d)
    def image(m):
        state["last"]=time.monotonic(); obs=Observation(False)
        if state["k"] is not None:
            corners,ids,_=detector.detectMarkers(bridge.imgmsg_to_cv2(m,"bgr8"))
            if ids is not None:
                for cs,mid in zip(corners,ids.reshape(-1)):
                    if int(mid)==a.target_id:
                        try:
                            r,t,yaw,error=estimate_pose(cs,a.marker_size,state["k"],state["d"]); obs=Observation(True,int(mid),float(t[0]),float(t[1]),float(t[2]),yaw,error)
                            pose=PoseStamped(); pose.header=m.header; pose.pose.position.x,pose.pose.position.y,pose.pose.position.z=map(float,t)
                            angle=float(np.linalg.norm(r)); axis=r/angle if angle>1e-9 else np.zeros(3); scale=math.sin(angle/2)
                            pose.pose.orientation.x=float(axis[0]*scale); pose.pose.orientation.y=float(axis[1]*scale); pose.pose.orientation.z=float(axis[2]*scale); pose.pose.orientation.w=float(math.cos(angle/2)); posepub.publish(pose)
                        except (ValueError,cv2.error): pass
                        break
        out=ctl.update(obs,a.arm); twist=Twist(); twist.linear.x=out.linear_x; twist.angular.z=out.angular_z; cmdpub.publish(twist); status=String(); status.data=json.dumps({"state":out.state.value,"armed":a.arm,"visible":obs.visible,"id":obs.marker_id,"x":obs.x,"z":obs.z,"yaw":obs.yaw,"error":obs.error if np.isfinite(obs.error) else None}); statpub.publish(status)
    node.create_subscription(CameraInfo,a.camera_info_topic,info,qos_profile_sensor_data); node.create_subscription(Image,a.image_topic,image,qos_profile_sensor_data); node.create_timer(.05,lambda: stop() if time.monotonic()-state["last"]>a.timeout else None)
    try: rclpy.spin(node)
    finally: stop(); stop(); stop(); node.destroy_node(); rclpy.shutdown()
if __name__=="__main__": main()
