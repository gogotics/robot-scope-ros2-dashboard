import math,unittest
from robot_dashboard.aruco_parking import parking_pose

class ParkingPoseTests(unittest.TestCase):
    def test_axis_aligned_bay(self):
        pose=parking_pose({0:(-.4,1.5),1:(.4,1.5),2:(-.4,2.1),3:(.4,2.1)},.5)
        self.assertAlmostEqual(pose.center_x,0.); self.assertAlmostEqual(pose.center_z,1.8); self.assertAlmostEqual(pose.yaw,0.); self.assertAlmostEqual(pose.entry_z,1.)
    def test_rotated_bay(self):
        angle=math.radians(12); forward=(math.sin(angle)*.6,math.cos(angle)*.6)
        points={0:(-.4,1.5),1:(.4,1.5),2:(-.4+forward[0],1.5+forward[1]),3:(.4+forward[0],1.5+forward[1])}
        self.assertAlmostEqual(parking_pose(points).yaw,angle)
    def test_requires_all_ids(self):
        with self.assertRaises(ValueError): parking_pose({0:(0,0)})

if __name__=="__main__": unittest.main()
