import unittest
from robot_dashboard.aruco_approach import Config,Controller,Observation,State
class TestController(unittest.TestCase):
 def setUp(self): self.c=Controller(Config(stable_frames=2))
 def o(self,**kw):
  v=dict(visible=True,marker_id=0,x=0.,z=1.,yaw=0.,error=.5); v.update(kw); return Observation(**v)
 def test_disarmed(self): self.assertEqual(self.c.update(self.o(),False).state,State.DISARMED)
 def test_stable_then_moves(self): self.assertEqual(self.c.update(self.o(),True).linear_x,0.); self.assertGreater(self.c.update(self.o(),True).linear_x,0.)
 def test_loss_stops(self): self.c.update(self.o(),True); self.c.update(self.o(),True); self.assertEqual(self.c.update(Observation(False),True).linear_x,0.)
 def test_wrong_id_stops(self): self.assertEqual(self.c.update(self.o(marker_id=2),True).linear_x,0.)
 def test_arrival_latches(self): self.c.update(self.o(z=.55),True); self.assertEqual(self.c.update(self.o(z=.55),True).state,State.ARRIVED); self.assertEqual(self.c.update(self.o(z=1.),True).linear_x,0.)
if __name__=="__main__": unittest.main()
