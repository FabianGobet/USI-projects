from unittest import TestCase
from benchmark.zellers_birthday import zeller

class Test_zellers_birthday(TestCase):
	def test_zeller_1(self):
		y = zeller(59,46,-45)
		self.assertEqual(y,"Thursday")

	def test_zeller_2(self):
		y = zeller(63,88,97)
		self.assertEqual(y,"Friday")

	def test_zeller_3(self):
		y = zeller(64,37,-25)
		self.assertEqual(y,"Tuesday")

	def test_zeller_4(self):
		y = zeller(-47,-92,-99)
		self.assertEqual(y,"Friday")

	def test_zeller_5(self):
		y = zeller(-6,9,-78)
		self.assertEqual(y,"Wednesday")

	def test_zeller_6(self):
		y = zeller(-44,100,-75)
		self.assertEqual(y,"Wednesday")

	def test_zeller_7(self):
		y = zeller(-47,41,87)
		self.assertEqual(y,"Wednesday")

	def test_zeller_8(self):
		y = zeller(-67,-86,-23)
		self.assertEqual(y,"Tuesday")

	def test_zeller_9(self):
		y = zeller(1,-85,55)
		self.assertEqual(y,"Tuesday")

	def test_zeller_10(self):
		y = zeller(75,51,-80)
		self.assertEqual(y,"Monday")

	def test_zeller_11(self):
		y = zeller(79,71,-2)
		self.assertEqual(y,"Wednesday")

	def test_zeller_12(self):
		y = zeller(49,69,35)
		self.assertEqual(y,"Saturday")

	def test_zeller_13(self):
		y = zeller(-55,15,52)
		self.assertEqual(y,"Friday")

	def test_zeller_14(self):
		y = zeller(-46,-3,-64)
		self.assertEqual(y,"Monday")

