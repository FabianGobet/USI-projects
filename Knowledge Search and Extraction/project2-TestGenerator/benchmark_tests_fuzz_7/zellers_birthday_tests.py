from unittest import TestCase
from benchmark.zellers_birthday import zeller

class Test_zellers_birthday(TestCase):
	def test_zeller_1(self):
		y = zeller(-23,-24,-65)
		self.assertEqual(y,"Saturday")

	def test_zeller_2(self):
		y = zeller(-75,69,98)
		self.assertEqual(y,"Wednesday")

	def test_zeller_3(self):
		y = zeller(7,39,-93)
		self.assertEqual(y,"Wednesday")

	def test_zeller_4(self):
		y = zeller(-57,-80,-94)
		self.assertEqual(y,"Tuesday")

	def test_zeller_5(self):
		y = zeller(70,54,0)
		self.assertEqual(y,"Sunday")

	def test_zeller_6(self):
		y = zeller(-19,5,99)
		self.assertEqual(y,"Wednesday")

	def test_zeller_7(self):
		y = zeller(36,-78,-23)
		self.assertEqual(y,"Friday")

	def test_zeller_8(self):
		y = zeller(21,10,-47)
		self.assertEqual(y,"Tuesday")

	def test_zeller_9(self):
		y = zeller(96,27,24)
		self.assertEqual(y,"Friday")

