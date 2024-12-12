from unittest import TestCase
from benchmark.zellers_birthday import zeller

class Test_zellers_birthday(TestCase):
	def test_zeller_1(self):
		y = zeller(59,86,-99)
		self.assertEqual(y,"Monday")

	def test_zeller_2(self):
		y = zeller(-62,-42,69)
		self.assertEqual(y,"Tuesday")

	def test_zeller_3(self):
		y = zeller(23,-58,-85)
		self.assertEqual(y,"Saturday")

	def test_zeller_4(self):
		y = zeller(-3,-34,69)
		self.assertEqual(y,"Monday")

	def test_zeller_5(self):
		y = zeller(23,63,-27)
		self.assertEqual(y,"Saturday")

	def test_zeller_6(self):
		y = zeller(94,24,24)
		self.assertEqual(y,"Wednesday")

	def test_zeller_7(self):
		y = zeller(-42,-27,-53)
		self.assertEqual(y,"Sunday")

	def test_zeller_8(self):
		y = zeller(-14,-22,-23)
		self.assertEqual(y,"Wednesday")

	def test_zeller_9(self):
		y = zeller(9,17,-96)
		self.assertEqual(y,"Sunday")

	def test_zeller_10(self):
		y = zeller(-21,-54,-79)
		self.assertEqual(y,"Saturday")

	def test_zeller_11(self):
		y = zeller(-24,-87,-37)
		self.assertEqual(y,"Saturday")

	def test_zeller_12(self):
		y = zeller(94,-84,76)
		self.assertEqual(y,"Friday")

	def test_zeller_13(self):
		y = zeller(87,26,-91)
		self.assertEqual(y,"Tuesday")

	def test_zeller_14(self):
		y = zeller(-13,79,-25)
		self.assertEqual(y,"Thursday")

	def test_zeller_15(self):
		y = zeller(18,52,95)
		self.assertEqual(y,"Thursday")

	def test_zeller_16(self):
		y = zeller(-73,11,-10)
		self.assertEqual(y,"Friday")

	def test_zeller_17(self):
		y = zeller(74,-70,26)
		self.assertEqual(y,"Saturday")

	def test_zeller_18(self):
		y = zeller(94,-84,20)
		self.assertEqual(y,"Thursday")

	def test_zeller_19(self):
		y = zeller(53,-43,-98)
		self.assertEqual(y,"Sunday")

	def test_zeller_20(self):
		y = zeller(-43,-3,-29)
		self.assertEqual(y,"Wednesday")

