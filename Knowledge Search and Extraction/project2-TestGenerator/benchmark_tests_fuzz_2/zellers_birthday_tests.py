from unittest import TestCase
from benchmark.zellers_birthday import zeller

class Test_zellers_birthday(TestCase):
	def test_zeller_1(self):
		y = zeller(37,-16,-3)
		self.assertEqual(y,"Wednesday")

	def test_zeller_2(self):
		y = zeller(44,-80,78)
		self.assertEqual(y,"Thursday")

	def test_zeller_3(self):
		y = zeller(-20,-92,22)
		self.assertEqual(y,"Tuesday")

	def test_zeller_4(self):
		y = zeller(-11,-24,-49)
		self.assertEqual(y,"Tuesday")

	def test_zeller_5(self):
		y = zeller(46,-70,34)
		self.assertEqual(y,"Friday")

	def test_zeller_6(self):
		y = zeller(-52,-38,25)
		self.assertEqual(y,"Sunday")

	def test_zeller_7(self):
		y = zeller(68,65,-95)
		self.assertEqual(y,"Wednesday")

	def test_zeller_8(self):
		y = zeller(-18,33,-26)
		self.assertEqual(y,"Monday")

	def test_zeller_9(self):
		y = zeller(95,75,35)
		self.assertEqual(y,"Wednesday")

	def test_zeller_10(self):
		y = zeller(-15,-65,97)
		self.assertEqual(y,"Sunday")

	def test_zeller_11(self):
		y = zeller(-22,-57,-43)
		self.assertEqual(y,"Friday")

	def test_zeller_12(self):
		y = zeller(53,15,28)
		self.assertEqual(y,"Monday")

	def test_zeller_13(self):
		y = zeller(-77,56,-99)
		self.assertEqual(y,"Thursday")

	def test_zeller_14(self):
		y = zeller(-85,56,23)
		self.assertEqual(y,"Monday")

	def test_zeller_15(self):
		y = zeller(78,86,-16)
		self.assertEqual(y,"Thursday")

	def test_zeller_16(self):
		y = zeller(-1,-2,96)
		self.assertEqual(y,"Thursday")

	def test_zeller_17(self):
		y = zeller(-2,7,-40)
		self.assertEqual(y,"Tuesday")

