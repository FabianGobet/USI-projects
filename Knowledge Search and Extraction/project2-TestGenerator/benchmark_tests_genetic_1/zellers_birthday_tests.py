from unittest import TestCase
from benchmark.zellers_birthday import zeller

class Test_zellers_birthday(TestCase):
	def test_zeller_1(self):
		y = zeller(8,25,-61)
		self.assertEqual(y,"Wednesday")

	def test_zeller_2(self):
		y = zeller(-59,92,-23)
		self.assertEqual(y,"Saturday")

	def test_zeller_3(self):
		y = zeller(-34,15,-5)
		self.assertEqual(y,"Monday")

	def test_zeller_4(self):
		y = zeller(-23,55,-27)
		self.assertEqual(y,"Tuesday")

	def test_zeller_5(self):
		y = zeller(40,-44,-95)
		self.assertEqual(y,"Sunday")

	def test_zeller_6(self):
		y = zeller(22,-44,-89)
		self.assertEqual(y,"Friday")

	def test_zeller_7(self):
		y = zeller(-98,58,99)
		self.assertEqual(y,"Saturday")

	def test_zeller_8(self):
		y = zeller(-16,-6,-15)
		self.assertEqual(y,"Tuesday")

	def test_zeller_9(self):
		y = zeller(33,-33,54)
		self.assertEqual(y,"Sunday")

	def test_zeller_10(self):
		y = zeller(-70,-29,97)
		self.assertEqual(y,"Monday")

	def test_zeller_11(self):
		y = zeller(-47,-98,-43)
		self.assertEqual(y,"Wednesday")

	def test_zeller_12(self):
		y = zeller(94,57,26)
		self.assertEqual(y,"Saturday")

