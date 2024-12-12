from unittest import TestCase
from benchmark.zellers_birthday import zeller

class Test_zellers_birthday(TestCase):
	def test_zeller_1(self):
		y = zeller(60,-36,-13)
		self.assertEqual(y,"Wednesday")

	def test_zeller_2(self):
		y = zeller(-21,-77,37)
		self.assertEqual(y,"Monday")

	def test_zeller_3(self):
		y = zeller(-54,-66,25)
		self.assertEqual(y,"Friday")

	def test_zeller_4(self):
		y = zeller(51,-27,47)
		self.assertEqual(y,"Monday")

	def test_zeller_5(self):
		y = zeller(-65,97,23)
		self.assertEqual(y,"Sunday")

	def test_zeller_6(self):
		y = zeller(57,88,24)
		self.assertEqual(y,"Tuesday")

	def test_zeller_7(self):
		y = zeller(53,46,99)
		self.assertEqual(y,"Tuesday")

	def test_zeller_8(self):
		y = zeller(62,10,32)
		self.assertEqual(y,"Saturday")

	def test_zeller_9(self):
		y = zeller(-4,14,93)
		self.assertEqual(y,"Thursday")

	def test_zeller_10(self):
		y = zeller(-26,50,13)
		self.assertEqual(y,"Tuesday")

