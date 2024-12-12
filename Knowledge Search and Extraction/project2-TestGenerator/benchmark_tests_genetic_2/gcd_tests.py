from unittest import TestCase
from benchmark.gcd import gcd

class Test_gcd(TestCase):
	def test_gcd_1(self):
		y = gcd(1,73)
		self.assertEqual(y,1)

	def test_gcd_2(self):
		y = gcd(90,1)
		self.assertEqual(y,1)

	def test_gcd_3(self):
		y = gcd(3,75)
		self.assertEqual(y,3)

	def test_gcd_4(self):
		y = gcd(59,55)
		self.assertEqual(y,1)

	def test_gcd_5(self):
		y = gcd(2,46)
		self.assertEqual(y,2)

	def test_gcd_6(self):
		y = gcd(70,29)
		self.assertEqual(y,1)

	def test_gcd_7(self):
		y = gcd(94,80)
		self.assertEqual(y,2)

	def test_gcd_8(self):
		y = gcd(5,68)
		self.assertEqual(y,1)

	def test_gcd_9(self):
		y = gcd(9,60)
		self.assertEqual(y,3)

	def test_gcd_10(self):
		y = gcd(92,18)
		self.assertEqual(y,2)

	def test_gcd_11(self):
		y = gcd(4,98)
		self.assertEqual(y,2)

	def test_gcd_12(self):
		y = gcd(30,82)
		self.assertEqual(y,2)

	def test_gcd_13(self):
		y = gcd(58,8)
		self.assertEqual(y,2)

	def test_gcd_14(self):
		y = gcd(33,33)
		self.assertEqual(y,33)

