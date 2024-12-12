from unittest import TestCase
from benchmark.gcd import gcd

class Test_gcd(TestCase):
	def test_gcd_1(self):
		y = gcd(66,1)
		self.assertEqual(y,1)

	def test_gcd_2(self):
		y = gcd(74,75)
		self.assertEqual(y,1)

	def test_gcd_3(self):
		y = gcd(24,87)
		self.assertEqual(y,3)

	def test_gcd_4(self):
		y = gcd(2,55)
		self.assertEqual(y,1)

	def test_gcd_5(self):
		y = gcd(45,3)
		self.assertEqual(y,3)

	def test_gcd_6(self):
		y = gcd(45,9)
		self.assertEqual(y,9)

	def test_gcd_7(self):
		y = gcd(1,52)
		self.assertEqual(y,1)

	def test_gcd_8(self):
		y = gcd(10,73)
		self.assertEqual(y,1)

	def test_gcd_9(self):
		y = gcd(40,40)
		self.assertEqual(y,40)

	def test_gcd_10(self):
		y = gcd(87,39)
		self.assertEqual(y,3)

