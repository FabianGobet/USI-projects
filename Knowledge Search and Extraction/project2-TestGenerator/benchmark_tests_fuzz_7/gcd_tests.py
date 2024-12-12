from unittest import TestCase
from benchmark.gcd import gcd

class Test_gcd(TestCase):
	def test_gcd_1(self):
		y = gcd(26,1)
		self.assertEqual(y,1)

	def test_gcd_2(self):
		y = gcd(18,16)
		self.assertEqual(y,2)

	def test_gcd_3(self):
		y = gcd(65,65)
		self.assertEqual(y,65)

	def test_gcd_4(self):
		y = gcd(38,2)
		self.assertEqual(y,2)

	def test_gcd_5(self):
		y = gcd(1,36)
		self.assertEqual(y,1)

	def test_gcd_6(self):
		y = gcd(2,5)
		self.assertEqual(y,1)

	def test_gcd_7(self):
		y = gcd(69,48)
		self.assertEqual(y,3)

	def test_gcd_8(self):
		y = gcd(81,64)
		self.assertEqual(y,1)

	def test_gcd_9(self):
		y = gcd(7,56)
		self.assertEqual(y,7)

