from unittest import TestCase
from benchmark.gcd import gcd

class Test_gcd(TestCase):
	def test_gcd_1(self):
		y = gcd(10,82)
		self.assertEqual(y,2)

	def test_gcd_2(self):
		y = gcd(8,43)
		self.assertEqual(y,1)

	def test_gcd_3(self):
		y = gcd(9,9)
		self.assertEqual(y,9)

	def test_gcd_4(self):
		y = gcd(60,1)
		self.assertEqual(y,1)

	def test_gcd_5(self):
		y = gcd(71,70)
		self.assertEqual(y,1)

	def test_gcd_6(self):
		y = gcd(3,10)
		self.assertEqual(y,1)

	def test_gcd_7(self):
		y = gcd(49,47)
		self.assertEqual(y,1)

	def test_gcd_8(self):
		y = gcd(100,85)
		self.assertEqual(y,5)

	def test_gcd_9(self):
		y = gcd(1,51)
		self.assertEqual(y,1)

	def test_gcd_10(self):
		y = gcd(15,57)
		self.assertEqual(y,3)

	def test_gcd_11(self):
		y = gcd(67,69)
		self.assertEqual(y,1)

