from unittest import TestCase
from benchmark.gcd import gcd

class Test_gcd(TestCase):
	def test_gcd_1(self):
		y = gcd(2,91)
		self.assertEqual(y,1)

	def test_gcd_2(self):
		y = gcd(76,1)
		self.assertEqual(y,1)

	def test_gcd_3(self):
		y = gcd(3,7)
		self.assertEqual(y,1)

	def test_gcd_4(self):
		y = gcd(41,58)
		self.assertEqual(y,1)

	def test_gcd_5(self):
		y = gcd(97,68)
		self.assertEqual(y,1)

	def test_gcd_6(self):
		y = gcd(85,87)
		self.assertEqual(y,1)

	def test_gcd_7(self):
		y = gcd(15,36)
		self.assertEqual(y,3)

	def test_gcd_8(self):
		y = gcd(24,35)
		self.assertEqual(y,1)

	def test_gcd_9(self):
		y = gcd(9,25)
		self.assertEqual(y,1)

	def test_gcd_10(self):
		y = gcd(13,13)
		self.assertEqual(y,13)

	def test_gcd_11(self):
		y = gcd(30,56)
		self.assertEqual(y,2)

