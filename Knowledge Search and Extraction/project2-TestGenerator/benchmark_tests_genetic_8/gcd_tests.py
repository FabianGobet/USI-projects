from unittest import TestCase
from benchmark.gcd import gcd

class Test_gcd(TestCase):
	def test_gcd_1(self):
		y = gcd(84,30)
		self.assertEqual(y,6)

	def test_gcd_2(self):
		y = gcd(3,98)
		self.assertEqual(y,1)

	def test_gcd_3(self):
		y = gcd(2,78)
		self.assertEqual(y,2)

	def test_gcd_4(self):
		y = gcd(41,43)
		self.assertEqual(y,1)

	def test_gcd_5(self):
		y = gcd(72,72)
		self.assertEqual(y,72)

	def test_gcd_6(self):
		y = gcd(39,1)
		self.assertEqual(y,1)

	def test_gcd_7(self):
		y = gcd(1,20)
		self.assertEqual(y,1)

	def test_gcd_8(self):
		y = gcd(95,99)
		self.assertEqual(y,1)

	def test_gcd_9(self):
		y = gcd(16,85)
		self.assertEqual(y,1)

	def test_gcd_10(self):
		y = gcd(42,87)
		self.assertEqual(y,3)

