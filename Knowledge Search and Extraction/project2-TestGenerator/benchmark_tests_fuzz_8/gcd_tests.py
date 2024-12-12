from unittest import TestCase
from benchmark.gcd import gcd

class Test_gcd(TestCase):
	def test_gcd_1(self):
		y = gcd(2,1)
		self.assertEqual(y,1)

	def test_gcd_2(self):
		y = gcd(20,88)
		self.assertEqual(y,4)

	def test_gcd_3(self):
		y = gcd(4,25)
		self.assertEqual(y,1)

	def test_gcd_4(self):
		y = gcd(64,45)
		self.assertEqual(y,1)

	def test_gcd_5(self):
		y = gcd(36,3)
		self.assertEqual(y,3)

	def test_gcd_6(self):
		y = gcd(66,76)
		self.assertEqual(y,2)

	def test_gcd_7(self):
		y = gcd(47,48)
		self.assertEqual(y,1)

	def test_gcd_8(self):
		y = gcd(98,96)
		self.assertEqual(y,2)

	def test_gcd_9(self):
		y = gcd(12,89)
		self.assertEqual(y,1)

	def test_gcd_10(self):
		y = gcd(47,47)
		self.assertEqual(y,47)

	def test_gcd_11(self):
		y = gcd(2,85)
		self.assertEqual(y,1)

	def test_gcd_12(self):
		y = gcd(1,22)
		self.assertEqual(y,1)

