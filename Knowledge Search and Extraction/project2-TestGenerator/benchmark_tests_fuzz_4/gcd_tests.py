from unittest import TestCase
from benchmark.gcd import gcd

class Test_gcd(TestCase):
	def test_gcd_1(self):
		y = gcd(8,69)
		self.assertEqual(y,1)

	def test_gcd_2(self):
		y = gcd(19,8)
		self.assertEqual(y,1)

	def test_gcd_3(self):
		y = gcd(56,4)
		self.assertEqual(y,4)

	def test_gcd_4(self):
		y = gcd(78,77)
		self.assertEqual(y,1)

	def test_gcd_5(self):
		y = gcd(20,1)
		self.assertEqual(y,1)

	def test_gcd_6(self):
		y = gcd(15,26)
		self.assertEqual(y,1)

	def test_gcd_7(self):
		y = gcd(86,67)
		self.assertEqual(y,1)

	def test_gcd_8(self):
		y = gcd(1,36)
		self.assertEqual(y,1)

	def test_gcd_9(self):
		y = gcd(5,29)
		self.assertEqual(y,1)

	def test_gcd_10(self):
		y = gcd(79,12)
		self.assertEqual(y,1)

	def test_gcd_11(self):
		y = gcd(78,15)
		self.assertEqual(y,3)

	def test_gcd_12(self):
		y = gcd(56,56)
		self.assertEqual(y,56)

	def test_gcd_13(self):
		y = gcd(35,95)
		self.assertEqual(y,5)

	def test_gcd_14(self):
		y = gcd(45,53)
		self.assertEqual(y,1)

