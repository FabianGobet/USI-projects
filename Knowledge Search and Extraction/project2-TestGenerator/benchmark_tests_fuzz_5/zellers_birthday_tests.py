from unittest import TestCase
from benchmark.zellers_birthday import zeller

class Test_zellers_birthday(TestCase):
	def test_zeller_1(self):
		y = zeller(72,27,28)
		self.assertEqual(y,"Wednesday")

	def test_zeller_2(self):
		y = zeller(-98,38,-99)
		self.assertEqual(y,"Saturday")

	def test_zeller_3(self):
		y = zeller(-66,-2,36)
		self.assertEqual(y,"Wednesday")

	def test_zeller_4(self):
		y = zeller(87,72,-33)
		self.assertEqual(y,"Thursday")

	def test_zeller_5(self):
		y = zeller(-12,16,-25)
		self.assertEqual(y,"Tuesday")

	def test_zeller_6(self):
		y = zeller(89,74,90)
		self.assertEqual(y,"Wednesday")

	def test_zeller_7(self):
		y = zeller(-22,-12,44)
		self.assertEqual(y,"Friday")

	def test_zeller_8(self):
		y = zeller(-50,93,-23)
		self.assertEqual(y,"Saturday")

	def test_zeller_9(self):
		y = zeller(-82,53,12)
		self.assertEqual(y,"Thursday")

	def test_zeller_10(self):
		y = zeller(83,3,92)
		self.assertEqual(y,"Sunday")

