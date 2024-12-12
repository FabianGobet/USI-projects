from unittest import TestCase
from benchmark.zellers_birthday import zeller

class Test_zellers_birthday(TestCase):
	def test_zeller_1(self):
		y = zeller(-4,82,-35)
		self.assertEqual(y,"Monday")

	def test_zeller_2(self):
		y = zeller(-78,-75,-18)
		self.assertEqual(y,"Tuesday")

	def test_zeller_3(self):
		y = zeller(-29,62,25)
		self.assertEqual(y,"Sunday")

	def test_zeller_4(self):
		y = zeller(34,85,75)
		self.assertEqual(y,"Tuesday")

	def test_zeller_5(self):
		y = zeller(-41,45,-23)
		self.assertEqual(y,"Thursday")

	def test_zeller_6(self):
		y = zeller(-39,-54,99)
		self.assertEqual(y,"Friday")

	def test_zeller_7(self):
		y = zeller(88,-91,-83)
		self.assertEqual(y,"Saturday")

	def test_zeller_8(self):
		y = zeller(95,11,-45)
		self.assertEqual(y,"Saturday")

	def test_zeller_9(self):
		y = zeller(-19,-99,-28)
		self.assertEqual(y,"Thursday")

