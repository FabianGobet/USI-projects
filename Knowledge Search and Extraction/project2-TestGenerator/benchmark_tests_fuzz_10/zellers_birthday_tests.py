from unittest import TestCase
from benchmark.zellers_birthday import zeller

class Test_zellers_birthday(TestCase):
	def test_zeller_1(self):
		y = zeller(78,-69,96)
		self.assertEqual(y,"Thursday")

	def test_zeller_2(self):
		y = zeller(47,-27,-97)
		self.assertEqual(y,"Thursday")

	def test_zeller_3(self):
		y = zeller(-45,-68,26)
		self.assertEqual(y,"Wednesday")

	def test_zeller_4(self):
		y = zeller(3,-27,-98)
		self.assertEqual(y,"Friday")

	def test_zeller_5(self):
		y = zeller(65,-25,-73)
		self.assertEqual(y,"Sunday")

	def test_zeller_6(self):
		y = zeller(-8,-60,18)
		self.assertEqual(y,"Monday")

	def test_zeller_7(self):
		y = zeller(-8,5,53)
		self.assertEqual(y,"Friday")

	def test_zeller_8(self):
		y = zeller(19,-69,99)
		self.assertEqual(y,"Tuesday")

	def test_zeller_9(self):
		y = zeller(-26,5,23)
		self.assertEqual(y,"Saturday")

	def test_zeller_10(self):
		y = zeller(60,-38,24)
		self.assertEqual(y,"Sunday")

	def test_zeller_11(self):
		y = zeller(68,69,-61)
		self.assertEqual(y,"Saturday")

