from unittest import TestCase
from benchmark.zellers_birthday import zeller

class Test_zellers_birthday(TestCase):
	def test_zeller_1(self):
		y = zeller(66,49,51)
		self.assertEqual(y,"Monday")

	def test_zeller_2(self):
		y = zeller(-41,19,-23)
		self.assertEqual(y,"Saturday")

	def test_zeller_3(self):
		y = zeller(-3,-52,68)
		self.assertEqual(y,"Friday")

	def test_zeller_4(self):
		y = zeller(100,34,-37)
		self.assertEqual(y,"Monday")

	def test_zeller_5(self):
		y = zeller(-76,68,-90)
		self.assertEqual(y,"Saturday")

	def test_zeller_6(self):
		y = zeller(18,-16,24)
		self.assertEqual(y,"Sunday")

	def test_zeller_7(self):
		y = zeller(-64,5,34)
		self.assertEqual(y,"Thursday")

	def test_zeller_8(self):
		y = zeller(63,56,12)
		self.assertEqual(y,"Sunday")

	def test_zeller_9(self):
		y = zeller(35,-93,98)
		self.assertEqual(y,"Monday")

	def test_zeller_10(self):
		y = zeller(-71,28,91)
		self.assertEqual(y,"Friday")

	def test_zeller_11(self):
		y = zeller(-29,-55,-85)
		self.assertEqual(y,"Thursday")

	def test_zeller_12(self):
		y = zeller(-75,-58,99)
		self.assertEqual(y,"Sunday")

