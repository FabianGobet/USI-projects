from unittest import TestCase
from benchmark.zellers_birthday import zeller

class Test_zellers_birthday(TestCase):
	def test_zeller_1(self):
		y = zeller(87,-70,-23)
		self.assertEqual(y,"Monday")

	def test_zeller_2(self):
		y = zeller(99,14,-34)
		self.assertEqual(y,"Wednesday")

	def test_zeller_3(self):
		y = zeller(95,68,6)
		self.assertEqual(y,"Sunday")

	def test_zeller_4(self):
		y = zeller(98,-9,66)
		self.assertEqual(y,"Tuesday")

	def test_zeller_5(self):
		y = zeller(-84,-32,98)
		self.assertEqual(y,"Wednesday")

	def test_zeller_6(self):
		y = zeller(-39,-33,24)
		self.assertEqual(y,"Thursday")

	def test_zeller_7(self):
		y = zeller(79,-76,99)
		self.assertEqual(y,"Tuesday")

	def test_zeller_8(self):
		y = zeller(55,18,-51)
		self.assertEqual(y,"Wednesday")

	def test_zeller_9(self):
		y = zeller(-87,50,-9)
		self.assertEqual(y,"Thursday")

	def test_zeller_10(self):
		y = zeller(50,-83,4)
		self.assertEqual(y,"Monday")

	def test_zeller_11(self):
		y = zeller(48,-82,-75)
		self.assertEqual(y,"Tuesday")

	def test_zeller_12(self):
		y = zeller(4,-69,31)
		self.assertEqual(y,"Sunday")

	def test_zeller_13(self):
		y = zeller(-34,-57,25)
		self.assertEqual(y,"Sunday")

	def test_zeller_14(self):
		y = zeller(-93,-23,52)
		self.assertEqual(y,"Monday")

	def test_zeller_15(self):
		y = zeller(93,-65,-47)
		self.assertEqual(y,"Sunday")

	def test_zeller_16(self):
		y = zeller(39,-37,21)
		self.assertEqual(y,"Tuesday")

	def test_zeller_17(self):
		y = zeller(-70,79,-48)
		self.assertEqual(y,"Monday")

	def test_zeller_18(self):
		y = zeller(-100,30,-13)
		self.assertEqual(y,"Monday")

