from unittest import TestCase
from benchmark.zellers_birthday import zeller

class Test_zellers_birthday(TestCase):
	def test_zeller_1(self):
		y = zeller(-53,-2,95)
		self.assertEqual(y,"Thursday")

	def test_zeller_2(self):
		y = zeller(-81,-93,99)
		self.assertEqual(y,"Wednesday")

	def test_zeller_3(self):
		y = zeller(-24,-45,-6)
		self.assertEqual(y,"Tuesday")

	def test_zeller_4(self):
		y = zeller(-51,-97,34)
		self.assertEqual(y,"Wednesday")

	def test_zeller_5(self):
		y = zeller(80,99,-97)
		self.assertEqual(y,"Saturday")

	def test_zeller_6(self):
		y = zeller(13,-73,-23)
		self.assertEqual(y,"Tuesday")

	def test_zeller_7(self):
		y = zeller(56,12,-25)
		self.assertEqual(y,"Saturday")

	def test_zeller_8(self):
		y = zeller(84,-5,29)
		self.assertEqual(y,"Thursday")

	def test_zeller_9(self):
		y = zeller(-23,-70,30)
		self.assertEqual(y,"Sunday")

	def test_zeller_10(self):
		y = zeller(-44,-87,-72)
		self.assertEqual(y,"Friday")

	def test_zeller_11(self):
		y = zeller(86,5,-82)
		self.assertEqual(y,"Tuesday")

	def test_zeller_12(self):
		y = zeller(4,7,-27)
		self.assertEqual(y,"Monday")

	def test_zeller_13(self):
		y = zeller(2,-55,35)
		self.assertEqual(y,"Friday")

