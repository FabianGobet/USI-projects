from unittest import TestCase
from benchmark.zellers_birthday import zeller

class Test_zellers_birthday(TestCase):
	def test_zeller_1(self):
		y = zeller(-3,-36,80)
		self.assertEqual(y,"Thursday")

	def test_zeller_2(self):
		y = zeller(-4,-40,5)
		self.assertEqual(y,"Wednesday")

	def test_zeller_3(self):
		y = zeller(76,41,-99)
		self.assertEqual(y,"Tuesday")

	def test_zeller_4(self):
		y = zeller(18,40,23)
		self.assertEqual(y,"Friday")

	def test_zeller_5(self):
		y = zeller(80,-62,-55)
		self.assertEqual(y,"Saturday")

	def test_zeller_6(self):
		y = zeller(83,-15,96)
		self.assertEqual(y,"Monday")

	def test_zeller_7(self):
		y = zeller(93,14,-49)
		self.assertEqual(y,"Tuesday")

	def test_zeller_8(self):
		y = zeller(43,-13,74)
		self.assertEqual(y,"Wednesday")

	def test_zeller_9(self):
		y = zeller(18,-5,78)
		self.assertEqual(y,"Thursday")

	def test_zeller_10(self):
		y = zeller(-86,-61,-31)
		self.assertEqual(y,"Wednesday")

	def test_zeller_11(self):
		y = zeller(38,48,-98)
		self.assertEqual(y,"Thursday")

