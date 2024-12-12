from unittest import TestCase
from benchmark.zellers_birthday import zeller

class Test_zellers_birthday(TestCase):
	def test_zeller_1(self):
		y = zeller(-1,-46,-63)
		self.assertEqual(y,"Friday")

	def test_zeller_2(self):
		y = zeller(20,71,-30)
		self.assertEqual(y,"Saturday")

	def test_zeller_3(self):
		y = zeller(-74,6,86)
		self.assertEqual(y,"Friday")

	def test_zeller_4(self):
		y = zeller(48,37,74)
		self.assertEqual(y,"Monday")

	def test_zeller_5(self):
		y = zeller(-74,-84,-23)
		self.assertEqual(y,"Saturday")

	def test_zeller_6(self):
		y = zeller(-63,-31,-32)
		self.assertEqual(y,"Tuesday")

	def test_zeller_7(self):
		y = zeller(-38,17,-36)
		self.assertEqual(y,"Monday")

	def test_zeller_8(self):
		y = zeller(53,1,99)
		self.assertEqual(y,"Saturday")

	def test_zeller_9(self):
		y = zeller(52,98,20)
		self.assertEqual(y,"Sunday")

	def test_zeller_10(self):
		y = zeller(-78,69,98)
		self.assertEqual(y,"Saturday")

