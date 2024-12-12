from unittest import TestCase
from benchmark.zellers_birthday import zeller

class Test_zellers_birthday(TestCase):
	def test_zeller_1(self):
		y = zeller(81,100,-99)
		self.assertEqual(y,"Thursday")

	def test_zeller_2(self):
		y = zeller(-48,-80,-10)
		self.assertEqual(y,"Saturday")

	def test_zeller_3(self):
		y = zeller(-63,-11,-36)
		self.assertEqual(y,"Monday")

	def test_zeller_4(self):
		y = zeller(39,-8,23)
		self.assertEqual(y,"Thursday")

	def test_zeller_5(self):
		y = zeller(7,-69,-27)
		self.assertEqual(y,"Friday")

	def test_zeller_6(self):
		y = zeller(-23,-60,-38)
		self.assertEqual(y,"Sunday")

	def test_zeller_7(self):
		y = zeller(62,-93,-29)
		self.assertEqual(y,"Tuesday")

	def test_zeller_8(self):
		y = zeller(-83,-72,24)
		self.assertEqual(y,"Tuesday")

	def test_zeller_9(self):
		y = zeller(-80,-50,-48)
		self.assertEqual(y,"Friday")

