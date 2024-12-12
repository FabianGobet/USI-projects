from unittest import TestCase
from benchmark.zellers_birthday import zeller

class Test_zellers_birthday(TestCase):
	def test_zeller_1(self):
		y = zeller(-44,96,50)
		self.assertEqual(y,"Saturday")

	def test_zeller_2(self):
		y = zeller(62,6,69)
		self.assertEqual(y,"Sunday")

	def test_zeller_3(self):
		y = zeller(33,-2,25)
		self.assertEqual(y,"Tuesday")

	def test_zeller_4(self):
		y = zeller(4,-65,99)
		self.assertEqual(y,"Friday")

	def test_zeller_5(self):
		y = zeller(68,-78,-98)
		self.assertEqual(y,"Tuesday")

	def test_zeller_6(self):
		y = zeller(25,-81,-7)
		self.assertEqual(y,"Thursday")

	def test_zeller_7(self):
		y = zeller(-71,-93,-24)
		self.assertEqual(y,"Friday")

	def test_zeller_8(self):
		y = zeller(87,13,35)
		self.assertEqual(y,"Tuesday")

	def test_zeller_9(self):
		y = zeller(46,69,23)
		self.assertEqual(y,"Tuesday")

