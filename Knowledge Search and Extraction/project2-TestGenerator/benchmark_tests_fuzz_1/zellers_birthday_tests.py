from unittest import TestCase
from benchmark.zellers_birthday import zeller

class Test_zellers_birthday(TestCase):
	def test_zeller_1(self):
		y = zeller(63,10,97)
		self.assertEqual(y,"Thursday")

	def test_zeller_2(self):
		y = zeller(57,94,-23)
		self.assertEqual(y,"Tuesday")

	def test_zeller_3(self):
		y = zeller(-85,18,99)
		self.assertEqual(y,"Saturday")

	def test_zeller_4(self):
		y = zeller(-99,-37,-5)
		self.assertEqual(y,"Monday")

	def test_zeller_5(self):
		y = zeller(-44,36,24)
		self.assertEqual(y,"Monday")

	def test_zeller_6(self):
		y = zeller(-28,58,-59)
		self.assertEqual(y,"Saturday")

