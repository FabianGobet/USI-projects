from unittest import TestCase
from benchmark.zellers_birthday import zeller

class Test_zellers_birthday(TestCase):
	def test_zeller_1(self):
		y = zeller(-86,-85,90)
		self.assertEqual(y,"Sunday")

	def test_zeller_2(self):
		y = zeller(43,34,46)
		self.assertEqual(y,"Wednesday")

	def test_zeller_3(self):
		y = zeller(29,46,-92)
		self.assertEqual(y,"Sunday")

	def test_zeller_4(self):
		y = zeller(53,-99,-30)
		self.assertEqual(y,"Wednesday")

	def test_zeller_5(self):
		y = zeller(-37,-14,-95)
		self.assertEqual(y,"Tuesday")

	def test_zeller_6(self):
		y = zeller(64,-8,48)
		self.assertEqual(y,"Tuesday")

	def test_zeller_7(self):
		y = zeller(-39,-59,-98)
		self.assertEqual(y,"Wednesday")

	def test_zeller_8(self):
		y = zeller(40,24,99)
		self.assertEqual(y,"Sunday")

	def test_zeller_9(self):
		y = zeller(-56,22,23)
		self.assertEqual(y,"Monday")

	def test_zeller_10(self):
		y = zeller(68,-65,97)
		self.assertEqual(y,"Saturday")

	def test_zeller_11(self):
		y = zeller(26,-25,53)
		self.assertEqual(y,"Thursday")

	def test_zeller_12(self):
		y = zeller(-52,71,10)
		self.assertEqual(y,"Wednesday")

	def test_zeller_13(self):
		y = zeller(46,81,-86)
		self.assertEqual(y,"Thursday")

	def test_zeller_14(self):
		y = zeller(-63,-58,25)
		self.assertEqual(y,"Monday")

	def test_zeller_15(self):
		y = zeller(94,-86,24)
		self.assertEqual(y,"Sunday")

	def test_zeller_16(self):
		y = zeller(-45,-64,27)
		self.assertEqual(y,"Sunday")

