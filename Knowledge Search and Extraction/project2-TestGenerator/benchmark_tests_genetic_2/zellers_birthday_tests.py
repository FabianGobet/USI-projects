from unittest import TestCase
from benchmark.zellers_birthday import zeller

class Test_zellers_birthday(TestCase):
	def test_zeller_1(self):
		y = zeller(-56,100,-1)
		self.assertEqual(y,"Saturday")

	def test_zeller_2(self):
		y = zeller(-90,-44,83)
		self.assertEqual(y,"Thursday")

	def test_zeller_3(self):
		y = zeller(97,59,95)
		self.assertEqual(y,"Tuesday")

	def test_zeller_4(self):
		y = zeller(70,-32,-96)
		self.assertEqual(y,"Monday")

	def test_zeller_5(self):
		y = zeller(-12,97,61)
		self.assertEqual(y,"Sunday")

	def test_zeller_6(self):
		y = zeller(42,51,23)
		self.assertEqual(y,"Thursday")

	def test_zeller_7(self):
		y = zeller(24,51,-52)
		self.assertEqual(y,"Thursday")

	def test_zeller_8(self):
		y = zeller(2,79,-25)
		self.assertEqual(y,"Sunday")

	def test_zeller_9(self):
		y = zeller(-22,86,93)
		self.assertEqual(y,"Monday")

	def test_zeller_10(self):
		y = zeller(-77,-90,-93)
		self.assertEqual(y,"Friday")

	def test_zeller_11(self):
		y = zeller(-74,-9,-90)
		self.assertEqual(y,"Thursday")

	def test_zeller_12(self):
		y = zeller(85,64,-99)
		self.assertEqual(y,"Monday")

