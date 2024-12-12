from unittest import TestCase
from benchmark.zellers_birthday import zeller

class Test_zellers_birthday(TestCase):
	def test_zeller_1(self):
		y = zeller(68,41,-90)
		self.assertEqual(y,"Thursday")

	def test_zeller_2(self):
		y = zeller(-91,5,-57)
		self.assertEqual(y,"Thursday")

	def test_zeller_3(self):
		y = zeller(-74,25,-23)
		self.assertEqual(y,"Tuesday")

	def test_zeller_4(self):
		y = zeller(44,-69,-59)
		self.assertEqual(y,"Wednesday")

	def test_zeller_5(self):
		y = zeller(20,-36,36)
		self.assertEqual(y,"Monday")

	def test_zeller_6(self):
		y = zeller(97,5,-21)
		self.assertEqual(y,"Wednesday")

	def test_zeller_7(self):
		y = zeller(50,67,-95)
		self.assertEqual(y,"Sunday")

	def test_zeller_8(self):
		y = zeller(2,-59,-77)
		self.assertEqual(y,"Friday")

	def test_zeller_9(self):
		y = zeller(-55,-23,-99)
		self.assertEqual(y,"Saturday")

