from unittest import TestCase
from benchmark.check_armstrong import check_armstrong

class Test_check_armstrong(TestCase):
	def test_check_armstrong_1(self):
		y = check_armstrong(12)
		self.assertEqual(y,False)

	def test_check_armstrong_2(self):
		y = check_armstrong(2)
		self.assertEqual(y,False)

	def test_check_armstrong_3(self):
		y = check_armstrong(95)
		self.assertEqual(y,False)

	def test_check_armstrong_4(self):
		y = check_armstrong(56)
		self.assertEqual(y,False)

	def test_check_armstrong_5(self):
		y = check_armstrong(98)
		self.assertEqual(y,False)

	def test_check_armstrong_6(self):
		y = check_armstrong(1)
		self.assertEqual(y,True)

	def test_check_armstrong_7(self):
		y = check_armstrong(100)
		self.assertEqual(y,False)

	def test_check_armstrong_8(self):
		y = check_armstrong(0)
		self.assertEqual(y,True)

	def test_check_armstrong_9(self):
		y = check_armstrong(35)
		self.assertEqual(y,False)

	def test_check_armstrong_10(self):
		y = check_armstrong(3)
		self.assertEqual(y,False)

