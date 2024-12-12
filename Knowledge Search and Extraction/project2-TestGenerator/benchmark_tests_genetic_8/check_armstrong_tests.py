from unittest import TestCase
from benchmark.check_armstrong import check_armstrong

class Test_check_armstrong(TestCase):
	def test_check_armstrong_1(self):
		y = check_armstrong(73)
		self.assertEqual(y,False)

	def test_check_armstrong_2(self):
		y = check_armstrong(12)
		self.assertEqual(y,False)

	def test_check_armstrong_3(self):
		y = check_armstrong(6)
		self.assertEqual(y,False)

	def test_check_armstrong_4(self):
		y = check_armstrong(2)
		self.assertEqual(y,False)

	def test_check_armstrong_5(self):
		y = check_armstrong(96)
		self.assertEqual(y,False)

	def test_check_armstrong_6(self):
		y = check_armstrong(60)
		self.assertEqual(y,False)

	def test_check_armstrong_7(self):
		y = check_armstrong(15)
		self.assertEqual(y,False)

	def test_check_armstrong_8(self):
		y = check_armstrong(95)
		self.assertEqual(y,False)

	def test_check_armstrong_9(self):
		y = check_armstrong(14)
		self.assertEqual(y,False)

	def test_check_armstrong_10(self):
		y = check_armstrong(1)
		self.assertEqual(y,True)

	def test_check_armstrong_11(self):
		y = check_armstrong(97)
		self.assertEqual(y,False)

	def test_check_armstrong_12(self):
		y = check_armstrong(3)
		self.assertEqual(y,False)

	def test_check_armstrong_13(self):
		y = check_armstrong(87)
		self.assertEqual(y,False)

	def test_check_armstrong_14(self):
		y = check_armstrong(26)
		self.assertEqual(y,False)

	def test_check_armstrong_15(self):
		y = check_armstrong(100)
		self.assertEqual(y,False)

	def test_check_armstrong_16(self):
		y = check_armstrong(0)
		self.assertEqual(y,True)

	def test_check_armstrong_17(self):
		y = check_armstrong(93)
		self.assertEqual(y,False)

	def test_check_armstrong_18(self):
		y = check_armstrong(99)
		self.assertEqual(y,False)

