from unittest import TestCase
from examples.example4 import f, g

class Test_example4(TestCase):
	def test_f_1(self):
		y = f(2,-10)
		self.assertEqual(y,2)

	def test_f_2(self):
		y = f(2,-7)
		self.assertEqual(y,2)

	def test_f_3(self):
		y = f(7,-10)
		self.assertEqual(y,7)

	def test_f_4(self):
		y = f(0,-7)
		self.assertEqual(y,0)

	def test_f_5(self):
		y = f(-10,-8)
		self.assertEqual(y,-8)

	def test_f_6(self):
		y = f(-9,-5)
		self.assertEqual(y,-5)

	def test_f_7(self):
		y = f(0,7)
		self.assertEqual(y,7)

	def test_f_8(self):
		y = f(-10,5)
		self.assertEqual(y,5)

	def test_f_9(self):
		y = f(-3,7)
		self.assertEqual(y,7)

	def test_f_10(self):
		y = f(1,9)
		self.assertEqual(y,9)

	def test_g_1(self):
		y = g(9,-2)
		self.assertEqual(y,11)

	def test_g_2(self):
		y = g(4,6)
		self.assertEqual(y,2)

