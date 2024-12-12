from unittest import TestCase
from examples.example3 import f

class Test_example3(TestCase):
	def test_f_1(self):
		y = f(10,0)
		self.assertEqual(y,10)

	def test_f_2(self):
		y = f(-1,10)
		self.assertEqual(y,10)

	def test_f_3(self):
		y = f(10,8)
		self.assertEqual(y,10)

	def test_f_4(self):
		y = f(8,9)
		self.assertEqual(y,9)

	def test_f_5(self):
		y = f(-8,-5)
		self.assertEqual(y,-5)

	def test_f_6(self):
		y = f(-9,-1)
		self.assertEqual(y,-1)

	def test_f_7(self):
		y = f(9,-7)
		self.assertEqual(y,9)

