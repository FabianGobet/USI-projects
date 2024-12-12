from unittest import TestCase
from benchmark.anagram_check import anagram_check

class Test_anagram_check(TestCase):
	def test_anagram_check_1(self):
		y = anagram_check("urbwzmjhznurlgf","aefnrfrhxosyqwgn")
		self.assertEqual(y,False)

	def test_anagram_check_2(self):
		y = anagram_check("ui","krruktybdoihloxe")
		self.assertEqual(y,False)

	def test_anagram_check_3(self):
		y = anagram_check("n","vdgzveoxwlegiqk")
		self.assertEqual(y,False)

	def test_anagram_check_4(self):
		y = anagram_check("","")
		self.assertEqual(y,True)

	def test_anagram_check_5(self):
		y = anagram_check("u","cndlzuvhsqor")
		self.assertEqual(y,False)

	def test_anagram_check_6(self):
		y = anagram_check("u","s")
		self.assertEqual(y,False)

	def test_anagram_check_7(self):
		y = anagram_check("jmrndxwu","scvjd")
		self.assertEqual(y,False)

	def test_anagram_check_8(self):
		y = anagram_check("akuddw","kockrsbsihhr")
		self.assertEqual(y,False)

	def test_anagram_check_9(self):
		y = anagram_check("wldm","uagi")
		self.assertEqual(y,False)

	def test_anagram_check_10(self):
		y = anagram_check("ki","ml")
		self.assertEqual(y,False)

