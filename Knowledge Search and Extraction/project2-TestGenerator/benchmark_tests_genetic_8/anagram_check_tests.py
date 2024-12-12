from unittest import TestCase
from benchmark.anagram_check import anagram_check

class Test_anagram_check(TestCase):
	def test_anagram_check_1(self):
		y = anagram_check("hpblkmqf","glkmpjaunev")
		self.assertEqual(y,False)

	def test_anagram_check_2(self):
		y = anagram_check("","")
		self.assertEqual(y,True)

	def test_anagram_check_3(self):
		y = anagram_check("","mogmdzziqgvajngppl")
		self.assertEqual(y,False)

	def test_anagram_check_4(self):
		y = anagram_check("x","sy")
		self.assertEqual(y,False)

	def test_anagram_check_5(self):
		y = anagram_check("gmtud","rswofzymtoezapklb")
		self.assertEqual(y,False)

	def test_anagram_check_6(self):
		y = anagram_check("n","xucnujullztwtpzfyvj")
		self.assertEqual(y,False)

	def test_anagram_check_7(self):
		y = anagram_check("icxoh","zrvkc")
		self.assertEqual(y,False)

	def test_anagram_check_8(self):
		y = anagram_check("i","otzb")
		self.assertEqual(y,False)

	def test_anagram_check_9(self):
		y = anagram_check("z","z")
		self.assertEqual(y,True)

