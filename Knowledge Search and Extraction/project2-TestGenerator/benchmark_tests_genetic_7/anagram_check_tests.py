from unittest import TestCase
from benchmark.anagram_check import anagram_check

class Test_anagram_check(TestCase):
	def test_anagram_check_1(self):
		y = anagram_check("sw","acragzwiukxhfroix")
		self.assertEqual(y,False)

	def test_anagram_check_2(self):
		y = anagram_check("","")
		self.assertEqual(y,True)

	def test_anagram_check_3(self):
		y = anagram_check("c","j")
		self.assertEqual(y,False)

	def test_anagram_check_4(self):
		y = anagram_check("psjgijecxgkdlmfvoik","tcq")
		self.assertEqual(y,False)

	def test_anagram_check_5(self):
		y = anagram_check("kun","nfy")
		self.assertEqual(y,False)

	def test_anagram_check_6(self):
		y = anagram_check("c","uhctc")
		self.assertEqual(y,False)

	def test_anagram_check_7(self):
		y = anagram_check("ixoqk","hg")
		self.assertEqual(y,False)

