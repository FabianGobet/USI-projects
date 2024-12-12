from unittest import TestCase
from benchmark.anagram_check import anagram_check

class Test_anagram_check(TestCase):
	def test_anagram_check_1(self):
		y = anagram_check("qzt","sys")
		self.assertEqual(y,False)

	def test_anagram_check_2(self):
		y = anagram_check("","")
		self.assertEqual(y,True)

	def test_anagram_check_3(self):
		y = anagram_check("cefriluj","ucxbu")
		self.assertEqual(y,False)

	def test_anagram_check_4(self):
		y = anagram_check("nedz","aryn")
		self.assertEqual(y,False)

	def test_anagram_check_5(self):
		y = anagram_check("k","q")
		self.assertEqual(y,False)

	def test_anagram_check_6(self):
		y = anagram_check("o","")
		self.assertEqual(y,False)

	def test_anagram_check_7(self):
		y = anagram_check("nosmgfwtufpcoxqve","eqguzcgifhiowbjoh")
		self.assertEqual(y,False)

	def test_anagram_check_8(self):
		y = anagram_check("h","xtqwbhrfareobjzbiib")
		self.assertEqual(y,False)

	def test_anagram_check_9(self):
		y = anagram_check("lwqtqyghbzntbybwhfhl","ugtyedhdwyqkeorjuhzp")
		self.assertEqual(y,False)

	def test_anagram_check_10(self):
		y = anagram_check("","mlavzh")
		self.assertEqual(y,False)

