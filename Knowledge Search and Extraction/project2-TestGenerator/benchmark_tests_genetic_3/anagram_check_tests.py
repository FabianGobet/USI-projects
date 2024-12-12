from unittest import TestCase
from benchmark.anagram_check import anagram_check

class Test_anagram_check(TestCase):
	def test_anagram_check_1(self):
		y = anagram_check("af","cf")
		self.assertEqual(y,False)

	def test_anagram_check_2(self):
		y = anagram_check("","")
		self.assertEqual(y,True)

