from unittest import TestCase
from benchmark.anagram_check import anagram_check

class Test_anagram_check(TestCase):
	def test_anagram_check_1(self):
		y = anagram_check("hkrsesdgpzvpoaufutv","jlpnnphibregjazxszt")
		self.assertEqual(y,False)

	def test_anagram_check_2(self):
		y = anagram_check("z","xjvcqzpqp")
		self.assertEqual(y,False)

	def test_anagram_check_3(self):
		y = anagram_check("u","jmlvepjo")
		self.assertEqual(y,False)

	def test_anagram_check_4(self):
		y = anagram_check("amlhpyylylruds","hwcssmixxji")
		self.assertEqual(y,False)

	def test_anagram_check_5(self):
		y = anagram_check("v","l")
		self.assertEqual(y,False)

	def test_anagram_check_6(self):
		y = anagram_check("","")
		self.assertEqual(y,True)

	def test_anagram_check_7(self):
		y = anagram_check("mkrxcxphwuuxqflyuw","rn")
		self.assertEqual(y,False)

	def test_anagram_check_8(self):
		y = anagram_check("ruemxa","izudqr")
		self.assertEqual(y,False)

	def test_anagram_check_9(self):
		y = anagram_check("nrjskydty","hgkzuqgtmpael")
		self.assertEqual(y,False)

