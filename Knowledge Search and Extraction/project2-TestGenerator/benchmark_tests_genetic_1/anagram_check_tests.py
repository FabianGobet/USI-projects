from unittest import TestCase
from benchmark.anagram_check import anagram_check

class Test_anagram_check(TestCase):
	def test_anagram_check_1(self):
		y = anagram_check("f","k")
		self.assertEqual(y,False)

	def test_anagram_check_2(self):
		y = anagram_check("i","izd")
		self.assertEqual(y,False)

	def test_anagram_check_3(self):
		y = anagram_check("","")
		self.assertEqual(y,True)

	def test_anagram_check_4(self):
		y = anagram_check("ao","go")
		self.assertEqual(y,False)

	def test_anagram_check_5(self):
		y = anagram_check("gzqieq","apfqqiwyh")
		self.assertEqual(y,False)

	def test_anagram_check_6(self):
		y = anagram_check("r","xlgk")
		self.assertEqual(y,False)

	def test_anagram_check_7(self):
		y = anagram_check("gx","ko")
		self.assertEqual(y,False)

	def test_anagram_check_8(self):
		y = anagram_check("xzdzswczufzuc","yagedytlbcj")
		self.assertEqual(y,False)

	def test_anagram_check_9(self):
		y = anagram_check("l","bt")
		self.assertEqual(y,False)

