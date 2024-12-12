from unittest import TestCase
from benchmark.anagram_check import anagram_check

class Test_anagram_check(TestCase):
	def test_anagram_check_1(self):
		y = anagram_check("u","nwolqwbruhvkab")
		self.assertEqual(y,False)

	def test_anagram_check_2(self):
		y = anagram_check("","")
		self.assertEqual(y,True)

	def test_anagram_check_3(self):
		y = anagram_check("srvl","crwrjdblvwbrqo")
		self.assertEqual(y,False)

	def test_anagram_check_4(self):
		y = anagram_check("i","q")
		self.assertEqual(y,False)

	def test_anagram_check_5(self):
		y = anagram_check("ogrduubpbkk","ashskjdieln")
		self.assertEqual(y,False)

	def test_anagram_check_6(self):
		y = anagram_check("lpawnexqgad","tdghzkebgq")
		self.assertEqual(y,False)

	def test_anagram_check_7(self):
		y = anagram_check("dphtdwoyujghn","svxkpm")
		self.assertEqual(y,False)

	def test_anagram_check_8(self):
		y = anagram_check("v","easyo")
		self.assertEqual(y,False)

	def test_anagram_check_9(self):
		y = anagram_check("r","bbaa")
		self.assertEqual(y,False)

	def test_anagram_check_10(self):
		y = anagram_check("b","")
		self.assertEqual(y,False)

	def test_anagram_check_11(self):
		y = anagram_check("f","xqutqdvqg")
		self.assertEqual(y,False)

	def test_anagram_check_12(self):
		y = anagram_check("pk","ub")
		self.assertEqual(y,False)

	def test_anagram_check_13(self):
		y = anagram_check("ddt","fpj")
		self.assertEqual(y,False)

	def test_anagram_check_14(self):
		y = anagram_check("r","gutzdkfdych")
		self.assertEqual(y,False)

	def test_anagram_check_15(self):
		y = anagram_check("ju","mozvvagoprbmdwncmd")
		self.assertEqual(y,False)

	def test_anagram_check_16(self):
		y = anagram_check("gkxonjhxufqeyiezfu","hllxuaywy")
		self.assertEqual(y,False)

