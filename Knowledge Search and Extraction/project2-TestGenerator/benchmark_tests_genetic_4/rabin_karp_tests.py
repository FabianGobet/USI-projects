from unittest import TestCase
from benchmark.rabin_karp import rabin_karp_search

class Test_rabin_karp(TestCase):
	def test_rabin_karp_search_1(self):
		y = rabin_karp_search("bogn","lcefetyrwtghwlggxx")
		self.assertEqual(y,[])

	def test_rabin_karp_search_2(self):
		y = rabin_karp_search("bjxe","kqotvevrsqjcz")
		self.assertEqual(y,[])

	def test_rabin_karp_search_3(self):
		y = rabin_karp_search("eebrc","qxsgsj")
		self.assertEqual(y,[])

	def test_rabin_karp_search_4(self):
		y = rabin_karp_search("nvg","bzaabprjw")
		self.assertEqual(y,[])

	def test_rabin_karp_search_5(self):
		y = rabin_karp_search("mzoabpzi","kipquttcanulunwggi")
		self.assertEqual(y,[])

	def test_rabin_karp_search_6(self):
		y = rabin_karp_search("axymxl","zrxonzidrypexicgoru")
		self.assertEqual(y,[])

	def test_rabin_karp_search_7(self):
		y = rabin_karp_search("xhzovemm","egludmzfkdarcoyhdm")
		self.assertEqual(y,[])

	def test_rabin_karp_search_8(self):
		y = rabin_karp_search("","a")
		self.assertEqual(y,[])

	def test_rabin_karp_search_9(self):
		y = rabin_karp_search("a","joai")
		self.assertEqual(y,[2])

