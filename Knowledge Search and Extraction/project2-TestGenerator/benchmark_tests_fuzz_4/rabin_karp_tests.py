from unittest import TestCase
from benchmark.rabin_karp import rabin_karp_search

class Test_rabin_karp(TestCase):
	def test_rabin_karp_search_1(self):
		y = rabin_karp_search("rzmmb","pqqbrdtrorspn")
		self.assertEqual(y,[])

	def test_rabin_karp_search_2(self):
		y = rabin_karp_search("rljyxeutuio","dgpsrvyiwuvnywu")
		self.assertEqual(y,[])

	def test_rabin_karp_search_3(self):
		y = rabin_karp_search("","qzryqkyhudiqzicf")
		self.assertEqual(y,[])

	def test_rabin_karp_search_4(self):
		y = rabin_karp_search("ozbscwubebfxd","cixcepyumwgosprinf")
		self.assertEqual(y,[])

	def test_rabin_karp_search_5(self):
		y = rabin_karp_search("s","kkyesmxlkfqmuh")
		self.assertEqual(y,[4])

	def test_rabin_karp_search_6(self):
		y = rabin_karp_search("v","wkhd")
		self.assertEqual(y,[])

