from unittest import TestCase
from benchmark.rabin_karp import rabin_karp_search

class Test_rabin_karp(TestCase):
	def test_rabin_karp_search_1(self):
		y = rabin_karp_search("asnvgiujijewp","zwwozrljgmwiifou")
		self.assertEqual(y,[])

	def test_rabin_karp_search_2(self):
		y = rabin_karp_search("te","shkqqyeayxv")
		self.assertEqual(y,[])

	def test_rabin_karp_search_3(self):
		y = rabin_karp_search("h","gohxe")
		self.assertEqual(y,[2])

	def test_rabin_karp_search_4(self):
		y = rabin_karp_search("bwhpn","jywhpgn")
		self.assertEqual(y,[])

	def test_rabin_karp_search_5(self):
		y = rabin_karp_search("","uglsshjeozvnoczvxtyp")
		self.assertEqual(y,[])

