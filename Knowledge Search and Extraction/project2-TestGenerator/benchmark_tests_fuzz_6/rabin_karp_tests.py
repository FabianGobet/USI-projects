from unittest import TestCase
from benchmark.rabin_karp import rabin_karp_search

class Test_rabin_karp(TestCase):
	def test_rabin_karp_search_1(self):
		y = rabin_karp_search("k","frqneqytowvrk")
		self.assertEqual(y,[12])

	def test_rabin_karp_search_2(self):
		y = rabin_karp_search("g","tyrufurryhjzuk")
		self.assertEqual(y,[])

	def test_rabin_karp_search_3(self):
		y = rabin_karp_search("fyjj","sgetyjnlvuksssvoibdc")
		self.assertEqual(y,[])

	def test_rabin_karp_search_4(self):
		y = rabin_karp_search("yzpxq","namecairljrtqpaovpbv")
		self.assertEqual(y,[])

