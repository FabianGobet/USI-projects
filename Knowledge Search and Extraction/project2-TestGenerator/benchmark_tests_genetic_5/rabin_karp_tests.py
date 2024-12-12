from unittest import TestCase
from benchmark.rabin_karp import rabin_karp_search

class Test_rabin_karp(TestCase):
	def test_rabin_karp_search_1(self):
		y = rabin_karp_search("nyk","gjurmkmvk")
		self.assertEqual(y,[])

	def test_rabin_karp_search_2(self):
		y = rabin_karp_search("hmxbfrajp","pjnucnyjrsckrortkyhz")
		self.assertEqual(y,[])

	def test_rabin_karp_search_3(self):
		y = rabin_karp_search("","pbtgvkax")
		self.assertEqual(y,[])

	def test_rabin_karp_search_4(self):
		y = rabin_karp_search("v","nvdeyjfczjwox")
		self.assertEqual(y,[1])

	def test_rabin_karp_search_5(self):
		y = rabin_karp_search("expsmprly","sybwqocetmtdkf")
		self.assertEqual(y,[])

