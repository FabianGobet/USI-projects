from unittest import TestCase
from benchmark.rabin_karp import rabin_karp_search

class Test_rabin_karp(TestCase):
	def test_rabin_karp_search_1(self):
		y = rabin_karp_search("z","qeabfzjjlwmsk")
		self.assertEqual(y,[5])

	def test_rabin_karp_search_2(self):
		y = rabin_karp_search("hpassb","xqpxmqrpvd")
		self.assertEqual(y,[])

	def test_rabin_karp_search_3(self):
		y = rabin_karp_search("zfmh","urcyczmjyzgvecgq")
		self.assertEqual(y,[])

	def test_rabin_karp_search_4(self):
		y = rabin_karp_search("gjmm","ihnlfewcbgc")
		self.assertEqual(y,[])

	def test_rabin_karp_search_5(self):
		y = rabin_karp_search("wnu","acmibqxzeb")
		self.assertEqual(y,[])

	def test_rabin_karp_search_6(self):
		y = rabin_karp_search("","vofyaufjfyewrhkyl")
		self.assertEqual(y,[])

	def test_rabin_karp_search_7(self):
		y = rabin_karp_search("o","wexelhrteqs")
		self.assertEqual(y,[])

