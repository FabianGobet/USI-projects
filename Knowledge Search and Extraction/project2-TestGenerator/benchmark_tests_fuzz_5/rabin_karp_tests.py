from unittest import TestCase
from benchmark.rabin_karp import rabin_karp_search

class Test_rabin_karp(TestCase):
	def test_rabin_karp_search_1(self):
		y = rabin_karp_search("mltoyxg","cdjtnauiyovvwseqq")
		self.assertEqual(y,[])

	def test_rabin_karp_search_2(self):
		y = rabin_karp_search("z","zmfqqjz")
		self.assertEqual(y,[0,6])

	def test_rabin_karp_search_3(self):
		y = rabin_karp_search("n","mfusdoszed")
		self.assertEqual(y,[])

	def test_rabin_karp_search_4(self):
		y = rabin_karp_search("ttxqghmm","kvlsvcvhyzigemy")
		self.assertEqual(y,[])

	def test_rabin_karp_search_5(self):
		y = rabin_karp_search("","rszrunfuzeallyws")
		self.assertEqual(y,[])

