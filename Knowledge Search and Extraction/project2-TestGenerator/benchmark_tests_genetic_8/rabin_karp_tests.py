from unittest import TestCase
from benchmark.rabin_karp import rabin_karp_search

class Test_rabin_karp(TestCase):
	def test_rabin_karp_search_1(self):
		y = rabin_karp_search("dsgmlslegvokngvqlvy","tfwzpjvolcrrhrbrjqe")
		self.assertEqual(y,[])

	def test_rabin_karp_search_2(self):
		y = rabin_karp_search("","xzvujzixcomqyascuwkh")
		self.assertEqual(y,[])

	def test_rabin_karp_search_3(self):
		y = rabin_karp_search("hnxpxinl","wnxxiplvqusigmqhco")
		self.assertEqual(y,[])

	def test_rabin_karp_search_4(self):
		y = rabin_karp_search("hkofgblho","miwvpqeowzctsyj")
		self.assertEqual(y,[])

	def test_rabin_karp_search_5(self):
		y = rabin_karp_search("fzwa","yhnludohknoqihx")
		self.assertEqual(y,[])

	def test_rabin_karp_search_6(self):
		y = rabin_karp_search("c","smmsxlcoaiyoqzocbd")
		self.assertEqual(y,[6,15])

