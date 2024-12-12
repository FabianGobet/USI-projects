from unittest import TestCase
from benchmark.rabin_karp import rabin_karp_search

class Test_rabin_karp(TestCase):
	def test_rabin_karp_search_1(self):
		y = rabin_karp_search("xuejqmn","mzolvkvbufvnd")
		self.assertEqual(y,[])

	def test_rabin_karp_search_2(self):
		y = rabin_karp_search("xoqrp","bpxeghpawfq")
		self.assertEqual(y,[])

	def test_rabin_karp_search_3(self):
		y = rabin_karp_search("gst","rfjuexnglkdpv")
		self.assertEqual(y,[])

	def test_rabin_karp_search_4(self):
		y = rabin_karp_search("","lakboghxz")
		self.assertEqual(y,[])

	def test_rabin_karp_search_5(self):
		y = rabin_karp_search("sfhcc","qazogwbdhbmjxx")
		self.assertEqual(y,[])

	def test_rabin_karp_search_6(self):
		y = rabin_karp_search("b","wbdpc")
		self.assertEqual(y,[1])

	def test_rabin_karp_search_7(self):
		y = rabin_karp_search("eitz","zfsbyyqvxtc")
		self.assertEqual(y,[])

