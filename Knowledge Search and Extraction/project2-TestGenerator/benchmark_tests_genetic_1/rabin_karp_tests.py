from unittest import TestCase
from benchmark.rabin_karp import rabin_karp_search

class Test_rabin_karp(TestCase):
	def test_rabin_karp_search_1(self):
		y = rabin_karp_search("eepzmio","sdhnvfdlmzabseq")
		self.assertEqual(y,[])

	def test_rabin_karp_search_2(self):
		y = rabin_karp_search("e","eillghcvegrtiksnb")
		self.assertEqual(y,[0,8])

	def test_rabin_karp_search_3(self):
		y = rabin_karp_search("ggvqt","rdvtgttgsnsnbicpetdf")
		self.assertEqual(y,[])

	def test_rabin_karp_search_4(self):
		y = rabin_karp_search("","wsjtx")
		self.assertEqual(y,[])

	def test_rabin_karp_search_5(self):
		y = rabin_karp_search("cqpowfjoanlqm","apfbcdorigkkabz")
		self.assertEqual(y,[])

	def test_rabin_karp_search_6(self):
		y = rabin_karp_search("jaqyzdnqpbyzaokk","kdsyyuqcrzewvuvoklj")
		self.assertEqual(y,[])

	def test_rabin_karp_search_7(self):
		y = rabin_karp_search("qnibqnlpzo","vbeykchcsha")
		self.assertEqual(y,[])

