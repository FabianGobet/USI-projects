from unittest import TestCase
from benchmark.longest_substring import longest_sorted_substr

class Test_longest_substring(TestCase):
	def test_longest_sorted_substr_1(self):
		y = longest_sorted_substr("yqbwtuymxuves")
		self.assertEqual(y,"tuy")

	def test_longest_sorted_substr_2(self):
		y = longest_sorted_substr("qmzul")
		self.assertEqual(y,"mz")

