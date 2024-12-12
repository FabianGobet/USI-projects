from unittest import TestCase
from benchmark.railfence_cipher import railencrypt, raildecrypt

class Test_railfence_cipher(TestCase):
	def test_railencrypt_1(self):
		y = railencrypt("zlhpzlmphktri",5)
		self.assertEqual(y,"zhlpkhmtplrzi")

	def test_railencrypt_2(self):
		y = railencrypt("qnqumqb",6)
		self.assertEqual(y,"qnqumbq")

	def test_railencrypt_3(self):
		y = railencrypt("lhcecxkdgggwrmomget",23)
		self.assertEqual(y,"lhcecxkdgggwrmomget")

	def test_raildecrypt_1(self):
		y = raildecrypt("fysicpggzfdduclrjwan",16)
		self.assertEqual(y,"fysicpggzfddcrwnajlu")

	def test_raildecrypt_2(self):
		y = raildecrypt("jgoixcmznyun",8)
		self.assertEqual(y,"jgoiczynunmx")

	def test_raildecrypt_3(self):
		y = raildecrypt("oyxwpeunksitkk",76)
		self.assertEqual(y,"oyxwpeunksitkk")

	def test_raildecrypt_4(self):
		y = raildecrypt("fhrejwopwvxbonjcxapk",6)
		self.assertEqual(y,"froxjpcbpehjwoxkanvw")

