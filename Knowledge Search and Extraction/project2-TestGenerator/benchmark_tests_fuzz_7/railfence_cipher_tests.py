from unittest import TestCase
from benchmark.railfence_cipher import railencrypt, raildecrypt

class Test_railfence_cipher(TestCase):
	def test_railencrypt_1(self):
		y = railencrypt("us",6)
		self.assertEqual(y,"us")

	def test_railencrypt_2(self):
		y = railencrypt("xtkcpf",2)
		self.assertEqual(y,"xkptcf")

	def test_railencrypt_3(self):
		y = railencrypt("hdmzwwmepjbkhrs",17)
		self.assertEqual(y,"hdmzwwmepjbkhrs")

	def test_railencrypt_4(self):
		y = railencrypt("nygvhwitvfxzowzqbhbf",18)
		self.assertEqual(y,"nygvhwitvfxzowzqfbbh")

	def test_railencrypt_5(self):
		y = railencrypt("iirhrc",34)
		self.assertEqual(y,"iirhrc")

	def test_railencrypt_6(self):
		y = railencrypt("mjfirkeltcnxfwl",30)
		self.assertEqual(y,"mjfirkeltcnxfwl")

	def test_raildecrypt_1(self):
		y = raildecrypt("eebtryzwdzigr",5)
		self.assertEqual(y,"ebydgzzterwir")

	def test_raildecrypt_2(self):
		y = raildecrypt("bmfdgwb",53)
		self.assertEqual(y,"bmfdgwb")

	def test_raildecrypt_3(self):
		y = raildecrypt("ptcxqbmytguwulaf",100)
		self.assertEqual(y,"ptcxqbmytguwulaf")

	def test_raildecrypt_4(self):
		y = raildecrypt("qwbqtljncuwzct",19)
		self.assertEqual(y,"qwbqtljncuwzct")

