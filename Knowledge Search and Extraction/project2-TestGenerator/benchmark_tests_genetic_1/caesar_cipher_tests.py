from unittest import TestCase
from benchmark.caesar_cipher import encrypt, decrypt

class Test_caesar_cipher(TestCase):
	def test_encrypt_1(self):
		y = encrypt("blsrne",14)
		self.assertEqual(y,"pz\"!|s")

	def test_decrypt_1(self):
		y = decrypt("orjcjssrebpkxpkvjij",40)
		self.assertEqual(y,"GJB;BKKJ=:HCPHCNBAB")

	def test_decrypt_2(self):
		y = decrypt("nhfto",90)
		self.assertEqual(y,"smkyt")

