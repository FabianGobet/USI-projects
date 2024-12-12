from unittest import TestCase
from benchmark.caesar_cipher import encrypt, decrypt

class Test_caesar_cipher(TestCase):
	def test_encrypt_1(self):
		y = encrypt("zjkyixnwjhmzo",1)
		self.assertEqual(y,"{klzjyoxkin{p")

	def test_encrypt_2(self):
		y = encrypt("qxcftazoitw",44)
		self.assertEqual(y,">E03A.G<6AD")

	def test_decrypt_1(self):
		y = decrypt("ddmedf",39)
		self.assertEqual(y,"==F>=?")

	def test_decrypt_2(self):
		y = decrypt("jnwh",78)
		self.assertEqual(y,"{ )y")

