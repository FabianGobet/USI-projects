from unittest import TestCase
from benchmark.caesar_cipher import encrypt, decrypt

class Test_caesar_cipher(TestCase):
	def test_encrypt_1(self):
		y = encrypt("dzdbxzwmbkdtj",71)
		self.assertEqual(y,"LbLJ`b_UJSL\\R")

	def test_encrypt_2(self):
		y = encrypt("zuqvevhrapondwrrosqc",9)
		self.assertEqual(y,"$~z n q{jyxwm!{{x|zl")

	def test_decrypt_1(self):
		y = decrypt("dzoxdqidsjlk",84)
		self.assertEqual(y,"o&z$o|to~uwv")

