from unittest import TestCase
from benchmark.caesar_cipher import encrypt, decrypt

class Test_caesar_cipher(TestCase):
	def test_encrypt_1(self):
		y = encrypt("henwplhlmow",19)
		self.assertEqual(y,"{x\"+$ { !#+")

	def test_decrypt_1(self):
		y = decrypt("ddz",64)
		self.assertEqual(y,"$$:")

	def test_decrypt_2(self):
		y = decrypt("xyxuvi",82)
		self.assertEqual(y,"&'&#$v")

	def test_decrypt_3(self):
		y = decrypt("jdijvlioi",57)
		self.assertEqual(y,"1+01=3060")

	def test_decrypt_4(self):
		y = decrypt("e",33)
		self.assertEqual(y,"D")

