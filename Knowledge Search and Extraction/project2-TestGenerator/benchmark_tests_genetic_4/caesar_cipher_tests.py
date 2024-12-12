from unittest import TestCase
from benchmark.caesar_cipher import encrypt, decrypt

class Test_caesar_cipher(TestCase):
	def test_encrypt_1(self):
		y = encrypt("jpnrhkxzbjosr",45)
		self.assertEqual(y,"8><@69FH08=A@")

	def test_encrypt_2(self):
		y = encrypt("mxntifj",30)
		self.assertEqual(y,",7-3(%)")

	def test_encrypt_3(self):
		y = encrypt("sqtfkyvwhzsaabahy",94)
		self.assertEqual(y,"rpsejxuvgyr``a`gx")

	def test_encrypt_4(self):
		y = encrypt("ujaoqnq",51)
		self.assertEqual(y,"I>5CEBE")

	def test_encrypt_5(self):
		y = encrypt("obbsaqlvaevxtd",5)
		self.assertEqual(y,"tggxfvq{fj{}yi")

	def test_decrypt_1(self):
		y = decrypt("ocjfucfahpp",54)
		self.assertEqual(y,"9-40?-0+2::")

	def test_decrypt_2(self):
		y = decrypt("x",30)
		self.assertEqual(y,"Z")

	def test_decrypt_3(self):
		y = decrypt("jemlyhzv",77)
		self.assertEqual(y,"|w ~,z-)")

