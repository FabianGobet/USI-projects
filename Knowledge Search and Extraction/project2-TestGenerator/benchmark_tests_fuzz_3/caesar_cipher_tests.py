from unittest import TestCase
from benchmark.caesar_cipher import encrypt, decrypt

class Test_caesar_cipher(TestCase):
	def test_encrypt_1(self):
		y = encrypt("g",53)
		self.assertEqual(y,"=")

	def test_encrypt_2(self):
		y = encrypt("iofkq",48)
		self.assertEqual(y,":@7<B")

	def test_encrypt_3(self):
		y = encrypt("lydajuzzuuhlslbifmf",3)
		self.assertEqual(y,"o|gdmx}}xxkovoelipi")

	def test_encrypt_4(self):
		y = encrypt("awxblyjwia",61)
		self.assertEqual(y,"?UV@JWHUG?")

	def test_decrypt_1(self):
		y = decrypt("itrzxqrqtjxppjdwkbxs",61)
		self.assertEqual(y,",75=;4547-;33-':.%;6")

	def test_decrypt_2(self):
		y = decrypt("fdsmtlidppn",85)
		self.assertEqual(y,"pn}w~vsnzzx")

