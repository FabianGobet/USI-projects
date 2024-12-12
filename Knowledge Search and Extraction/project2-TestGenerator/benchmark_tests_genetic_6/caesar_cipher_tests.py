from unittest import TestCase
from benchmark.caesar_cipher import encrypt, decrypt

class Test_caesar_cipher(TestCase):
	def test_encrypt_1(self):
		y = encrypt("mhkarigwzsjlttd",15)
		self.assertEqual(y,"|wzp\"xv'*#y{$$s")

	def test_encrypt_2(self):
		y = encrypt("vxnftuxqfloe",35)
		self.assertEqual(y,":<2*89<5*03)")

	def test_decrypt_1(self):
		y = decrypt("wswqslmufmehdpywgip",57)
		self.assertEqual(y,">:>8:34<-4,/+7@>.07")

	def test_decrypt_2(self):
		y = decrypt("fcee",89)
		self.assertEqual(y,"likk")

	def test_decrypt_3(self):
		y = decrypt("pkfguwwmhi",55)
		self.assertEqual(y,"94/0>@@612")

