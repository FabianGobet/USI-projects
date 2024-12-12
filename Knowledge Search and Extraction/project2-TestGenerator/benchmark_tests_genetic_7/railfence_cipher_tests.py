from unittest import TestCase
from benchmark.railfence_cipher import railencrypt, raildecrypt

class Test_railfence_cipher(TestCase):
	def test_railencrypt_1(self):
		y = railencrypt("rcufzjtxuvbxydnjfbk",31)
		self.assertEqual(y,"rcufzjtxuvbxydnjfbk")

	def test_railencrypt_2(self):
		y = railencrypt("uhyovzghyohccogsck",2)
		self.assertEqual(y,"uyvgyhcgchozhocosk")

	def test_railencrypt_3(self):
		y = railencrypt("yv",87)
		self.assertEqual(y,"yv")

	def test_railencrypt_4(self):
		y = railencrypt("hhhewlpgbsrxugb",24)
		self.assertEqual(y,"hhhewlpgbsrxugb")

	def test_railencrypt_5(self):
		y = railencrypt("lqm",75)
		self.assertEqual(y,"lqm")

	def test_railencrypt_6(self):
		y = railencrypt("qgyffxjuta",9)
		self.assertEqual(y,"qgyffxjuat")

	def test_raildecrypt_1(self):
		y = raildecrypt("hwqnvxegediskmxvstp",5)
		self.assertEqual(y,"hnemtxdvwxivpsseqgk")

	def test_raildecrypt_2(self):
		y = raildecrypt("wnafb",57)
		self.assertEqual(y,"wnafb")

