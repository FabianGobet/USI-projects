from unittest import TestCase
from benchmark.railfence_cipher import railencrypt, raildecrypt

class Test_railfence_cipher(TestCase):
	def test_railencrypt_1(self):
		y = railencrypt("qiaceybyzsc",3)
		self.assertEqual(y,"qezicyysabc")

	def test_railencrypt_2(self):
		y = railencrypt("wmcavt",16)
		self.assertEqual(y,"wmcavt")

	def test_raildecrypt_1(self):
		y = raildecrypt("lfm",97)
		self.assertEqual(y,"lfm")

	def test_raildecrypt_2(self):
		y = raildecrypt("vzsawnwjeeqmyhgrkgc",3)
		self.assertEqual(y,"vngwzjresekqamgywhc")

	def test_raildecrypt_3(self):
		y = raildecrypt("mczubvahvmsoqjywe",50)
		self.assertEqual(y,"mczubvahvmsoqjywe")

