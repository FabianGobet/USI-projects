from unittest import TestCase
from benchmark.railfence_cipher import railencrypt, raildecrypt

class Test_railfence_cipher(TestCase):
	def test_railencrypt_1(self):
		y = railencrypt("m",67)
		self.assertEqual(y,"m")

	def test_railencrypt_2(self):
		y = railencrypt("vvaedpfjaqiaacy",4)
		self.assertEqual(y,"vfavpjacadaiyeq")

	def test_railencrypt_3(self):
		y = railencrypt("dodiftlolpmmgoyarqud",21)
		self.assertEqual(y,"dodiftlolpmmgoyarqud")

	def test_railencrypt_4(self):
		y = railencrypt("htqlgmmikbcjwdkezg",23)
		self.assertEqual(y,"htqlgmmikbcjwdkezg")

	def test_railencrypt_5(self):
		y = railencrypt("duagbcn",47)
		self.assertEqual(y,"duagbcn")

	def test_raildecrypt_1(self):
		y = raildecrypt("szmpycbzueddbbxozl",7)
		self.assertEqual(y,"smcudxlobebpzyzdbz")

	def test_raildecrypt_2(self):
		y = raildecrypt("ykhiqkvigqanprp",62)
		self.assertEqual(y,"ykhiqkvigqanprp")

