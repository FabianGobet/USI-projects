from unittest import TestCase
from benchmark.railfence_cipher import railencrypt, raildecrypt

class Test_railfence_cipher(TestCase):
	def test_railencrypt_1(self):
		y = railencrypt("jlpgehtacefostq",34)
		self.assertEqual(y,"jlpgehtacefostq")

	def test_railencrypt_2(self):
		y = railencrypt("vzjynvcxampodqcscum",5)
		self.assertEqual(y,"vaczxmsujcpcmyvoqnd")

	def test_railencrypt_3(self):
		y = railencrypt("argyczbjnwn",22)
		self.assertEqual(y,"argyczbjnwn")

	def test_raildecrypt_1(self):
		y = raildecrypt("tjdrswbafzxgxfiwoto",29)
		self.assertEqual(y,"tjdrswbafzxgxfiwoto")

	def test_raildecrypt_2(self):
		y = raildecrypt("zgsvnz",79)
		self.assertEqual(y,"zgsvnz")

	def test_raildecrypt_3(self):
		y = raildecrypt("jleitooqp",5)
		self.assertEqual(y,"jetopqoil")

