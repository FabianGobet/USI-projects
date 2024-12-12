from unittest import TestCase
from benchmark.railfence_cipher import railencrypt, raildecrypt

class Test_railfence_cipher(TestCase):
	def test_railencrypt_1(self):
		y = railencrypt("odpwtnbynpsysoy",39)
		self.assertEqual(y,"odpwtnbynpsysoy")

	def test_railencrypt_2(self):
		y = railencrypt("fvoysjgrhird",61)
		self.assertEqual(y,"fvoysjgrhird")

	def test_railencrypt_3(self):
		y = railencrypt("eywackccpebmficwipoy",8)
		self.assertEqual(y,"ecyiwwfiampcbokeycpc")

	def test_raildecrypt_1(self):
		y = raildecrypt("faoyxivem",2)
		self.assertEqual(y,"fiavoeymx")

	def test_raildecrypt_2(self):
		y = raildecrypt("isqptqhcwb",8)
		self.assertEqual(y,"isqptqcbwh")

	def test_raildecrypt_3(self):
		y = raildecrypt("ickhbekillc",63)
		self.assertEqual(y,"ickhbekillc")

	def test_raildecrypt_4(self):
		y = raildecrypt("kimbfihnxlpvatjfbmi",20)
		self.assertEqual(y,"kimbfihnxlpvatjfbmi")

