from unittest import TestCase
from benchmark.railfence_cipher import railencrypt, raildecrypt

class Test_railfence_cipher(TestCase):
	def test_railencrypt_1(self):
		y = railencrypt("cn",46)
		self.assertEqual(y,"cn")

	def test_railencrypt_2(self):
		y = railencrypt("miiufvx",11)
		self.assertEqual(y,"miiufvx")

	def test_railencrypt_3(self):
		y = railencrypt("bahkpfirojjvfksyaqzl",5)
		self.assertEqual(y,"boaarjyqhijszkfvklpf")

	def test_railencrypt_4(self):
		y = railencrypt("atiloxozbr",7)
		self.assertEqual(y,"atilrobxzo")

	def test_railencrypt_5(self):
		y = railencrypt("yhemxidob",72)
		self.assertEqual(y,"yhemxidob")

	def test_railencrypt_6(self):
		y = railencrypt("ishdhjjrxytsmbmfmno",39)
		self.assertEqual(y,"ishdhjjrxytsmbmfmno")

	def test_raildecrypt_1(self):
		y = raildecrypt("lasleaknfytgq",3)
		self.assertEqual(y,"letaakgnsfqyl")

	def test_raildecrypt_2(self):
		y = raildecrypt("sdlvppbpe",11)
		self.assertEqual(y,"sdlvppbpe")

	def test_raildecrypt_3(self):
		y = raildecrypt("upqauthxpjqiq",91)
		self.assertEqual(y,"upqauthxpjqiq")

