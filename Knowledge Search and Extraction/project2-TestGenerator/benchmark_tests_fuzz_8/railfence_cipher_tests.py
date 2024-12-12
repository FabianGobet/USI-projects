from unittest import TestCase
from benchmark.railfence_cipher import railencrypt, raildecrypt

class Test_railfence_cipher(TestCase):
	def test_railencrypt_1(self):
		y = railencrypt("esozvyjskeyx",12)
		self.assertEqual(y,"esozvyjskeyx")

	def test_railencrypt_2(self):
		y = railencrypt("yfn",58)
		self.assertEqual(y,"yfn")

	def test_railencrypt_3(self):
		y = railencrypt("lzhjoycgpg",38)
		self.assertEqual(y,"lzhjoycgpg")

	def test_railencrypt_4(self):
		y = railencrypt("nb",16)
		self.assertEqual(y,"nb")

	def test_railencrypt_5(self):
		y = railencrypt("bqzohq",3)
		self.assertEqual(y,"bhqoqz")

	def test_raildecrypt_1(self):
		y = raildecrypt("guwpchntp",7)
		self.assertEqual(y,"guwpcnpth")

	def test_raildecrypt_2(self):
		y = raildecrypt("senqfy",64)
		self.assertEqual(y,"senqfy")

	def test_raildecrypt_3(self):
		y = raildecrypt("qqqkgfo",7)
		self.assertEqual(y,"qqqkgfo")

	def test_raildecrypt_4(self):
		y = raildecrypt("oetubzsrlpmdkhx",3)
		self.assertEqual(y,"obdzeskrtlhpumx")

	def test_raildecrypt_5(self):
		y = raildecrypt("wacqomdsvluoryieycbq",46)
		self.assertEqual(y,"wacqomdsvluoryieycbq")

	def test_raildecrypt_6(self):
		y = raildecrypt("maxxknfdbgq",95)
		self.assertEqual(y,"maxxknfdbgq")

