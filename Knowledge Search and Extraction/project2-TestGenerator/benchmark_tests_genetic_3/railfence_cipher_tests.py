from unittest import TestCase
from benchmark.railfence_cipher import railencrypt, raildecrypt

class Test_railfence_cipher(TestCase):
	def test_railencrypt_1(self):
		y = railencrypt("ndptjmyl",9)
		self.assertEqual(y,"ndptjmyl")

	def test_railencrypt_2(self):
		y = railencrypt("tjlitbine",25)
		self.assertEqual(y,"tjlitbine")

	def test_railencrypt_3(self):
		y = railencrypt("qbecqyayugc",41)
		self.assertEqual(y,"qbecqyayugc")

	def test_railencrypt_4(self):
		y = railencrypt("wzvujrqiu",21)
		self.assertEqual(y,"wzvujrqiu")

	def test_railencrypt_5(self):
		y = railencrypt("snrhuvgesoqjnl",4)
		self.assertEqual(y,"sgnnvejlrusqho")

	def test_raildecrypt_1(self):
		y = raildecrypt("ckevhlmad",24)
		self.assertEqual(y,"ckevhlmad")

	def test_raildecrypt_2(self):
		y = raildecrypt("vxnukpvozbgbvensgt",60)
		self.assertEqual(y,"vxnukpvozbgbvensgt")

	def test_raildecrypt_3(self):
		y = raildecrypt("aqurngzatkosljjy",6)
		self.assertEqual(y,"augtsjlkzrqnaojy")

	def test_raildecrypt_4(self):
		y = raildecrypt("vtmqraqevulule",20)
		self.assertEqual(y,"vtmqraqevulule")

	def test_raildecrypt_5(self):
		y = raildecrypt("vfwadiqoycyjdkunxi",52)
		self.assertEqual(y,"vfwadiqoycyjdkunxi")

	def test_raildecrypt_6(self):
		y = raildecrypt("nyrn",25)
		self.assertEqual(y,"nyrn")

	def test_raildecrypt_7(self):
		y = raildecrypt("jmgjapbwevho",7)
		self.assertEqual(y,"jmjpwvohebag")

