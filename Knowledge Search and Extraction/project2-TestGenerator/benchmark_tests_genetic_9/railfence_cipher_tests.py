from unittest import TestCase
from benchmark.railfence_cipher import railencrypt, raildecrypt

class Test_railfence_cipher(TestCase):
	def test_railencrypt_1(self):
		y = railencrypt("nqpo",20)
		self.assertEqual(y,"nqpo")

	def test_railencrypt_2(self):
		y = railencrypt("zyvwti",59)
		self.assertEqual(y,"zyvwti")

	def test_railencrypt_3(self):
		y = railencrypt("guvhhyovwuadyewapiom",30)
		self.assertEqual(y,"guvhhyovwuadyewapiom")

	def test_railencrypt_4(self):
		y = railencrypt("yovxtucq",3)
		self.assertEqual(y,"ytoxuqvc")

	def test_raildecrypt_1(self):
		y = raildecrypt("bj",3)
		self.assertEqual(y,"bj")

	def test_raildecrypt_2(self):
		y = raildecrypt("cxhupdqgzmcxadogfw",6)
		self.assertEqual(y,"chdzafdmquxpgcowgx")

	def test_raildecrypt_3(self):
		y = raildecrypt("aqqjgebwbwhh",83)
		self.assertEqual(y,"aqqjgebwbwhh")

	def test_raildecrypt_4(self):
		y = raildecrypt("cqmdgshjytbinze",35)
		self.assertEqual(y,"cqmdgshjytbinze")

	def test_raildecrypt_5(self):
		y = raildecrypt("nwbljzzxzte",17)
		self.assertEqual(y,"nwbljzzxzte")

	def test_raildecrypt_6(self):
		y = raildecrypt("vvkp",4)
		self.assertEqual(y,"vvkp")

