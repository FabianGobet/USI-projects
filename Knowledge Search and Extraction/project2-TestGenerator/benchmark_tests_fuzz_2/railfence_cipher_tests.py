from unittest import TestCase
from benchmark.railfence_cipher import railencrypt, raildecrypt

class Test_railfence_cipher(TestCase):
	def test_railencrypt_1(self):
		y = railencrypt("ntokbghbnvvgspqjumqm",9)
		self.assertEqual(y,"nutjmoqqkpmbsgghvbvn")

	def test_raildecrypt_1(self):
		y = raildecrypt("yzdq",3)
		self.assertEqual(y,"yzqd")

	def test_raildecrypt_2(self):
		y = raildecrypt("czadazfs",3)
		self.assertEqual(y,"cafdzasz")

	def test_raildecrypt_3(self):
		y = raildecrypt("qumaqtosz",42)
		self.assertEqual(y,"qumaqtosz")

	def test_raildecrypt_4(self):
		y = raildecrypt("hqmovbcxbykhozgn",58)
		self.assertEqual(y,"hqmovbcxbykhozgn")

	def test_raildecrypt_5(self):
		y = raildecrypt("soaudncvjgcqa",14)
		self.assertEqual(y,"soaudncvjgcqa")

	def test_raildecrypt_6(self):
		y = raildecrypt("epckprw",18)
		self.assertEqual(y,"epckprw")

	def test_raildecrypt_7(self):
		y = raildecrypt("bvsnnsdsblepjtyqi",10)
		self.assertEqual(y,"bvsndbejyiqtplssn")

	def test_raildecrypt_8(self):
		y = raildecrypt("bqaudncvjgcqhuoc",11)
		self.assertEqual(y,"bqaudnvgqucohcjc")

