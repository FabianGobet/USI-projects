from unittest import TestCase
from benchmark.railfence_cipher import railencrypt, raildecrypt

class Test_railfence_cipher(TestCase):
	def test_railencrypt_1(self):
		y = railencrypt("offtfnlceoolzvc",75)
		self.assertEqual(y,"offtfnlceoolzvc")

	def test_railencrypt_2(self):
		y = railencrypt("zugq",17)
		self.assertEqual(y,"zugq")

	def test_railencrypt_3(self):
		y = railencrypt("gramazmzuoiylv",7)
		self.assertEqual(y,"glryvaimoauzzm")

	def test_railencrypt_4(self):
		y = railencrypt("qykltb",4)
		self.assertEqual(y,"qybktl")

	def test_raildecrypt_1(self):
		y = raildecrypt("yymayzuhwlcgjcyelfjh",2)
		self.assertEqual(y,"ycygmjacyyzeulhfwjlh")

	def test_raildecrypt_2(self):
		y = raildecrypt("qbbrqntxdwlu",100)
		self.assertEqual(y,"qbbrqntxdwlu")

	def test_raildecrypt_3(self):
		y = raildecrypt("nttnmopqgka",4)
		self.assertEqual(y,"ntokpntmqag")

	def test_raildecrypt_4(self):
		y = raildecrypt("fddkgbwowzfjchfpry",33)
		self.assertEqual(y,"fddkgbwowzfjchfpry")

	def test_raildecrypt_5(self):
		y = raildecrypt("xynynmfvx",27)
		self.assertEqual(y,"xynynmfvx")

