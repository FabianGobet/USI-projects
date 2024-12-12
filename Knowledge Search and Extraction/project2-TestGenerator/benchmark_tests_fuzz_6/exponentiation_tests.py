from unittest import TestCase
from benchmark.exponentiation import exponentiation

class Test_exponentiation(TestCase):
	def test_exponentiation_1(self):
		y = exponentiation(-88,31)
		self.assertEqual(y,-1900915608262144564536756271499648295949109556758117783437312)

	def test_exponentiation_2(self):
		y = exponentiation(92,57)
		self.assertEqual(y,8627920975900039361706390180117528652750832376655562647487019273649128000793939961011323622552442303651199844352)

