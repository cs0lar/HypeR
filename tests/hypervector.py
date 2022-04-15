import unittest

import numpy as np

from bitarray import bitarray

from tqdm import tqdm

from hyper.hypervector import BinaryHypervector

import matplotlib.pyplot as plt

from scipy.stats import norm


class TestBinaryHipervector( unittest.TestCase ):


	def testGetindex( self ):

		a = bitarray( '0000110011' )

		A = BinaryHypervector( hv=a )

		for i in range( len( A ) ):

			self.assertEqual( a[ i ], A[ i ] )


	def testHamming( self ):

		A = BinaryHypervector( hv=bitarray( '0000110011' ) )
		B = BinaryHypervector( hv=bitarray( '1011000101' ) )

		expected = .7
		actual = BinaryHypervector.hamming( A, B )

		self.assertAlmostEqual( actual, expected )


	def testPermute( self ):

		D = BinaryHypervector( hv=bitarray( '0000110011' ) )

		expected = BinaryHypervector( hv=bitarray( '0110000110' ) )
		
		actual = D.permute( c=3 )

		self.assertEqual( actual, expected )
		self.assertEqual( D, expected )

		expectedInv = BinaryHypervector( hv=bitarray( '0000110011' ) ) 
		actual = D.permute( c=3, inverse=True )

		self.assertEqual( actual, expectedInv )


	def testSum( self ):

		A = BinaryHypervector( hv=bitarray( '0000110011' ) )
		B = BinaryHypervector( hv=bitarray( '1011000101' ) )
		C = BinaryHypervector( hv=bitarray( '0010101101' ) )
	
		expected = BinaryHypervector( hv=bitarray( '0010100101' ) )

		actual = BinaryHypervector.sum( A, B, C )

		self.assertEqual( actual, expected )


	def testMul( self ):

		A = BinaryHypervector( hv=bitarray( '0000110011' ) )
		B = BinaryHypervector( hv=bitarray( '1011000101' ) )

		expected = BinaryHypervector( hv=bitarray( '1011110110' ) )
		actual = BinaryHypervector.mul( A, B )

		self.assertEqual( actual, expected )


	def testNew( self ):
		"""
		It must be demonstrated that as the dimensionality of the
		hypervectors increases so does the probability that the 
		hamming distance between two random hyper vectors is 0.5, i.e.
		any two random vectors are most likely orthogonal. 
		"""
		
		dims = [ 100, 500, 2500, 10000 ]
		colors = [ 'r', 'b', 'g', 'c', 'm' ]
		labels = [ f'd={v}' for v in dims ]
		iters = 15000

		rng = np.random.default_rng()

		def sample():

			X = BinaryHypervector.new( d, rng )
			Y = BinaryHypervector.new( d, rng )

			return BinaryHypervector.hamming( X, Y )

		for i, d in enumerate( dims ): 
			samples = [ sample() for i in tqdm( range( iters ) ) ]

			
			mean = np.mean( samples )
			std = np.std( samples )

			x = np.linspace( mean - 3. * std, mean + 3. * std, 100 )

			plt.plot( x, norm.pdf( x, mean, std ), colors[ i ] )

		plt.xlabel( 'Normalised Hamming distance' )
		plt.ylabel( 'Probability (%)' )
		plt.legend( labels )
		plt.ylim( 0, 100 )
		plt.xticks( np.linspace( 0, 1., 5 ) )
		plt.savefig( 'testnew.png' )



