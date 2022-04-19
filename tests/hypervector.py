import unittest

import numpy as np

from bitarray import bitarray

from tqdm import tqdm

from hyper.hypervector import Permutation, BinaryHypervector, BipolarHypervector

import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.spatial import distance


class TestBinaryHipervector( unittest.TestCase ):


	def testGetindex( self ):

		a = bitarray( '0000110011' )

		A = BinaryHypervector( hv=a )

		for i in range( len( A ) ):

			self.assertEqual( a[ i ], A[ i ] )


	def testHamming( self ):

		A = BinaryHypervector( hv='0000110011' )
		B = BinaryHypervector( hv='1011000101' )

		expected = .7
		actual = BinaryHypervector.hamming( A, B )

		self.assertAlmostEqual( actual, expected )


	def testPermute( self ):

		D = BinaryHypervector( hv='0000110011' )

		expected = BinaryHypervector( hv='0110000110' )
		
		actual = D.permute( c=3 )

		self.assertEqual( actual, expected )
		self.assertEqual( D, expected )

		expectedInv = BinaryHypervector( hv='0000110011' ) 
		actual = D.permute( c=3, inverse=True )

		self.assertEqual( actual, expectedInv )


	def testSum( self ):

		A = BinaryHypervector( hv='0000110011' )
		B = BinaryHypervector( hv='1011000101' )
		C = BinaryHypervector( hv='0010101101' )
	
		expected = BinaryHypervector( hv='0010100101' )

		actual = BinaryHypervector.sum( A, B, C )

		self.assertEqual( actual, expected )


	def testMul( self ):

		A = BinaryHypervector( hv='0000110011' )
		B = BinaryHypervector( hv='1011000101' )

		expected = BinaryHypervector( hv='1011110110' )
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
		plt.savefig( 'testbinarynew.png' )



class TestBipolaripervector( unittest.TestCase ):

	def testNew( self ):
		"""
		It must be demonstrated that as the dimensionality of the
		hypervectors increases so does the probability that the 
		cosine distance between two random hyper vectors approaches 0, i.e.
		any two random vectors are most likely orthogonal. 
		"""

		dims = [ 100, 500, 2500, 10000 ]
		colors = [ 'r', 'b', 'g', 'c', 'm' ]
		labels = [ f'd={v}' for v in dims ]
		iters = 1500

		rng = np.random.default_rng()

		def sample():

			X = BipolarHypervector.new( d, rng )
			Y = BipolarHypervector.new( d, rng )

			return BipolarHypervector.cosine( X, Y )

		for i, d in enumerate( dims ): 
			samples = [ sample() for i in tqdm( range( iters ) ) ]

			
			mean = np.mean( samples )
			std = np.std( samples )

			x = np.linspace( mean - 3. * std, mean + 3. * std, 100 )

			plt.plot( x, norm.pdf( x, mean, std ), colors[ i ] )

		plt.xlabel( 'cosine similarity' )
		plt.ylabel( 'Probability (%)' )
		plt.legend( labels )
		plt.ylim( 0, 100 )
		plt.xticks( np.linspace( -.5, .5, 5 ) )
		plt.savefig( 'testbipolarnew.png' )


	def testPermute( self ):

		numPerms = 4
		dims = 10000

		rng = np.random.default_rng()

		perms = Permutation( rng=rng )

		perms.generate( dims, numPerms )

		basePerm = perms[ 0 ]

		hv = BipolarHypervector.new( dims, rng )._hv
		v = np.array( hv )

		for i in range( numPerms ):
			
			w = BipolarHypervector( hv=hv )

			self.assertEqual( w.permute( perms, c=i ), BipolarHypervector( hv=v ) )

			v = v[ basePerm ]

	def testCosine( self ):

		A = BipolarHypervector( hv=np.array( [ -1, -1, -1, -1,  1,  1, -1, -1,  1, 1 ] ) )
		B = BipolarHypervector( hv=np.array( [  1, -1,  1,  1, -1, -1, -1,  1, -1, 1 ] ) )

		self.assertAlmostEqual( BipolarHypervector.cosine( A, A ), 1. )

		expected = 1. - distance.cosine( A._hv, B._hv )
		actual = BipolarHypervector.cosine( A, B )

		self.assertAlmostEqual( actual, expected )


	def testSum( self ):

		A = BipolarHypervector( hv=np.array( [ -1, -1, -1, -1,  1,  1, -1, -1,  1, 1 ] ) )
		B = BipolarHypervector( hv=np.array( [  1, -1,  1,  1, -1, -1, -1,  1, -1, 1 ] ) )
		C = BipolarHypervector( hv=np.array( [ -1, -1,  1, -1,  1, -1,  1,  1, -1, 1 ] ) )
	
		expected = BipolarHypervector( hv=np.array( [ -1, -1, 1, -1, 1, -1, -1, 1, -1, 1 ] ) ) 

		actual = BipolarHypervector.sum( A, B, C )

		self.assertEqual( actual, expected )
		
		# The sum (and the mean) of random hypervectors has the
		# property that it is similar to each of the
		# hypervectors being added together. 

		dims = 10000

		rng = np.random.default_rng()
		
		scores = []

		A, B, C = [ BipolarHypervector.new( dims, rng ) for i in range( 3 ) ] 

		Z = BipolarHypervector.sum( A, B, C )

		for V in [ A, B, C ]:

			scores.append( BipolarHypervector.cosine( Z, V ) )
		
		self.assertTrue( np.average( scores ) > .4 )

	def testMul( self ):

		A = BipolarHypervector( hv=np.array( [ -1, -1, -1, -1,  1,  1, -1, -1,  1, 1 ] ) )
		B = BipolarHypervector( hv=np.array( [  1, -1,  1,  1, -1, -1, -1,  1, -1, 1 ] ) )
		
		expected = A._hv * B._hv
		actual = BipolarHypervector.mul( A, B )

		self.assertEqual

		

