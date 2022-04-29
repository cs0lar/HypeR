import unittest

import numpy as np 

from hyper.hypervector import BipolarHypervector
from hyper.encodings.text import Alphabet, TextEncoding

class TestTextEncoding( unittest.TestCase ):

	def setUp( self ):

		self.alphabet = Alphabet( d=10000, rng=np.random.default_rng() )

	def testEncodengram( self ):

		n = 4
		encoding = TextEncoding( self.alphabet, n )
		ngram = 'grab'

		G = self.alphabet[ 'g' ]
		R = self.alphabet[ 'r' ]
		A = self.alphabet[ 'a' ]
		B = self.alphabet[ 'b' ]

		expected = BipolarHypervector.mul( G.permute( c=( n-1 ) ), BipolarHypervector.mul( R.permute( c=( n-2 ) ), BipolarHypervector.mul( A.permute( n-3 ), B ) ) )

		actual = encoding.encodengram( ngram )

		self.assertEqual( actual, expected )



	def testEncodeblock( self ):

		n = 4
		encoding = TextEncoding( self.alphabet, n )
		prevngram = 'grab'
		nextngram = 'rabs'

		G = self.alphabet[ 'g' ]
		R = self.alphabet[ 'r' ]
		A = self.alphabet[ 'a' ]
		B = self.alphabet[ 'b' ]
		

		GRAB = BipolarHypervector.mul( G.permute( c=( n-1 ) ), BipolarHypervector.mul( R.permute( c=( n-2 ) ), BipolarHypervector.mul( A.permute( n-3 ), B ) ) )

		R = self.alphabet[ 'r' ]
		A = self.alphabet[ 'a' ]
		B = self.alphabet[ 'b' ]
		S = self.alphabet[ 's' ]

		expected = BipolarHypervector.mul( R.permute( c=( n-1 ) ), BipolarHypervector.mul( A.permute( c=( n-2 ) ), BipolarHypervector.mul( B.permute( n-3 ), S ) ) )

		actual = encoding.encodeblock( 'g', 's', GRAB )

		self.assertEqual( actual, expected )


	def testEncode( self ):

		n = 4
		text = 'sleep nezuko'

		encoding = TextEncoding( self.alphabet, n )
		
		ngrams = [ 'slee', 'leep', 'eep ', 'ep n', 'p ne', ' nez', 'nezu', 'ezuk', 'zuko' ]

		vectors = [ encoding.encodengram( ngram ) for ngram in ngrams ]

		expected = BipolarHypervector.sum( *vectors ).threshold()

		actual = encoding.encode( text )

		self.assertEqual( actual, expected )
