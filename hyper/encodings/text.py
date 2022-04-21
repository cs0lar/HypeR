import time, sys

import numpy as np

from hyper.hypervector import Permutation, BipolarHypervector

class Alphabet():

	ALPHABET = [ chr( i ) for i in range( 32, 65 ) ] + [ chr( i ) for i in range( 91, 127 ) ]

	def __init__( self, d, rng, alphabet=ALPHABET ):

		self._alphabet = { k:BipolarHypervector.new( d, rng ) for k in alphabet }

	def __getitem__( self, k ):

		return BipolarHypervector( hv=self._alphabet[ k ]._hv )

	def __len__( self ):

		return len( self._alphabet )


class TextEncoding():
	"""
	Implementation of https://link.springer.com/chapter/10.1007/978-3-319-52289-0_21
	"""
	def __init__( self, alphabet, perm, n ):
		
		if not alphabet or not isinstance( alphabet, Alphabet ):

			raise ValueError( 'alphabet must be an instance of hyper.encodings.text.Alphabet' )

		if not perm or not isinstance( perm, Permutation ):

			raise ValueError( 'perm must be an instance of hyper.hypervector.Permutation' )
		
		self._alphabet = alphabet
		self._perm = perm 
		self._n = n 

	def encodengram( self, ngram ):

		def mul( A, B ):
			A._hv = BipolarHypervector.mul( A, B )._hv

		try:

			l = len( ngram )

			A = self._alphabet[ ngram[ 0 ] ].permute( self._perm, c=( l-1 ) )

			[ mul( A, self._alphabet[ ngram[ i ] ].permute( self._perm, c=( l-i-1 ) ) ) for i in range( 1, l ) ]

			return A 

		except KeyError as e:

			raise ValueError( f'key {e.args[ 0 ]} not in alphabet' )

		except IndexError as e:

			raise ValueError( f'encoding ngram={ngram} requires more precomputed permutations that are available' )

	def encode( self, entity ):

		if not isinstance( entity, str ):

			raise ValueError( 'entity must be a string' )

		size = len( entity )

		if size < self._n:

			raise ValueError( 'size of entity is less than size of ngram' )

		text = f' {entity.lower().strip( " " )} '

		start = 0
		step = self._n

		# the running vector sum
		A = self.encodengram( text[ 0:step ] )

		# the last block we computed
		V = BipolarHypervector( hv=A._hv )
		
		[ A.add( self.encodeblock( text[ i ], text[ i+step ], step, V ) ) for i in range( len( text )-step  ) ]

		return A.threshold()

	def encodeblock( self, v, w, n, A ):

		V = self._alphabet[ v ].permute( self._perm, c=( n-1 ) )
		W = self._alphabet[ w ]

		V = BipolarHypervector.mul( V, A )
		V.permute( self._perm, c=1 )

		A._hv = BipolarHypervector.mul( V, W )._hv 

		return A




if __name__ == '__main__':
	
	d = 10000
	n = 5
	rng = np.random.default_rng(  )

	perm = Permutation( rng=rng )

	# an n-gram requires n-1 permutations
	perm.generate( d, n - 1 )
	
	alphabet = Alphabet( d=d, rng=rng )

	encoding = TextEncoding( alphabet, perm, n )

	def process( string, alphabet ):

		return ''.join( c for c in string if c in alphabet )

	with open( sys.argv[ 1 ], 'r' ) as f:
		
		lines = [ process( line.split( '\t' )[ 1 ].lower(), Alphabet.ALPHABET ) for line in f ]
		
	text = ' '.join( lines )[ :100000 ]
	start = time.time()
	textVector = encoding.encode( text ) 
	print( f'finished in {time.time() - start} seconds' )
