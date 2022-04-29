import numpy as np

from tqdm import tqdm

from hyper.hypervector import BipolarHypervector


class Alphabet():
	"""
	This class allows to represent a collection of symbols ( e.g. letters, words, etc )
	as random hypervectors. By default it will generate random bipolar hypervectors
	for the 26 ASCII characters corresponding to the lower-case english alphabet letters
	plus the space character. These "atomic" hypervectors can then be combined together to 
	encode sets, sequences and bindings of the alphabet symbols according to the
	basic operations on hypervectors.
	Once an instance is constructed it can be used like a dictionary, retrieving the 
	hypervector for a particular symbol by using that symbols as an indexing key.

	Attributes
	----------

	ALPHABET : array
		A static list of the 26 ASCII english alphabet characters plus the space character.

	"""
	ALPHABET = [ chr( 32 ) ] + [ chr( i ) for i in range( 97, 123 ) ]

	def __init__( self, d, rng, alphabet=ALPHABET ):

		self._alphabet = { k:BipolarHypervector.new( d, rng ) for k in alphabet }

	def __getitem__( self, k ):

		return BipolarHypervector( hv=self._alphabet[ k ]._hv )

	def __len__( self ):

		return len( self._alphabet )


class TextEncoding():
	"""
	It implements a Text hypervector as described in
	https://redwood.berkeley.edu/wp-content/uploads/2020/08/JoshiEtAl-QI2016-language-geometry-copy.pdf.
	A piece of text is encoded as the hypervector sum of its ngrams. An ngram 
	is a sequence of n consecutive characters extracted from the text and itself encoded as 
	an hypervector using multiplication and permutation.

	Attributes
	----------

	_alphabet : Alphabet
		An instance of `Alphabet` containing the hypervector representation of the appropriate language alphabet's symbols.

	_n :  int
		The desired ngram size.

	Methods
	---------

	encodengram( ngram ):
		It encodes a single ngram ( a sequence of n consecutive characters from the text ) 

	encodeblock( v, w, n, A ):
		It encodes a single ngram given the previous ngram and the new character from this
		block to be added.

	encode( text ):
		It encodes a all textual string using ngram and text hypervector encoding.	

	"""
	def __init__( self, alphabet, n ):
		
		if not alphabet or not isinstance( alphabet, Alphabet ):

			raise ValueError( 'alphabet must be an instance of hyper.encodings.text.Alphabet' )
		
		self._alphabet = alphabet
		self._n = n 

	def encodengram( self, ngram ):
		"""
		It encodes a single ngram ( a sequence of n consecutive characters from the text ) using 
		hypevector sequence encoding. Let \rho be the permutation operation, then 
		the ngram `GRAB` is encoded by performing the following hypervector computation:

			\rho( \rho ( \rho( G ) ) ) * \rho( \rho( R ) ) * \rho( A ) * B

		where G, R, A and B are hypervector representations of the ASCII characters 
		`g`, `r`, `a` and `b` respectively.
	
		Parameters
		----------

		ngram : string
			A string of `n` characters.

		Returns
		-------

		BipolarHypervector
			A `BipolarHypervector` encoding the ngram sequence.

		"""
		def mul( A, B ):
			A._hv = BipolarHypervector.mul( A, B )._hv

		try:

			n = len( ngram )

			A = self._alphabet[ ngram[ 0 ] ].permute( c=( n-1 ) )

			[ mul( A, self._alphabet[ ngram[ i ] ].permute( c=( n-i-1 ) ) ) for i in range( 1, n ) ]

			return A 

		except KeyError as e:

			raise ValueError( f'key {e.args[ 0 ]} not in alphabet' )


	def encode( self, text, progressbar=True ):
		"""
		It encodes a textual string as the sum of all its ngram sequences. Let the text be
		'with great powers' and n=4 be the ngram size. The collection of ngrams is
		[ 'with' , 'ith ', 'th g', 'h gr', ' gre', 'grea', 'reat', 'eat ', 'at p', 't po', ' pow', 'powe', 'ower', 'wers' ]
		Each of the 14 ngrams is encoded as a hypervector using permutations and multiplications ( see `encodengram` )
		and these hypervectors are summed together to obtain a single hypervector representing the entire text.

		Parameters
		----------

		text: string
			The text to be encoded.

		Returns
		-------

		BipolarHypervector
			A `BipolarHypervector` encoding the text.

		"""
		if not isinstance( text, str ):

			raise ValueError( 'text must be a string' )

		size = len( text )

		if size < self._n:

			raise ValueError( 'size of text is less than size of ngram' )

		text = text.lower().strip( " " )

		start = 0
		step = self._n

		# the running vector sum
		A = self.encodengram( text[ 0:step ] )

		# the last block we computed
		V = BipolarHypervector( hv=A._hv )
		
		iterator = tqdm( range( len( text )-step  ) ) if progressbar else range( len( text )-step  )

		[ A.add( self.encodeblock( text[ i ], text[ i+step ], V ) ) for i in iterator ]

		return A.threshold()

	def encodeblock( self, v, w, A ):
		"""
		It implements a fast ngram encoding algorithm that can be used
		to encode the m+1 th ngram given the m th ngram hypervector ( refer to section 2.2 of paper ).
		Note that this function actually modifies `A` to be the m+1 th ngram encoding 
		and returns it.

		Parameters
		----------

		v - string
			The first character of the m th ngram.

		w - string
			The last character for the m+1 th ngram.

		A - BipolarHypervector
			The hypervector representation of the m th ngram.

		Returns
		--------
			A `BipolarHypervector` encoding the m+1 th ngram.

		"""
		V = self._alphabet[ v ].permute( c=( self._n-1 ) )
		W = self._alphabet[ w ]

		V = BipolarHypervector.mul( V, A )
		V.permute( c=1 )

		A._hv = BipolarHypervector.mul( V, W )._hv 
		return A


	



