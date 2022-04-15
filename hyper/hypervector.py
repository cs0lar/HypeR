import math

import numpy as np

from bitarray import bitarray
from bitarray.util import count_xor

class Hypervector:
	"""
	Base class for all hyper vector types ( e.g. binary, bipolar, integer )

	Attributes
	----------

	_hv: mixed
		The hypevector in its underlying data type implemented by inheriting classes. 

	"""
	def __init__( self, hv=None ):

		self._hv = hv



class BinaryHypervector( Hypervector ):
	"""
	Implementation of a binary hypervector. It uses a bitarray as the underlying
	hyper vector data type.

	Attributes
	----------

	_hv: bitarray
		The actual underlying hyper vector in its bitarray form.


	Methods
	-------

	new( d, rng )
		Static method that creates a new random Binary Hypervector of size d.

	majority( bits )
		Static method that computes the majority rule for a given bitarray.

	sum( *args )
		Static method that computes the binary hypervector sum of a variable
		number of BinaryHypervectors.

	permute( c )
		It permutes the underlying bitarray using a right circular shift
		of count c.

	hamming( X, Y )
		Static method that computes the Hamming distance between two binary hypervectors.

	mul( A, B )
		Static method that performs the binding operation which for binary hypervectors 
		takes for form of the XOR operator.

	
	"""
	def __init__( self, hv ):

		super().__init__( hv )

		if self._hv and isinstance( self._hv, str ):

			self._hv = bitarray( self._hv )

	@staticmethod
	def new( d, rng ):
		"""
		It creates a new random BinaryHypervector using a bitarray
		as underlying data type.
	
		Parameters
		----------

		d : int
			The dimensionality of the hyper vector e.g. 10,000

		rng : numpy.random.Generator
			The random number generator to use for creating the vector.

		Returns
		-------

		BynaryHypervector
			A random binary hypervector.

		"""
		hv = bitarray( rng.integers( 2, size=d ).tolist() )

		return BinaryHypervector( hv=hv )


	def __len__( self ):
		"""
		It overrides the `__len__` magic function so that 
		the `len` function can be invoked on a BinaryHypervector.

		Returns
		-------
		
		int
			The length of the underlying bitarray.

		"""
		return len( self._hv )


	def __getitem__( self, i ):
		"""

		It overrides the `__getitem__` magic function so that
		BinaryHypervector object can be indexed like a list.

		Parameters
		----------

		i : int
			The index into the underlying bitarray to return a value for.


		Returns
		--------

		int
			Either 1 or 0 depending on the value of the underlying bitarray at index i.

		"""

		return self._hv[ i ]


	def permute( self, c=1, inverse=False ):
		"""
		It computes a permutation of the underlying bitarray based on right circular shift of 
		count `c`. Refer to: https://en.wikipedia.org/wiki/Circular_shift. If `inverse` is `True`
		it computes left circular shift.

		Parameters
		----------

		c : int
			The amount of shift to apply.

		inverse : boolean
			Whether to perform a left or right circular shift. Default is `False`, 
			i.e. right circular shift.


		Returns
		-------

		BinaryHypervector
			The permuted hypervector. The permutation occurs in-place.

		"""
		l = len( self._hv )

		if inverse:

			self._hv = self._hv << c | self._hv >> ( l - c )

		else:

			self._hv = self._hv >> c | self._hv << ( l - c )

		return self


	@staticmethod
	def majority( bits ):
		"""
		It computes the majority function of a string of bits. Refer to: https://en.wikipedia.org/wiki/Majority_function

		Parameters
		----------

		bits : bitarray
			A bitarray to compute the majority function for.

		Returns
		-------

		int
			0 or 1 depending on the computation.

		"""
		c = bits.count()

		return math.floor( .5 + ( c - .5 ) / len( bits ) )


	@staticmethod
	def sum( *args ):
		"""
		It computes the thresholded and binarised sum of an arbitrary number of 
		BinaryHypervectors by applying the majority function.

		Parameters
		----------
		
		*args : BinaryHypervectors
			A variable number of BinaryHypervectors.

		Returns
		-------

		BinaryHypervector
			A new BinaryHypervector representing the thresholded, binarised 
			sum of the input BinaryHypervectors.

		"""
		vecs = [ arg for arg in args if isinstance( arg, BinaryHypervector ) ]

		l = len( vecs[ 0 ] )

		if ( len( vecs ) % 2 ) == 0:
			# to avoid 0 bias when the number of hypervectors to sum 
			# is even, we add a random hypervector to make the number odd.
			vecs.append( BinaryHypervector.new( l ) )

		hv = bitarray( [ BinaryHypervector.majority( bitarray( [ vec[ i ] for vec in vecs ] ) ) for i in range( l ) ] )

		return BinaryHypervector( hv=hv )


	@staticmethod
	def mul( A, B ):
		"""
		It performs a binding operation, binding A and B together for form X = A*B.
		For binary hypervectors the binding operation is the XOR operation.

		Parameters
		----------

		A : BinaryHypervector
			A binary hypervector

		B : BinaryHypervector
			A binary hypervector

		Returns
		-------

		BinaryHypervector
			XOR( A, B )

		"""

		xor = A._hv ^ B._hv

		return BinaryHypervector( hv=xor )
		

	@staticmethod
	def hamming( X, Y ):
		"""
		It computes the Hamming distance between binary hypervectors X and Y.

		Parameters
		----------

		X : BinaryHypervector
			A binary hypervector

		Y : BinaryHypervector
			A binary hypervector

		Returns
		-------

		float
			The Hamming distance between X and Y. If `hamming( X, Y )` = .5 then
			X and Y are orthogonal or dissimilar. X and Y are diametrically opposite
			when `hamming( X, Y )` = 1.

		"""
		d = len( X )

		return count_xor( X._hv, Y._hv ) / d


	def __str__( self ):
		"""
		It overrides the magic `__str__` function in order to pretty print 
		the underlying bitarray.

		Returns
		-------

		str
			The content of the underlying bitarray in string format.

		"""
		return str( self._hv.tolist() )

	def __eq__( self, other ):
		"""
		It overrides the magic `__eq__` function in order to support
		comparison of two binary hypervectors.

		Parameters
		----------

		other : BinaryHypervector
			A binary hypervector to compare with self.

		Returns
		-------

		boolean
			`True` if the underlying bitarrays are equal, `False` otherwise.
		"""
		if isinstance( other, BinaryHypervector ):
			return self._hv == other._hv

		raise False



