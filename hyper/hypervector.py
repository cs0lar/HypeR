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

	@staticmethod
	def check( clazz, *args ):
		"""
		Static utility to check whether an arbitrary number of objects 
		are instances of the given Hypervector subclass.

		Parameters
		----------

		clazz : str
			The name of the subclass the input objects should be instances of.

		*args : Hypervectors
			An arbitrary number of objects to be checked.

		Returns
		-------

		boolean:
			True if all input objects are instances of the same, given 
			Hypervector subclass. False otherwise.

		"""

		isSubclass = [ issubclass( arg.__class__, Hypervector ) for arg in args ]
		isClass = [ ( arg.__class__.__name__ == clazz ) for arg in args ]
		
		return all( isSubclass + isClass )


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

		if Hypervector.check( 'BinaryHypervector', *args ):

			vecs = [ arg for arg in args if isinstance( arg, BinaryHypervector ) ]

			l = len( vecs[ 0 ] )

			if ( len( vecs ) % 2 ) == 0:
				# to avoid 0 bias when the number of hypervectors to sum 
				# is even, we add a random hypervector to make the number odd.
				vecs.append( BinaryHypervector.new( l, np.random.default_rng() ) )

			hv = bitarray( [ BinaryHypervector.majority( bitarray( [ vec[ i ] for vec in vecs ] ) ) for i in range( l ) ] )

			return BinaryHypervector( hv=hv )

		raise ValueError( 'All inputs must be BinaryHypervectors' )

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

		if Hypervector.check( 'BinaryHypervector', A, B ):
			
			xor = A._hv ^ B._hv

			return BinaryHypervector( hv=xor )
		
		raise ValueError( 'A and B must be BinaryHypervectors' )

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

		if Hypervector.check( 'BinaryHypervector', X, Y ):
			
			d = len( X )

			return count_xor( X._hv, Y._hv ) / d

		raise ValueError( 'X and Y must be BinaryHypervectors' )

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

	def __hash__( self ):
		"""
		It overrides the magic `__hash__` function in order to support 
		using `BinaryHypervector`s as keys in dictionaries.

		Returns
		-------

		int : the hash value of the object

		"""
		return hash( self._hv )

class BipolarHypervector( Hypervector ):
	"""
	Implementation of a bipolar hypervector X where the value of a component x_i is in {-1, 1}. 
	It uses a numpy array as the underlying hypervector data type.

	Attributes
	----------

	_hv: ndarray
		The actual underlying hypervector in its numpy array form.


	Methods
	-------

	new( d, rng )
		Static method that creates a new random Bipolar Hypervector of size d.

	sum( *args )
		Static method that computes the sum of a variable
		number of BipolarHypervectors.

	permute( rng )
		It permutes the underlying numpy array 

	cosine( X, Y )
		Static method that computes the cosine similarity between two bipolar hypervectors.

	mul( A, B )
		Static method that performs the binding operation which for bipolar hypervectors 
		takes of element-wise product.

	threshold( )
		It threholds the underlying numpy array to bipolar values. Useful after iterated
		`BypolarHypervector` sums.

	add( B )
		It adds `B` bipolar values to `self`.

	"""
	def __init__( self, hv ):

		super().__init__( hv )

	@classmethod
	def new( clazz, d, rng ):
		"""
		It creates a new random BipolarHypervector using a numpy
		array as underlying data type. Each element of the vector
		has a value in {-1, 1}.
	
		Parameters
		----------

		d : int
			The dimensionality of the hyper vector e.g. 10,000

		rng : numpy.random.Generator
			The random number generator to use for creating the vector.

		Returns
		-------

		BipolarHypervector
			A random bipolar hypervector.

		"""
		v = [-1, 1 ]

		hv = np.array( [ v[ i ] for i in rng.integers( 2, size=d ) ], dtype='int32' )

		return clazz( hv=hv )


	@classmethod
	def cosine( clazz, X, Y ):
		"""	
		It computes the cosine similarity between two bipolar hypervectors X and Y.

		Parameters
		----------

		X : BipolarHypervector
			A bipolar hypervector

		Y : BipolarHypervector
			A bipolar hypervector

		Returns
		-------
		
		float
			The cosine similarity between X and Y. If `cosine( X, Y )` = 0 then
			X and Y are orthogonal. X and Y are identical when `cosine( X, Y )` = 1.

		"""

		if Hypervector.check( clazz.__name__, X, Y ):
			
			normX = np.linalg.norm( X._hv )
			normY = np.linalg.norm( Y._hv )

			return np.dot( X._hv, Y._hv ) / ( normX * normY )

		raise ValueError( f'X and Y must be {clazz.__name__}' )

	def __len__( self ):
		"""
		It overrides the `__len__` magic function so that 
		the `len` function can be invoked on a BipolarHypervector.

		Returns
		-------
		
		int
			The length of the underlying numpy array.

		"""
		return len( self._hv )


	def permute( self, c=1 ):
		"""
		It computes a permutation of the underlying numpy array using right-cyclic shift
		(rotate by one coordinate position). The permutation is applied `c` times.

		Parameters
		----------

		c : int
			The number of permutations to apply.

		Returns
		-------

		BipolarHypervector
			The permuted hypervector. The permutation occurs in-place.

		"""

		self._hv = np.concatenate( ( self._hv[ -c: ], self._hv[ :-c ] ) )

		return self

	def __str__( self ):
		"""
		It overrides the magic `__str__` function in order to pretty print 
		the underlying numpy array.

		Returns
		-------

		str
			The content of the underlying numpy array in string format.

		"""
		return str( self._hv.tolist() )

	def __eq__( self, other ):
		"""
		It overrides the magic `__eq__` function in order to support
		comparison of two bipolar hypervectors.

		Parameters
		----------

		other : BipolarHypervector
			A bipolar hypervector to compare with self.

		Returns
		-------

		boolean
			`True` if the underlying numpy arrays are equal, `False` otherwise.
		
		"""
		
		if isinstance( other, BipolarHypervector ):
		
			return np.array_equal( self._hv, other._hv )

		raise False

	def __hash__( self ):
		"""
		It overrides the magic `__hash__` function in order to support 
		using `BipolarHypervector`s as keys in dictionaries.

		Returns
		-------

		int : the hash value of the object

		"""
		return hash( self._hv.tostring() )

	@staticmethod
	def sum( *args ):
		"""
		It computes the element-wise sum of an arbitrary number of 
		BipolarHypervectors.

		Parameters
		----------
		
		*args : BipolarHypervectors
			A variable number of BipolarHypervectors.

		Returns
		-------

		BipolarHypervector
			A new BipolarHypervector representing the element-wise sum 
			of the input BipolarHypervectors.

		"""

		if len( args ) == 2:
			return BipolarHypervector( hv=np.add( args[0]._hv, args[1]._hv ) )
		
		hv = np.vstack( [ arg._hv for arg in args ] )
		
		hv = np.sum( hv, axis=0, dtype='int32' )

		return BipolarHypervector( hv=hv )

	def add( self, other ):
		"""
		Convenience method that adds a bipolar vector to `self` in-place, useful for 
		speeding up iterated sums.

		Parameters
		----------

		other : BipolarHypervector

		"""
		self._hv = np.add( self._hv, other._hv )

	def threshold( self ):
		"""
		It thresholds the underlying numpy array so that each value is bipolar.
		This is decoupled from the `sum` method in order to support iterated 
		sums and thresholding of the final result as otherwise sums of thresholded
		vectors might converge very quickly.

		Returs
		-------
		
		BipolarHypervector
			It returns `self` with the underlying numpy array thrsholded to bipolar values,
			specifically, values greater than 0 are thresholded to 1 and values equal to
			or less than 0 are thresholded to -1.

		"""
		self._hv[ self._hv > 0 ] = 1
		self._hv[ self._hv <= 0 ] = -1

		return self

	@staticmethod
	def mul( A, B ):
		"""
		It performs a binding operation, binding A and B together for form X = A*B.
		For bipolar hypervectors the binding operation is element-wise multiplication.

		Parameters
		----------

		A : BipolarHypervector
			A bipolar hypervector

		B : BipolarHypervector
			A bipolar hypervector

		Returns
		-------

		BipolarHypervector
			A * B

		"""

		return BipolarHypervector( hv=A._hv * B._hv )

		


