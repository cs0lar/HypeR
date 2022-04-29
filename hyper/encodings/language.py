import numpy as np

from hyper.hypervector import BipolarHypervector
from hyper.encodings.text import TextEncoding

class LanguageEncoding():
	"""
	It implements a Language hypervector as described in
	https://redwood.berkeley.edu/wp-content/uploads/2020/08/JoshiEtAl-QI2016-language-geometry-copy.pdf.
	A language is encoded as the hypervector sum of sample documents. That is, given a corpus of
	documents for a given language, each sample document is encoded into an hypervector and
	all resultants hypervectors are summed together to form a single-vector representation
	of the target language.

	Attributes
	----------

	_encode: TextEncoding
		The underlying hypervector encoder for text.

	_preprocessor: function
		A function that takes a string and returns a modified string that
		reflects any cleanup or alteration required before encoding, e.g. removing
		characters for which there is no hypervector mapping. Default is `None`.

	Methods
	--------

	encodesentence( sentence ):
		It uses the underlying `TextEncoding` to encode a sentence/paragraph of text.
		If a preprocessor has been specified, this is applied to the text before
		submitting to the encoder.

	encode( langfile ):
		It takes the path to a file containing sentences in the target language
		encoding each sentence and producing the ultimate language encoding hypervector.

	find( unknown, languages ):
		It finds the most probable language for any text that is encoded as an hypervector.

	"""
	def __init__( self, alphabet, n, preprocessor=None ):

		self._encoder = TextEncoding( alphabet, n )
		self._preprocessor = preprocessor

	def encodesentence( self, sentence ):
		"""
		It encodes a sentence or paragraph of text into a bipolar hypervector. If a preprocessor
		function has been supplied to the constructor of this class, it will be used here to 
		apply any required clean-up and modifications before encoding.

		Parameters
		-----------

		sentence: string
			The text to be encoded into an hypervector.

		Returns
		-------

		BipolarHypervector
			The hypervector representation of the input sentence

		"""
		if callable( self._preprocessor ):

			sentence = self._preprocessor( sentence )
		
		return self._encoder.encode( sentence, progressbar=False )

	def encode( self, langfile ):
		"""
		It takes the path to a file containing a corpus of samples of the target language
		and computes a single `BipolarHypervector` representing that language. It is assumed that
		each line of the corpus file contains a sentence/paragraph. Any preprocessing step required
		for cleaning the input text should be passed to this class constructor as the `preprocessor`
		parameter which is assumed to be a callable that takes a string and returns another string.

		Parameters
		----------

		langfile: string
			The path to a language corpus file containing a sentence/paragraph per line 

		Returns
		-------

		BipolarHypervector
			A `BipolarVector` representation of the target language.

		"""
		with open( langfile, 'r' ) as f:

			# encode the first sentence
			langvector = self.encodesentence( f.readline() )

			# create a language vector bu summing all the encoded text samples
			[ langvector.add( self.encodesentence( line ) ) for line in f ]

			return langvector.threshold()

	@staticmethod
	def find( unknown, langvectors ):
		"""
		Static method that returns the most probable language given the hypervector representation
		of some text. Specifically, given a list L of hypervectors representing a set of languages it computes 
		argmax_i cosine( L_i, U ) where U is the hypervector representation of the input text.

		Parameters
		----------

		unknown : BipolarHypervector
			A `BipolarHypervector` encoding of some text for which we wish to know the language.
		
		langvectors : list
			A list of `BipolarHypervector`s representing a quorum of languages.

		Returns
		--------

		int
			The index of the vector most similar to `unknown` according to cosine similarity.


		"""

		return np.argmax( [ BipolarHypervector.cosine( unknown, V ) for V in langvectors ] )
