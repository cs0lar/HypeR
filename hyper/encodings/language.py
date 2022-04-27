from annoy import AnnoyIndex

from hyper.encodings.text import TextEncoding

class LanguageEncoding():

	def __init__( self, alphabet, n, preprocessor=None ):

		self._encoder = TextEncoding( alphabet, n )
		self._preprocessor = preprocessor

	def encodesentence( self, sentence ):

		if callable( self._preprocessor ):

			sentence = self._preprocessor( sentence )
		
		return self._encoder.encode( sentence, progressbar=False )

	def encode( self, langfile ):

		with open( langfile, 'r' ) as f:

			# encode the first sentence
			langvector = self.encodesentence( f.readline() )

			# create a language vector bu summing all the encoded text samples
			[ langvector.add( self.encodesentence( line ) ) for line in f ]

			return langvector.threshold()

	@staticmethod
	def index( langvectors, d, ntrees=10 ):

		index = AnnoyIndex( d, 'angular' )

		for i, v in enumerate( langvectors ):

			index.add_item( i, v )

		index.build( ntrees )

		return index

	@staticmethod
	def find( unknown, index, nn=1 ):

		return index.get_nns_by_vector( unknown, nn )
