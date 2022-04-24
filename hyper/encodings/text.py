import time, sys, re

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

from unidecode import unidecode

from sklearn.manifold import TSNE

from hyper.hypervector import BipolarHypervector


class Alphabet():

	ALPHABET = [ chr( 32 ) ] + [ chr( i ) for i in range( 97, 123 ) ]

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
	def __init__( self, alphabet, n ):
		
		if not alphabet or not isinstance( alphabet, Alphabet ):

			raise ValueError( 'alphabet must be an instance of hyper.encodings.text.Alphabet' )
		
		self._alphabet = alphabet
		self._n = n 

	def encodengram( self, ngram ):

		def mul( A, B ):
			A._hv = BipolarHypervector.mul( A, B )._hv

		try:

			n = len( ngram )

			A = self._alphabet[ ngram[ 0 ] ].permute( c=( n-1 ) )

			[ mul( A, self._alphabet[ ngram[ i ] ].permute( c=( n-i-1 ) ) ) for i in range( 1, n ) ]

			return A 

		except KeyError as e:

			raise ValueError( f'key {e.args[ 0 ]} not in alphabet' )


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
		
		[ A.add( self.encodeblock( text[ i ], text[ i+step ], step, V ) ) for i in tqdm( range( len( text )-step  ) ) ]

		return A.threshold()

	def encodeblock( self, v, w, n, A ):

		V = self._alphabet[ v ].permute( c=( n-1 ) )
		W = self._alphabet[ w ]

		V = BipolarHypervector.mul( V, A )
		V.permute( c=1 )

		A._hv = BipolarHypervector.mul( V, W )._hv 
		return A



if __name__ == '__main__':
	
	langs = [

		'bul_news_2020_30K',
		'ces_news_2020_30K',
		'dan_news_2020_30K',
		'deu_news_2021_30K',
		'ell_news_2020_30K',
		'eng_news_2020_30K',
		'est_news_2020_30K',
		'fin_news_2020_30K',
		'fra_news_2020_30K',
		'hun_news_2020_30K',
		'ita_news_2020_30K',
		'lav_news_2020_30K',
		'lit_news_2020_30K',
		'nld_news_2020_30K',
		'pol_news_2020_30K',
		'por_news_2020_30K',
		'ron_news_2020_30K',
		'slk_news_2020_30K',
		'slv_news_2020_30K',
		'spa_news_2020_30K',
		'swe_news_2020_30K'	
	
	]

	labels = [

		'bul', 'ces', 'dan', 'deu', 'ell', 'eng',
		'est', 'fin', 'fra', 'hun', 'ita', 'lav',
		'lit', 'nld', 'pol', 'por', 'ron', 'slk',
		'slv', 'spa', 'swe'

	]
	
	suffix = '-sentences.txt'

	d = 10000
	n = 4
	l = len( langs )
	rng = np.random.default_rng(  )

	alphabet = Alphabet( d=d, rng=rng )

	encoding = TextEncoding( alphabet, n )

	regex = re.compile( r'[\W\d\s]+' )

	def process( string, alphabet ):

		unidecoded = unidecode( string ).lower()
		unidecoded = regex.sub( ' ', unidecoded )
		out = ''.join( c for c in unidecoded )

		return out

	base = sys.argv[ 1 ]
	L = []

	for i, lang in enumerate( langs ):
		with open( f'{base}/{lang}/{lang}{suffix}', 'r' ) as f:
		
			lines = [ process( line.split( '\t' )[ 1 ].lower(), Alphabet.ALPHABET ) for line in f ]
		
		text = ' '.join( lines )[ :1000000 ]

		textVector = encoding.encode( text ) 
		L.append( textVector )

	D = np.zeros( ( l, l ) )

	for i in range( l ):
		for j in range( l ):
			D[ i, j ] = BipolarHypervector.cosine( L[ i ], L[ j ] )
	
	embed = TSNE( n_components=2, learning_rate='auto', metric='precomputed', init='random', square_distances=True ).fit_transform( D )

	markers = [ 

		'o', 'v', '^', '<', '>', '1', '2', '3', 
		'4', '8', 's', 'p', '*', 'h', 'H', '+',
		'x', 'X', 'D', 'd', '|'

	]

	cmap = plt.cm.get_cmap( 'hsv', l )

	for i in range( l ):
		x, y = embed[ i ]
		plt.scatter( x, y, color=cmap( i ), alpha=0.5, marker=markers[ i ] )
	
	plt.legend( labels )

	plt.show()