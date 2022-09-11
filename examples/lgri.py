import re, sys 

import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

from unidecode import unidecode

from hyper.hypervector import BipolarHypervector
from hyper.encodings.text import Alphabet
from hyper.encodings.language import LanguageEncoding

class LanguageGeometryWithRandomIndexing():
	"""
	Implementation of the evaluation of language geometry using random indexing from 
	https://redwood.berkeley.edu/wp-content/uploads/2020/08/JoshiEtAl-QI2016-language-geometry-copy.pdf.

	Sentences from 21 languages are downloaded from https://wortschatz.uni-leipzig.de/en/download, specifically
	the 10k news sentence files for each language.

	This code assumes that the data for each language has been downloaded and extracted into a local directory 
	that will be specified as a parameter to the python command to run this example.

	"""
	LANGUAGE_FILES = [

		'bul_news_2020_10K',
		'ces_news_2020_10K',
		'dan_news_2020_10K',
		'deu_news_2020_10K',
		'ell_news_2020_10K',
		'eng_news_2020_10K',
		'est_news_2020_10K',
		'fin_news_2020_10K',
		'fra_news_2020_10K',
		'hun_news_2020_10K',
		'ita_news_2020_10K',
		'lav_news_2020_10K',
		'lit_news_2020_10K',
		'nld_news_2020_10K',
		'pol_news_2020_10K',
		'por_news_2020_10K',
		'ron_news_2020_10K',
		'slk_news_2020_10K',
		'slv_news_2020_10K',
		'spa_news_2020_10K',
		'swe_news_2020_10K'	

	]

	LABELS = [

		'bul', 'ces', 'dan', 'deu', 'ell', 'eng',
		'est', 'fin', 'fra', 'hun', 'ita', 'lav',
		'lit', 'nld', 'pol', 'por', 'ron', 'slk',
		'slv', 'spa', 'swe'

	]

	LABELS_ABBREV = [
		
		'bg', 'cs', 'da', 'de', 'el', 'en', 'et', 'fi', 'fr', 'hu',
		'it', 'lv', 'lt', 'nl', 'pl', 'pt', 'ro', 'sk', 'sl', 'es', 'sv'
	
	]

	"""
	The europarl test set from: S. Nakatani. langdetect is updated(added profiles of Estonian / Lithuanian
	/ Latvian / Slovene, and so on. http://shuyo.wordpress.com/2011/09/29/langdetect-is-updatedadded-profiles-of-estonian-lit
	Dowloadable from Google Code archives: https://code.google.com/archive/p/language-detection/downloads
	"""
	TESTSET_FILE = 'europarl.test'

	def __init__( self, datadir, d, n, rng ):

		self._datadir = datadir
		self._alphabet = Alphabet( d=d, rng=rng )
		self.encoding = LanguageEncoding( self._alphabet, n, preprocessor=self.preprocess )
		self._regex = re.compile( r'[\W\d\s]+' )
		self.langvecs = []

	def preprocess( self, text ):

		text = text.split( '\t' )[ 1 ] 
		
		# try to convert unicode character to ASCII characters
		unidecoded = unidecode( text.lower() )

		# replace sequences of characters outside of ALPHABET
		# with a single space character 
		unidecoded = self._regex.sub( ' ', unidecoded )

		return unidecoded


	def testset( self, lang ):

		testsetpath = f'{self._datadir}/{LanguageGeometryWithRandomIndexing.TESTSET_FILE}'

		with open( testsetpath, 'r' ) as f:

			sentences = [ line for line in f if line.startswith( lang ) ]

		return sentences


def buildlangvecs( datadir, show=False ):

	d = 10000

	lgwri = LanguageGeometryWithRandomIndexing( datadir=datadir, d=d, n=5, rng=np.random.default_rng() )

	i = 0
	
	# encode each language into a Language Hypervector
	for langfile in tqdm( LanguageGeometryWithRandomIndexing.LANGUAGE_FILES ):

		file = f'{datadir}/{langfile}/{langfile}-sentences.txt'

		lgwri.langvecs.append( lgwri.encoding.encode( file ) )

		i += 1

	if show:

		# compute the full similiarty matrix across all encoded languages
		M = np.zeros( ( len( lgwri.langvecs ), len( lgwri.langvecs ) ) )

		for i, X in enumerate( lgwri.langvecs ):

			for j, Y in enumerate( lgwri.langvecs ):

				M[ i, j ] = BipolarHypervector.cosine( X, Y )

		# use TSNE algorithm to map the languages into 2D space using the distance matrix
		tsne = TSNE( n_components=2, perplexity=8, learning_rate='auto', metric='precomputed', square_distances=True )

		L_2d = tsne.fit_transform( 1.- M )

		# plot the 2D version of the hypervectors
		plt.scatter( L_2d[ :, 0 ], L_2d[ :, 1 ] )

		for label, x, y in zip( LanguageGeometryWithRandomIndexing.LABELS, L_2d[ :, 0 ], L_2d[ :, 1 ] ):

			plt.annotate(

				label,
				xy=( x, y ),
				xytext=( -20, 20 ),
				textcoords=  'offset points',
				ha='right',
				va='bottom',
				bbox=dict( boxstyle='round, pad=0.5', fc='yellow', alpha=0.5 ),
				arrowprops=dict( arrowstyle='->', connectionstyle='arc3, rad=0' ),
				fontsize='x-large'

			)

		frame = plt.gca()

		frame.axes.get_xaxis().set_ticks( [] )	
		frame.axes.get_yaxis().set_ticks( [] )

		plt.show()

	return lgwri

def detectlang( sentence, lgwri ):

	unknown = lgwri.encoding.encodesentence( sentence )

	return LanguageGeometryWithRandomIndexing.LABELS[ lgwri.encoding.find( unknown, lgwri.langvecs ) ]


if __name__ == '__main__':
	
	lgwri = buildlangvecs( datadir=sys.argv[ 1 ], show=True )

	matches = 0
	sentences = 0

	for i in tqdm( range( len( LanguageGeometryWithRandomIndexing.LABELS ) ) ):
		
		expected = LanguageGeometryWithRandomIndexing.LABELS[ i ]
		unknowns = lgwri.testset( LanguageGeometryWithRandomIndexing.LABELS_ABBREV[ i ] )

		result = [ detectlang( sentence, lgwri ) ==  expected for sentence in unknowns ]
		
		sentences += len( unknowns )
		matches += np.sum( result )

	print ( f'Detection success = {matches / sentences}' )

			

	

