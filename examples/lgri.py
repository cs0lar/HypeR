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

	def __init__( self, datadir, d, n, rng ):

		self._datadir = datadir
		self._alphabet = Alphabet( d=d, rng=rng )
		self.encoding = LanguageEncoding( self._alphabet, n, preprocessor=self.preprocess )
		self._regex = re.compile( r'[\W\d\s]+' )

	def preprocess( self, text ):

		text = text.split( '\t' )[ 1 ] 
		
		# try to convert unicode character to ASCII characters
		unidecoded = unidecode( text.lower() )

		# replace sequences of characters outside of ALPHABET
		# with a single space character 
		unidecoded = self._regex.sub( ' ', unidecoded )

		return unidecoded


# https://wortschatz.uni-leipzig.de/en/download
def main( datadir ):

	langs = []

	d = 10000

	lgwri = LanguageGeometryWithRandomIndexing( datadir=datadir, d=d, n=4, rng=np.random.default_rng() )

	i = 0
	
	for langfile in tqdm( LanguageGeometryWithRandomIndexing.LANGUAGE_FILES ):

		file = f'{datadir}/{langfile}/{langfile}-sentences.txt'

		langs.append( lgwri.encoding.encode( file ) )

		i += 1

	idx = LanguageEncoding.index( langs, d )

	inferred = LanguageEncoding.find( langs[ 0 ], idx )

	print ( inferred )
	# tsne = TSNE( n_components=2, perplexity=8, learning_rate='auto', metric='precomputed', square_distances=True )

	# M = np.zeros( ( len( langs ), len( langs ) ) )

	# for i, X in enumerate( langs ):

	# 	for j, Y in enumerate( langs ):

	# 		M[ i, j ] = BipolarHypervector.cosine( X, Y )

	# L_2d = tsne.fit_transform( 1.- M )

	# plt.scatter( L_2d[ :, 0 ], L_2d[ :, 1 ] )

	# for label, x, y in zip( LanguageGeometryWithRandomIndexing.LABELS, L_2d[ :, 0 ], L_2d[ :, 1 ] ):

	# 	plt.annotate(

	# 		label,
	# 		xy=( x, y ),
	# 		xytext=( -20, 20 ),
	# 		textcoords=  'offset points',
	# 		ha='right',
	# 		va='bottom',
	# 		bbox=dict( boxstyle='round, pad=0.5', fc='yellow', alpha=0.5 ),
	# 		arrowprops=dict( arrowstyle='->', connectionstyle='arc3, rad=0' ),
	# 		fontsize='x-large'

	# 	)

	# frame = plt.gca()

	# frame.axes.get_xaxis().set_ticks( [] )	
	# frame.axes.get_yaxis().set_ticks( [] )

	# plt.show()

if __name__ == '__main__':
	
	main( datadir=sys.argv[ 1 ] )


	

