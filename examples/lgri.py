import re 

from unidecode import unidecode

from hyper.hypervector import BipolarHypervector
from hyper.encodings.text import Alphabet, TextEncoding

class LanguageGeometryWithRandomIndexing():

	LANGUAGE_FILES = [

		'bul_news_2020_10K',
		'ces_news_2020_10K',
		'dan_news_2020_10K',
		'deu_news_2021_10K',
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
		self._encoding = TextEncoding( self._alphabet, n )
		self._regex = re.compile( r'[\W\d\s]+' )

	def preprocess( self, text ):

		# try to convert unicode character to ASCII characters
		unidecoded = unidecode( text.lower() )

		# replace sequences of characters outside of ALPHABET
		# with a single space character 
		unidecoded = regex.sub( ' ', unidecoded )

		return unidecoded

	def encodesentence( self, sentence ):

		text = self.preprocess( sentence.split( '\t' )[ 1 ] )

		return self._encoding.encode( text )

	def encodelang( self, langidx ):

		langfile = LANGUAGE_FILES[ langidx ]

		with open( f'{self._datadir}/{langfile}/{langfile}-sentences.txt' ) as f:

			# encode the first sentence
			langvector = self.encodesentence( f.readline() )

			# create a language vector by summing all the encoded text samples
			[ langvector.add( self.encodesentence( line ) ) for line in f ]

			return langvector.threshold()


	

