import sys, re

import numpy as np

from tqdm import tqdm

from stop_words import get_stop_words

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

from hyper.hypervector import BipolarHypervector
from hyper.encodings.text import Alphabet

nltk.download('omw-1.4')

class WordContextVectorisation():

	def __init__( self, V, d, rng ):

		self._alphabet = Alphabet( d=d, rng=rng, alphabet=V )


	def vectorise( self, doc, w, l, d ):
		"""
		Parameters
		----------

		doc: the document containing the word to vectorise

		w: the word to vectorise
		
		l: the window either side of the target word that constitutes its context

		d: dimensionality of the hypervector
		
		"""

		idxs = [ i for i, x in enumerate( doc ) if x == w ]

		W = BipolarHypervector( hv=np.zeros( d, dtype=np.uint8 ) )

		for idx in idxs:
			
			start = max( 0, idx - l )
			end = min( idx + l + 1, len( doc ) ) 

			[ W.add( self._alphabet[ doc[ x ] ] ) for x in range( start, end ) if x != idx ]


		return W.threshold()


lemmatiser = WordNetLemmatizer()

def vocabularise( doc ):

	regex = r"([\w]+(?:(?!\s)\W?[\w]+)*)"
	tokeniser = RegexpTokenizer( regex )

	def isalphanumstr( string ):

		return any( [ ch.isalnum() for ch in string ] ) and not string.isdigit()


	doc = re.sub( r'[—_:’,\.]', ' ', doc )

	tokens = [ w.lower() for w in tokeniser.tokenize( doc ) if w not in get_stop_words( 'en' ) and isalphanumstr( w ) and len( w ) > 1 ]

	doc = [ lemmatiser.lemmatize( t ) for t in tokens ] 
	
	return doc, list( set( doc ) )


if __name__ == '__main__':
	
	d = 10000
	l = 5
	rng = np.random.default_rng( 12345 ) 

	docpath = sys.argv[ 1 ]

	with open( docpath, 'r' ) as f:
		doc = f.read()

	doc, V = vocabularise( doc )
	wcv = WordContextVectorisation( V, d, rng )

	# print ( wcv.vectorise( doc, 'confess', l, d ) )
	# print ( wcv._alphabet[ 'unapproachable' ] )
	# X = wcv.vectorise( doc, lemmatiser.lemmatize( 'artery' ), l, d )
	# Y = wcv.vectorise( doc, lemmatiser.lemmatize( 'traffic' ), l, d )
	W2V = {}
	V2W = {}

	for w in tqdm( V ):
		vector = wcv.vectorise( doc, w, l, d )
		W2V[ w ] = vector
		V2W[ vector ] = w 

	queriesidxs = np.random.randint( len( V ), size=20 )
	vectors = list( W2V.values() )

	for qidx in queriesidxs:

		q = V[ qidx ]
		Q = W2V[ q ]

		vectors = sorted( vectors, key=lambda X : BipolarHypervector.cosine( Q, X ) )

		result = [ V2W[ v ] for v in vectors[ :3 ] ]
		print ( f'( {q}: { result }' )