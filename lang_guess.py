#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""N-Gram-based Language Identification."""

import codecs
import os
import re
import sys

from math   import sqrt
from common import *

__author__  = "Eric Hildebrand <hildebrand.eric@gmail.com>"
__version__ = "0.9"

SEP = '#'
        
# Logger instance
LOGGER         = logging.getLogger( __name__ )
formatter      = logging.Formatter( '%(asctime)s\t%(levelname)s:\t%(message)s', '%Y-%m-%d %H:%M:%S' )
consoleHandler = logging.StreamHandler( )
consoleHandler.setFormatter( formatter )
LOGGER.addHandler( consoleHandler )
LOGGER.setLevel( logging.DEBUG )


def getNGrams( term, n ):
    """Creates a list of n-grams from a term given the size of n.

    term:    A string containing the term.
    n:       An integer denoting the size of the n-grams.

    returns: A list of strings containing the n-grams of the term.
    """
    nGrams = []
    if not term:
        LOGGER.warning( 'Empty term...' )
        return nGrams
    if len( term ) < n:
        numSEP = n - len( term )
        return [numSEP * SEP + term, term + numSEP * SEP]
    if n > 1:
        term = SEP + term + SEP # Add single borders
    for idx in xrange( len( term ) - n + 1 ):
        nGrams.append( term[idx:idx + n] )
    return nGrams


def getModelFromFile( fn, n ):
    """Generates an n-gram model from a text file.

    fn: A string containing the path to a text file.
    n: An integer denoting the size of the n-grams.

    returns: A mapping of n-grams to their corresponding frequencies.
             N-Grams are lists of strings containing the n-grams of a term.
    """
    nGrams = { }
    terms = codecs.open( fn, 'r', 'utf8' ).read( ).split( )
    for term in terms:
        for nGram in getNGrams( term, n ):
            nGrams[nGram] = nGrams.get( nGram, 0 ) + 1
    return nGrams


def createModel( lang, src, n = 2, writeToDisc = False ):
    """Creates a n-gram model for a given text collection and n.

    lang:        A string containing the language of the files.
    src:         A string containing the path to a text file or a folder
                 of text files. Text files must have the suffix '_prep.txt',
                 ideally created by prep.py.
    n:           An integer denoting the size of the n-grams.
    writeToDisc: Optionally creates a file for the resulting n-gram model.
    returns:     A mapping of n-grams to their corresponding frequencies.
                 N-Grams are lists of strings containing the n-grams of a term.
    """
    LOGGER.info( 'Creating language model for <%s>...'%( lang ) )
    model = { }
    if os.path.isdir( src ):
        processed = 0
        for root, dirs, files in os.walk( src ):
            for textFile in filter( lambda fileName: fileName.endswith( '_prep.txt' ), files ):
                processed += 1
                if processed % 100 == 0:
                    LOGGER.info( '... processed %d files...'%( processed ) )
                for nGram, freq in getModelFromFile( os.path.join( root, textFile ), n ).iteritems( ):
                    model[nGram] = model.get( nGram, 0 ) + freq
    elif os.path.isfile( src ):
        model = getModelFromFile( src, n )
    if writeToDisc:
        fn = 'model_%s_%d.txt'%( lang, n )
        modelFile = codecs.open( fn, 'w', 'utf8' )
        LOGGER.info( '... sorting... ' )
        for nGram in sorted( model, key = model.get, reverse = True )[:500]:
            modelFile.write( nGram + '\t' + str( model[nGram] ) + '\n' )
        modelFile.close( )
        LOGGER.info( '... saved model in <%s>...'%( fn ) )
    LOGGER.info( '... done.' )
    return model


def getCosSim( model1, model2 ):
    """Computes the cosine similarity of two n-gram models.

    model1:  A mapping of n-grams to their corresponding frequencies.
             N-Grams are lists of strings containing the n-grams of a term.
    model2:  Another mapping n-grams to their corresponding frequencies.
    returns: The cosine similarity of the model's n-gram 'vectors'.
    """
    num    = 0
    denom1 = 0
    denom2 = 0
    for term1 in model1:
        term2Freq = model2.get( term1, 0 )
        num    += model1[term1] * term2Freq
        denom1 += model1[term1] * model1[term1]
        denom2 += term2Freq * term2Freq
    for term2 in set( model2.iterkeys( ) ) - set( model1.iterkeys( ) ):
        denom2 += model2[term2] * model2[term2]
    return num / ( sqrt( denom1 ) * sqrt( denom2 ) )


def computeSimilarities( lang, model, src, n ):
    """Computes the similarities of an n-gram model and a collection of text files.

    The similarity of each text is written in a new file named sim_<lang>_<n>.txt.

    lang:  A string containing the language of the files.
    model: A string containing the path to a n-gram model file.
    src:   A string containing the path to a text file or a folder
           of text files. Text files must have the suffix '_prep.txt',
           ideally created by prep.py.
    n:     An integer denoting the size of the n-grams.
    """
    ref = { }
    for line in codecs.open( model, 'r', 'utf8' ):
        nGram, freq = line.split( )
        ref[nGram] = int( freq )
    simFile = codecs.open( 'sim_%s_%d.txt'%( lang, n ), 'w', 'utf8' )
    model = { }
    if os.path.isdir( src ):
        processed = 0
        for root, dirs, files in os.walk( src ):
            for textFile in filter( lambda fileName: fileName.endswith( '_prep.txt' ), files ):
                processed += 1
                if processed % 100 == 0:
                    LOGGER.info( '... processed %d files...'%( processed ) )
                fn = os.path.join( root, textFile )
                for nGram in getModelFromFile( fn, n ):
                    model[nGram] = model.get( nGram, 0 ) + 1
                simFile.write( fn.decode( 'utf8' ) + '\t' + str( getCosSim( ref, model ) ) + '\n' )
    elif os.path.isFile( src ):
        for nGram in getModelFromFile( src, n ):
            model[nGram] = model.get( nGram, 0 ) + 1
        simFile.write( src + '\t' + getCosSim( ref, model ) + '\n' )
    simFile.close( )


def showHelp( ):
    """Prints a short help message on how to use this program."""
    print 'usage: python %s  -s source_file_or_dir [-l lang] [-n n_gram_size] [-r ref_model]'%( sys.argv[0] )
    sys.exit( )


def main( ):
    if len( sys.argv ) == 1 or '-h' in sys.argv:
        showHelp( )
    
    lang = sys.argv[sys.argv.index( '-l' ) + 1] if '-l' in sys.argv else 'en'
    n    = int( sys.argv[sys.argv.index( '-n' ) + 1] ) if '-n' in sys.argv else 2
    src = sys.argv[sys.argv.index( '-s' ) + 1]
    if '-r' in sys.argv:
        ref = sys.argv[sys.argv.index( '-r' ) + 1]
        computeSimilarities( lang, ref, src, n )
    else:
        createModel( lang, src, n, True )
        
        
if __name__ == '__main__':
    main( )