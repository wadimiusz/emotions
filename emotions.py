import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
def emotions (train_address, test_address, types):
	for type in types:
		tree = ET.parse(train_address)
		root = tree.getroot()
		X_train_not_vectorized = []
		y_train = []
		tables = root.find('./database')
		for bank_tables in tables.findall('table'):
			X_train_not_vectorized.append(bank_tables.find('./column[4]').text)
			y_train.append(bank_tables.find("./*[@name='%s']" % type).text)
		count_vect = CountVectorizer()
		#positive = y_train.count('1')
		#negative = y_train.count('-1')
		#neutral = y_train.count('0')
		#print('%s: %f / %f / %f (in the test set)' % (type, positive, neutral, negative))
		X_train = count_vect.fit_transform(X_train_not_vectorized)
		clf = MultinomialNB().fit(X_train, y_train)
		tree = ET.parse(test_address)
		root = tree.getroot()
		X_test_not_vectorized = []
		y_test = []
		tables = root.find('./database')
		for bank_tables in tables.findall('table'):
			X_test_not_vectorized.append(bank_tables.find('./column[4]').text)
			y_test.append(bank_tables.find("./*[@name='%s']" % type).text)
		X_test = count_vect.transform(X_test_not_vectorized)
		predicted = clf.predict(X_test)
		errors = np.sum(predicted != y_test)
		print('%s: %f percent errors on the test set' % (type, errors * 100 / len(y_test)))
print('Banks:')
emotions('database/bank_train_2016.xml', 'database/banks_test_etalon.xml', ['sberbank', 'vtb', 'gazprom', 'alfabank', 'bankmoskvy', 'alfabank', 'uralsib', 'rshb'])
print('Tkk')
emotions('database/tkk_train_2016.xml', 'database/tkk_test_etalon.xml', ['beeline', 'mts', 'megafon', 'tele2', 'rostelecom', 'komstar', 'skylink'])