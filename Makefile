run:
	python preprocess.py training.csv

	python project2.py result.csv testing.csv

	python rankvocs.py result.csv vocabulary.txt
	
	python withbeta.py result.csv testing.csv

	python plotbeta.py

clean:
	$(RM) preprocess *~
	$(RM) project2 *~
	$(RM) rankvocs *~
	$(RM) withbeta *~
	$(RM) plotbeta *~
