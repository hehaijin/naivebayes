run:
	python preprocess.py training.csv

	python project2.py result.csv testing.csv

	python ... 

	python rankvocs.py result.csv vocabulary.txt

clean:
	$(RM) preprocess *~
	$(RM) project2 *~
	$(RM) ...
	$(RM) rankvocs *~