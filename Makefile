##############################
# Analyses

%.lm.nlz:: anti-analyser.py orn_lm.py %.npz
	nix-shell runtime.nix --run "python $< orn_lm $*.npz"

%.analysis:: anti-analyser.py orn_%.py orn_%.npz
	nix-shell runtime.nix --run "python $< orn_$* orn_$*.npz"

############################
# Dyck experiment

%_lm.npz: dyck.py %_lm.py
	nix-shell runtime.nix --run "LD_PRELOAD=/lib64/libcuda.so.1 python $< $*_lm"

LM: LM.hs typedflow.nix
	nix-shell --run "ghc -O2 --make $<"

%_lm.py: LM
	./LM $*

############################
# ambncmdn experiment

%_xdep.npz: xdep.py %_xdep.py
	nix-shell runtime.nix --run "LD_PRELOAD=/lib64/libcuda.so.1 python $< $*_xdep"


orn_xdep.py lstm_xdep.py: XDep.hs typedflow.nix
	nix-shell --run "ghci -e main $<"


################################
# Agreement experiment

%_classifier.npz: main_agr.py %_classifier.py data/DICT data/agr_50_mostcommon_10K.tsv
	nix-shell runtime.nix --run "python main_agr.py $*_classifier"

lstm_classifier.py orn_classifier.py: Classifier.hs
	nix-shell --pure --run "ghci -e main $<"

data/agr_50_mostcommon_10K.tsv.gz:
	mkdir -p data
	curl -o $@ -O http://tallinzen.net/media/rnn_agreement/agr_50_mostcommon_10K.tsv.gz

data/wikipedia.parsed.subset.50.gz:
	mkdir -p data
	curl -o $@ -O http://tallinzen.net/media/rnn_agreement/wikipedia.parsed.subset.50.gz

data/agr_50_mostcommon_10K.tsv: data/agr_50_mostcommon_10K.tsv.gz
	gunzip -k $<

data/wikipedia.parsed.subset.50: data/wikipedia.parsed.subset.50.gz
	gunzip -k $<

# data/wiki.vocab:
# 	mkdir -p data
# 	curl -o $@ -O https://github.com/GU-CLASP/agreement/raw/jyp/data/wiki.vocab

data/wiki.vocab.sorted: data/wiki.vocab
	tail -n +2 $< | sort --reverse --numeric-sort --key=3 > $@

data/wiki.vocab.50000: data/wiki.vocab
	tail -n +2 $< | sort --reverse --numeric-sort --key=3 | head -n 50000 | cut -f 1 | tr '[:upper:]' '[:lower:]' 	| sort | uniq > $@

data/POS: data/wiki.vocab
	tail -n +2 $< | cut --fields=2 | cut --d=' ' -f 1 | sort | uniq > $@

data/DICT: data/POS data/wiki.vocab.50000
	cat $^ > $@

data/rnn_agr_simple.tar.gz:
	mkdir -p data
	curl -o $@ -O http://tallinzen.net/media/rnn_agreement/rnn_agr_simple.tar.gz





#####################
# Infra

clean:
	rm -f *.aux *.bcf *.bbl *.blg

