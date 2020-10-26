gloss.txt ==> mapping between WordNet ID and glosses for all synsets (85K)
words.txt ==> Obtain the words of a synset / words of each entity, i.e., n03206908 -> dish


WN9_IMG is originally from Image-embodied knowledge representation learning (IKRL).
IKRL says 'The triple part of WN9-IMG is the subset of a classical
KG dataset WN18 [Bordes et al., 2014]' which is Semantic Matching Energy Function for Learning with Multi-relational Data (SME) https://github.com/glorotxa/SME and wordnet-mlj12 is that dataset folder, but not need to track back since I found it in following GitHub.

WN9-IMG dataset is downloaded from
https://github.com/xrb92/IKRL - GitHub of one of the author of IKRL. It was named data.rar which later I renamed to WN9-IMG_IKRL.rar.



To export conda package:
conda list --export > package-list.txt

Reinstall packages from the export file:
conda create -n myenv --file package-list.txt