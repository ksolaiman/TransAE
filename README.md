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


CUDA_VISIBLE_DEVICES=1,2 python myscript.py


For Debug:
####### en_reln_mapping needs to be wn9, as there is no small entity2id or relation2id mapping files
####### retrain_text_layer can be false for subsequent runs
python3 main.py --nouse_gpu --dataset_path=synth_data_WN9/ --validation_freq=1 --en_reln_mapping='wn9' --retrain_text_layer=True