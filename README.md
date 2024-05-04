# Contrastive speaker embedding with sequential disentanglement
This is a MoCo implementation of "Contrastive speaker embedding with sequential disentanglement".

## Requirements
Python >= 3.8, Pytorch >= 2.10, Pandas, Scipy

## Running examples
### Embdding training
```sh
python train.py
```
This code will load VoxCeleb2-dev, MUSAN, and RIR from "../corpus", the default directory of corpora. To simplify data loading,
meta information files containing path, utt_id, spk_id, No. of samples, duration, and speaker label are prepared and saved under
the "meta" directory. Running this code will create a "model_ckpt" directory and the checkpoints will be saved in this folder.

### Embdding extraction
```sh
python extract.py --ckpt_dir=model_ckpt/ckpt_20240503_0142 --ckpt_num=1
```
This code will extract embeddings using the saved checkpoint and save the embeddings under "eval/xvectors".

### Performance evaluation
```sh
python -m be_run.be_voxceleb
```
Based on the extracted embeddings, Cosine scores will be produced under "eval/scores" and EER and minDCF will be computed.

## References
If you are interested and find it helpful, please cite the following papers. Thanks.
```bibtex
@inproceedings{Tu24-contrastive_sd,
  title={Contrastive Speaker Embedding With Sequential Disentanglement},
  author={Tu, Y. Z. and  Mak, M. W. and Chien, J. T.},
  booktitle={Proc. International Conference on Acoustics, Speech, and Signal Processing},
  pages={10891--10895},
  year={2024}
}
```
```bibtex
@article{Tu24-contrastive_sd_j,
	Author = {Y. Z. Tu and M. W. Mak and J. T. Chien},
	Journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
	Title = {Contrastive Self-Supervised Speaker Embedding With Sequential Disentanglement},
	Year = {2024}
}
