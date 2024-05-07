"""ESM protein embedding"""


"""install esm and esmfold in terminal"""
#pip install fair-esm 
#pip install "fair-esm[esmfold]"
#pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
#pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'


"""import libraries"""
import torch
import esm


"""Load ESM-2 model"""
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

batch_converter = alphabet.get_batch_converter()

model.eval()

Enzymedata = [("enzyme1","MQKKVIAAIIGTSAISAVAATQANAATTHTVKPGESVWAISNKYGISIAKLKSLNNLTSNLIFPNQVLKVSGSSNSTSNSSRPSTNSGGGSYYTVQAGDSLSLIASKYGTTYQNIMRLNGLNNFFIYPGQKLKVSGTASSSNAASNSSRPSTNSGGGSYYTVQAGDSLSLIASKYGTTYQKIMSLNGLNNFFIYPGQKLKVTGNASTNSGSATTTNRGYNTPVFSHQNLYTWGQCTYHVFNRRAEIGKGISTYWWNANNWDNAAAADGYTIDNRPTVGSIAQTDVGYYGHVMFVERVNNDGSILVSEMNYSAAPGILTYRTVPAYQVNNYRYIH")]

batch_labels, batch_strs, batch_tokens = batch_converter(Enzymedata)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33]

# Sequence representation
sequence_representations = []
for i, tokens_len in enumerate(batch_lens):
    sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

print(sequence_representations[0])
print(sequence_representations[0].shape)

import matplotlib.pyplot as plt
for (_, seq), tokens_len, attention_contacts in zip(Enzymedata, batch_lens, results["contacts"]):
    plt.matshow(attention_contacts[: tokens_len, : tokens_len])
    plt.title('Enzyme')
    plt.savefig('matrix.jpg',dpi=300)
    plt.show()

