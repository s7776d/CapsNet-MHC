import os
import math
import random

import torch
import numpy as np
import torch.nn.functional as F
import os, time

if not os.path.isfile("esmfold.model"):
  # download esmfold params
  os.system("apt-get install aria2 -qq")
  os.system("aria2c -q -x 16 https://colabfold.steineggerlab.workers.dev/esm/esmfold.model &")

  # install libs
  os.system("pip install -q omegaconf pytorch_lightning biopython ml_collections einops py3Dmol")
  os.system("pip install -q git+https://github.com/NVIDIA/dllogger.git")

  # install openfold
  commit = "6908936b68ae89f67755240e2f588c09ec31d4c8"
  os.system(f"pip install -q git+https://github.com/aqlaboratory/openfold.git@{commit}")

  # install esmfold
  os.system(f"pip install -q git+https://github.com/sokrypton/esm.git")

  # wait for Params to finish downloading...
  if not os.path.isfile("esmfold.model"):
    # backup source!
    os.system("aria2c -q -x 16 https://files.ipd.uw.edu/pub/esmfold/esmfold.model")
  else:
    while os.path.isfile("esmfold.model.aria2"):
      time.sleep(5)


from string import ascii_uppercase, ascii_lowercase
import hashlib, re, os
import numpy as np
from jax.tree_util import tree_map
import matplotlib.pyplot as plt
from scipy.special import softmax
jobname = "test" #@param {type:"string"}
jobname = re.sub(r'\W+', '', jobname)[:50]



def parse_output(output):
  pae = (output["aligned_confidence_probs"][0] * np.arange(64)).mean(-1) * 31
  plddt = output["plddt"][0,:,1]
  
  bins = np.append(0,np.linspace(2.3125,21.6875,63))
  sm_contacts = softmax(output["distogram_logits"],-1)[0]
  sm_contacts = sm_contacts[...,bins<8].sum(-1)
  xyz = output["positions"][-1,0,:,1]
  mask = output["atom37_atom_exists"][0,:,1] == 1
  o = {"pae":pae[mask,:][:,mask],
       "plddt":plddt[mask],
       "sm_contacts":sm_contacts[mask,:][:,mask],
       "xyz":xyz[mask]}
  return o['pae']

def get_hash(x): return hashlib.sha1(x.encode()).hexdigest()

def generate_protein_pretraining_representation(prots):
      
      sequence = prots
      sequence = re.sub("[^A-Z:]", "", sequence.replace("/",":").upper())
      sequence = re.sub(":+",":",sequence)
      sequence = re.sub("^[:]+","",sequence)
      sequence = re.sub("[:]+$","",sequence)
      copies = 1 #@param {type:"integer"}
      if copies == "" or copies <= 0: copies = 1
      sequence = ":".join([sequence] * copies)
      num_recycles = 3 #@param ["0", "1", "2", "3", "6", "12", "24"] {type:"raw"}
      chain_linker = 25 

      ID = jobname+"_"+get_hash(sequence)[:5]
      seqs = sequence.split(":")
      lengths = [len(s) for s in seqs]
      length = sum(lengths)
      c.append(length)
      print("length",length)

      u_seqs = list(set(seqs))
      if len(seqs) == 1: mode = "mono"
      elif len(u_seqs) == 1: mode = "homo"
      else: mode = "hetero"

      if "model" not in dir():
        import torch
        model = torch.load("esmfold.model")
        model.eval().cuda().requires_grad_(False)

      # optimized for Tesla T4
      if length > 700:
        model.set_chunk_size(64)
      else:
        model.set_chunk_size(128)

      torch.cuda.empty_cache()
      output = model.infer(sequence,
                          num_recycles=num_recycles,
                          chain_linker="X"*chain_linker,
                          residue_index_offset=512)

      pdb_str = model.output_to_pdb(output)[0]
      output = tree_map(lambda x: x.cpu().numpy(), output)
      return parse_output(output)


############################################################################
# Data provider
############################################################################

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
class DataProvider:
    def __init__(self, sequence_encode_func, sequence_encode_func2, data_file, test_file, batch_size, max_len_hla=273, max_len_pep=37,
      model_count=5, shuffle=True):
        self.batch_size = batch_size
        self.data_file = data_file

        self.test_file = test_file
        self.sequence_encode_func = sequence_encode_func
        self.sequence_encode_func2= sequence_encode_func2
        self.shuffle = shuffle
        self.max_len_hla = max_len_hla
        self.max_len_pep = max_len_pep
        self.model_count=model_count

        self.batch_index_train = 0
        self.batch_index_val = 0
        self.batch_index_test = 0
        # cache
        self.pep_encode_dict = {}
        self.hla_encode_dict = {}
        self.hla_encode_dict2= {}
        self.pep_encode_dict2= {}

        self.hla_sequence = {}
        self.read_hla_sequences()

        self.samples = []  
        self.train_samples = []
        self.validation_samples = []
        self.read_training_data()
        self.split_train_and_val()

        self.weekly_samples = []
        self.read_weekly_data()


    def train_steps(self):
        return math.ceil(len(self.train_samples[0]) / self.batch_size)

    def val_steps(self):
        return math.ceil(len(self.validation_samples[0]) / self.batch_size)

    def test_steps(self):
        return math.ceil(len(self.weekly_samples) / self.batch_size)

    def read_hla_sequences(self):

        file_path = os.path.join(BASE_DIR, '../..','Anthem_dataset',  'proteins_esmfold.txt')
        with open(file_path, 'r') as in_file:
            for line_num, line in enumerate(in_file):
                if line_num == 0:
                    continue

                info = line.strip('\n').split(' ')
                seq=info[1]
                if(len(seq)>=self.max_len_hla):
                   seq=seq[:self.max_len_hla]
                self.hla_sequence[info[0]] = seq

    
    def read_weekly_data(self):

        with open(self.test_file) as in_file:
            for line_num, line in enumerate(in_file):
                if line_num == 0:
                    continue

                info = line.strip('\n').split('\t')
                alleles = info[0]
                peptide = info[1]
                if len(peptide) > self.max_len_pep:
                    continue
                hla_a = alleles
                if hla_a not in self.hla_sequence :
                    continue
                uid = '{hla_a}-{peptide}'.format(
                    hla_a=hla_a,
                    peptide=peptide,
                )
                self.weekly_samples.append((hla_a,peptide, uid))
            
    def read_training_data(self):

        with open(self.data_file) as in_file:
            for line_num, line in enumerate(in_file):
                if line_num == 0:
                    continue

                info = line.strip('\n').split('\t')
                # print(info)

                hla_a = info[0]

                if hla_a not in self.hla_sequence :
                    continue

                peptide = info[1]
                if len(peptide) > self.max_len_pep:
                    continue

                ic50 = float(info[2])

                self.samples.append((hla_a,peptide, ic50))

        if self.shuffle:
            random.shuffle(self.samples)


    def split_train_and_val(self):

        vd_count=math.ceil(len(self.samples)/max(self.model_count,5))
        for i in range(max(self.model_count-1,4)):
            self.validation_samples.append(self.samples[i*vd_count:(i+1)*vd_count])
            temp_sample=self.samples[:]
            del(temp_sample[i*vd_count:(i+1)*vd_count])
            self.train_samples.append(temp_sample)


        self.validation_samples.append(self.samples[len(self.samples)-vd_count:])
        temp_sample=self.samples[:]
        del(temp_sample[len(self.samples)-vd_count:])
        self.train_samples.append(temp_sample)


    def batch_train(self,order):
        """A batch of training data
        """
        data = self.batch(self.batch_index_train, self.train_samples[order])
        self.batch_index_train += 1
        return data

    def batch_val(self,order):
        """A batch of validation data
        """
        data = self.batch(self.batch_index_val, self.validation_samples[order])
        self.batch_index_val += 1
        return data

    def batch_test(self):
        """A batch of test data
        """
        data = self.batch(self.batch_index_test, self.weekly_samples, testing=True)
        self.batch_index_test += 1
        return data

    def new_epoch(self):
        """New epoch. Reset batch index
        """
        self.batch_index_train = 0
        self.batch_index_val = 0
        self.batch_index_test = 0

    

    def batch(self, batch_index, sample_set, testing=False):
        """Get a batch of samples
        """
        hla_a_tensors11=[]
        hla_a_tensors = []
        hla_a_mask = []
        hla_a_tensors2 = []
        hla_a_mask2 = []        
        pep_tensors = []
        pep_mask = []
        pep_tensors2 = []
        pep_mask2 = []
        ic50_list = []

        # for testing
        uid_list = []
        #validation_call
        sample_prototype=[]

        def encode_sample(sample):
            hla_a_allele = sample[0]
            pep = sample[1]
            if not testing:
                ic50 = sample[2]
            else:
                uid = sample[2]
            
            
            if hla_a_allele not in self.hla_encode_dict:
                hla_a_tensor = generate_protein_pretraining_representation(self.hla_sequence[hla_a_allele])
                # print(hla_a_tensor.shape)
                hla_a_tensor = torch.Tensor(hla_a_tensor)
                if list(hla_a_tensor.size()) != [366,366]:
                    if list(hla_a_tensor.size()) == [181,181]:
                      hla_a_tensor = F.pad(input=hla_a_tensor , pad=(92, 93, 92, 93), mode='constant', value=0)
                    if list(hla_a_tensor.size()) == [206,206]:
                      hla_a_tensor = F.pad(input=hla_a_tensor , pad=(80, 80, 80, 80), mode='constant', value=0)
                    if list(hla_a_tensor.size()) == [362,362]:
                      hla_a_tensor = F.pad(input=hla_a_tensor , pad=(0, 4, 0, 4), mode='constant', value=0)
                    if list(hla_a_tensor.size()) == [363,363]:
                      hla_a_tensor = F.pad(input=hla_a_tensor , pad=(0, 3, 0, 3), mode='constant', value=0)
                    if list(hla_a_tensor.size()) == [365,365]:
                      hla_a_tensor = F.pad(input=hla_a_tensor , pad=(0, 1, 0, 1), mode='constant', value=0)
                self.hla_encode_dict[hla_a_allele] = hla_a_tensor

            hla_a_tensors.append(self.hla_encode_dict[hla_a_allele])
            hla_a_mask.append(self.hla_encode_dict[hla_a_allele])


            if hla_a_allele not in self.hla_encode_dict2:
                hla_a_tensor2 = self.sequence_encode_func2(self.hla_sequence[hla_a_allele])
                self.hla_encode_dict2[hla_a_allele] = (hla_a_tensor2)

            hla_a_tensors2.append(self.hla_encode_dict2[hla_a_allele][0])
            hla_a_mask2.append(self.hla_encode_dict2[hla_a_allele][1])

            if pep not in self.pep_encode_dict:
                pep_tensor, mask = self.sequence_encode_func(pep, self.max_len_pep)              
                self.pep_encode_dict[pep] = (pep_tensor, mask)
            pep_tensors.append(self.pep_encode_dict[pep][0])
            pep_mask.append(self.pep_encode_dict[pep][1])

            if pep not in self.pep_encode_dict2:
                pep_tensor2,_= self.sequence_encode_func2(pep, self.max_len_pep)
                self.pep_encode_dict2[pep] = (pep_tensor2)
            pep_tensors2.append(self.pep_encode_dict2[pep])
            


            if not testing:
                ic50_list.append(ic50)
            else:
                uid_list.append(uid)
            
        
        start_i = batch_index * self.batch_size
        end_i = start_i + self.batch_size
        
        for sample in sample_set[start_i: end_i]:
           # doesn't matter if the end_i exceed the maximum index
           #validation_call
           sample_prototype.append(sample)
           encode_sample(sample)

        if len(hla_a_tensors) < self.batch_size:
            if len(sample_set) < self.batch_size:
                for _ in range(self.batch_size - len(hla_a_tensors)):
                    #validation_call
                    temp=random.choice(sample_set)
                    sample_prototype.append(temp)                    
                    encode_sample(temp)

            else:
                for i in random.sample(range(start_i), self.batch_size - len(hla_a_tensors)):
                    #validation_call
                    sample_prototype.append(sample_set[i])
                    encode_sample(sample_set[i])
        if not testing:
            
            return (
                torch.stack(hla_a_tensors, dim=0),
                torch.stack(hla_a_mask, dim=0),
                torch.stack(hla_a_tensors2, dim=0),
                torch.stack(hla_a_mask2, dim=0),


                torch.stack(pep_tensors2, dim=0),
                torch.stack(pep_mask, dim=0),
                pep_tensors2,
                torch.stack(pep_mask2, dim=0),

                torch.tensor(ic50_list),
                #validation_call
                sample_prototype  

            )
        else:
            return (
                torch.stack(hla_a_tensors, dim=0),
                torch.stack(hla_a_mask, dim=0),
                torch.stack(hla_a_tensors2, dim=0),
                torch.stack(hla_a_mask2, dim=0),

                pep_tensors,
                torch.stack(pep_mask, dim=0),
                torch.stack(pep_tensors2, dim=0),
                torch.stack(pep_mask2, dim=0),

                uid_list,
            )
