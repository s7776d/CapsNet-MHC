import sys

import torch
import torch.nn as nn
import torch.nn.functional as func
import math
import numpy as np


from config_parser import Config

#####################################################################################################################
#
# Weight initial setup
#
def weight_initial(model, config):
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            if m.bias is not None:
               nn.init.constant_(m.bias.data, 0.0)
      
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.uniform_(param, a=-0.01, b=0.01)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#####################################################################################################################

class Conv1dSame(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, depth_wise=False, batch_norm=True):
        super(Conv1dSame, self).__init__()
        if depth_wise:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, groups=in_channels)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.pad = nn.ConstantPad1d(kernel_size//2, 0)
        self.batch_norm = batch_norm
        self.act = nn.LeakyReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, input_tensor):
        out = self.conv(self.pad(input_tensor))
        out = self.act(out)
        if self.batch_norm:
            out = self.bn(out)
        return out


#####################################################################################################################

def squash(x):
    length2 = x.pow(2).sum(dim=2)+1e-7
    length = length2.sqrt()
    x = x*(length2/(length2+1)/length).view(x.size(0), x.size(1), -1)
    return x



class CapsLayer(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim):
        super(CapsLayer, self).__init__()
        self.input_caps = input_caps
        self.input_dim = input_dim
        self.output_caps = output_caps
        self.output_dim = output_dim
        self.weights = nn.Parameter(torch.Tensor(self.input_caps, self.input_dim, self.output_caps * self.output_dim))
        self.routing_module = AgreementRouting(self.input_caps, self.output_caps)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    # 输入的x格式为：(batch_m, input_caps, input_dim)
    def forward(self, u):
        u = u.unsqueeze(2)
        #print(u.shape[0])
        
        #weights = self.weights.unsqueeze(2)
        #print(self.weights.shape)
        #u_predict = torch.einsum('abik, ackj -> abcij', weights, u)
        #print('u_predict size is : ', u.size())
        u_predict = u.matmul(self.weights)
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)
        
        s = u_predict
        #print('s size is : ', list(s.size()))
        v = self.routing_module(s)
        #v = squash(s)
        #print('v squash size is : ', list(v.size()))
        probs = v.pow(2).sqrt()
        #print('probs size is : ', list(probs.size()))
        return v, probs


class AgreementRouting(nn.Module):
    def __init__(self, input_caps, output_caps, n_iterations=3):
        super().__init__()
        self.n_iterations = n_iterations
        self.b = torch.zeros((input_caps, output_caps)).cuda()

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()
        self.b.zero_()
        c = func.softmax(self.b, dim=1)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))  # 这块带上batch是因为每个样本的c都不一样
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)

                c = func.softmax(b_batch.view(-1, output_caps), dim=1).view(-1, input_caps, output_caps, 1)
                s = (c * u_predict).sum(dim=1)
                v = squash(s)

        return v

'''class CNN_Peptide_Encoder(nn.Module):
    def __init__(self, input_dim):
        super(CNN_Peptide_Encoder, self).__init__()


        self.conv_0 =nn.Conv1d(input_dim, 32, kernel_size=1,bias=False)
        #self.att_0 = Attention(32,15)

        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv_1", nn.Conv1d(32, 64, kernel_size=3))
        self.conv.add_module("bn_1",nn.BatchNorm1d(64))
        self.conv.add_module("ReLU_1",nn.LeakyReLU())
        self.conv.add_module("conv_2", nn.Conv1d(64, 10, kernel_size=3))
        self.conv.add_module("bn_2",nn.BatchNorm1d(10))
        self.conv.add_module("ReLU_2",nn.LeakyReLU())

    def forward(self, x):
        x = self.conv_0(x)
        #print('x size for att pep : ', list(x.size()))
        #y,att=self.att_0(x)
        #print('y size for att pep : ', list(y.size()))
        y=self.conv(x)
        #print('y after conv size for att pep : ', list(y.size()))
        # x is (batch_size, 10, 11), due to the max_len_pep is 15
        return y

class CNN_HLA_Encoder(nn.Module):
    def __init__(self,input_dim):
        super(CNN_HLA_Encoder, self).__init__()

        self.conv = torch.nn.Sequential()

        self.conv.add_module("conv_1", nn.Conv1d(input_dim, 64, kernel_size=3))
        self.conv.add_module("bn_1",nn.BatchNorm1d(64))
        self.conv.add_module("ReLU_1",nn.LeakyReLU())
        self.conv.add_module("conv_1_2", nn.Conv1d(64, 128, kernel_size=4))
        self.conv.add_module("bn_1_2",nn.BatchNorm1d(128))
        self.conv.add_module("ReLU_1_2",nn.LeakyReLU())       
        self.conv.add_module("maxpool_1", torch.nn.MaxPool1d(kernel_size=4))


        #self.att_0 = Attention2(128,95)

        self.conv1 = torch.nn.Sequential()
       
        self.conv1.add_module("conv_2", nn.Conv1d(128, 256, kernel_size=4)) # out 92
        self.conv1.add_module("bn_2",nn.BatchNorm1d(256))
        self.conv1.add_module("ReLU_2",nn.LeakyReLU())
        self.conv1.add_module("maxpool_2", torch.nn.MaxPool1d(kernel_size=4)) #out 23
        self.conv1.add_module("conv_3", nn.Conv1d(256, 10, kernel_size=2)) #out 22
        self.conv1.add_module("bn_3",nn.BatchNorm1d(10))
        self.conv1.add_module("ReLU_3",nn.LeakyReLU())
        self.conv1.add_module("maxpool_3", torch.nn.MaxPool1d(kernel_size=2)) # out 11


    def forward(self, x):
        x = self.conv(x)
        #print('x size for att hla : ', list(x.size()))
        #y,att=self.att_0(x)
        #print('y size for att hla : ', list(y.size()))
        y=self.conv1(x)
        #print('y after conv size for att hla : ', list(y.size()))
        # x is (batch_size, 10, 11), due to the max_len_pep is 15
        return y
'''

class CNN_Peptide_Encoder(nn.Module):
    def __init__(self, input_dim):
        super(CNN_Peptide_Encoder, self).__init__()


        self.conv_0 =nn.Conv1d(input_dim, 32, kernel_size=1,bias=False)
        self.att_0 = Attention(32,15)

        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv_1", nn.Conv1d(32, 64, kernel_size=3))
        self.conv.add_module("bn_1",nn.BatchNorm1d(64))
        self.conv.add_module("ReLU_1",nn.LeakyReLU())
        self.conv.add_module("conv_2", nn.Conv1d(64, 20, kernel_size=3))
        self.conv.add_module("bn_2",nn.BatchNorm1d(20))
        self.conv.add_module("ReLU_2",nn.LeakyReLU())

    def forward(self, x):
        x = self.conv_0(x)
        #print('x size for att pep : ', list(x.size()))
        y,att=self.att_0(x)
        #print('y size for att pep : ', list(y.size()))
        y=self.conv(y)
        #print('y after conv size for att pep : ', list(y.size()))
        # x is (batch_size, 10, 11), due to the max_len_pep is 15
        return y

class CNN_HLA_Encoder(nn.Module):
    def __init__(self,input_dim):
        super(CNN_HLA_Encoder, self).__init__()

        self.conv = torch.nn.Sequential()

        self.conv.add_module("conv_1", nn.Conv1d(input_dim, 64, kernel_size=3))
        self.conv.add_module("bn_1",nn.BatchNorm1d(64))
        self.conv.add_module("ReLU_1",nn.LeakyReLU())
        self.conv.add_module("conv_1_2", nn.Conv1d(64, 128, kernel_size=4))
        self.conv.add_module("bn_1_2",nn.BatchNorm1d(128))
        self.conv.add_module("ReLU_1_2",nn.LeakyReLU())       
        self.conv.add_module("maxpool_1", torch.nn.MaxPool1d(kernel_size=4))


        self.att_0 = Attention2(128,95)

        self.conv1 = torch.nn.Sequential()
       
        self.conv1.add_module("conv_2", nn.Conv1d(128, 256, kernel_size=4)) # out 92
        self.conv1.add_module("bn_2",nn.BatchNorm1d(256))
        self.conv1.add_module("ReLU_2",nn.LeakyReLU())
        self.conv1.add_module("maxpool_2", torch.nn.MaxPool1d(kernel_size=4)) #out 23
        self.conv1.add_module("conv_3", nn.Conv1d(256, 20, kernel_size=2)) #out 22
        self.conv1.add_module("bn_3",nn.BatchNorm1d(20))
        self.conv1.add_module("ReLU_3",nn.LeakyReLU())
        self.conv1.add_module("maxpool_3", torch.nn.MaxPool1d(kernel_size=2)) # out 11


    def forward(self, x):
        #print(x.shape)
        x = self.conv(x)
        #print('x size for att hla : ', x.shape)
        y,att=self.att_0(x)
        #print('y size for att hla : ', y.shape)
        y=self.conv1(y)
        #print('y after conv size for att hla : ', list(y.size()))
        # x is (batch_size, 10, 11), due to the max_len_pep is 15
        return y

class Attention(nn.Module):
    def __init__(self, in_channels,seq_length):
        super(Attention,self).__init__()
                
        self.seq_length=seq_length
        for i in range(seq_length):
        	setattr(self,"fc%d" % i, nn.Linear(in_channels,1))
        self.sm=nn.Softmax(dim=1)


    def forward(self, seq_feature):
        #get weight

        seq_feature = seq_feature.permute(0,2,1).contiguous()
        # shape to [batch, seq_length,in_channels]
        attn_weight = [0]*self.seq_length
        for i in range(self.seq_length):
        	attn_weight[i]=getattr(self,"fc%d" % i)(seq_feature[:,i,:])
        #     attn_weight.append(self.fc_list[i](seq_feature[:,i,:]))
        attn_weight = torch.stack(attn_weight,dim=1)
        # output dimension: [batch, seq_length, 1]
        attn_weight = self.sm(attn_weight)
        # output dimension: [batch, seq_length, 1] 
        out = seq_feature*attn_weight
        # output dimension: [batch, seq_length, in_channels] 
        out = out.permute(0,2,1).contiguous()
        # output dimension: [batch, in_channels,seq_length]  
        attn_weight2=torch.reshape(attn_weight, (attn_weight.size(0),attn_weight.size(1)))

        return out,attn_weight2
        # [batch,in_channels,seq_length]


###############################################################################################################################


class Attention2(nn.Module):
    def __init__(self, in_channels,seq_length):
        super(Attention2,self).__init__()
    
        self.fc=nn.Linear(in_channels+seq_length,1) 
        # the output will be (batch_size,seq_length,1)
        self.sm=nn.Softmax(dim=1)


    def forward(self, seq_feature):
        #get wight
 

        original= seq_feature

        pos = [torch.eye(seq_feature.size(2))]*seq_feature.size(0)
        pos =  torch.stack(pos,dim = 0).cuda()  
        seq_feature = torch.cat([seq_feature,pos],dim=1)
        # print(seq_feature)
        # print(seq_feature.size())
        # exit()


        seq_feature = seq_feature.permute(0,2,1).contiguous()
        # shape to [batch, seq_length,in_channels+seq_length]
        attn_weight = self.fc(seq_feature)
        # output dimension: [batch, seq_length, 1]
        attn_weight = self.sm(attn_weight)
        # output dimension: [batch, seq_length, 1] 
        # attn_weight = attn_weight*(math.sqrt(self.seq_size))
        # #  # output dimension: [batch, seq_length, 1]   
        original = original.permute(0,2,1).contiguous()
        # shape to [batch, seq_length,in_channels]       
        out = original*attn_weight     
        # shape to [batch, seq_length,in_channels]
        # out = out+seq_feature
        out = out.permute(0,2,1).contiguous()
        # shape to [batch, in_channels,seq_length]        
        # attn_weight = attn_weight.permute(0,2,1).contiguous()
        # # output dimension: [batch, 1, seq_length]  
        # out = torch.bmm(attn_weight,seq_feature)
        # # get[batch,1,in_channels]
        # print(out.size())
        # exit()

        return out, attn_weight.view(attn_weight.size(0),-1)
        # [batch,1,in_channels]

###############################################################################################################################


#
# Context extractor
#

class Context_extractor(nn.Module):

    def __init__(self, seq_size):
        super(Context_extractor, self).__init__()
        self.net = CapsLayer(input_caps=40, input_dim=11, output_caps=20, output_dim=11)
        #self.conv_0 =nn.Conv1d(20, 20, kernel_size=1,bias=False)
        #self.att_0 = Attention(20, 1)
        #print('herekjhigsdlchsdgdugyu')
        #self.prelu = nn.PReLU()
        #print('hello')
        #self.dropout = nn.Dropout(p=0.5)
        self.out_vector_dim = 20 * seq_size

    def forward(self, list_tensors):
        out = torch.cat(list_tensors, dim=1)
        #print('size of out 1 :', list(out.size()))
        out, out_ = self.net(out)
        #x = self.conv_0(out)
        #print('size of out :', list(out.size()))
        #out1, out2 = self.att_0(x)
        #print(out)
        #print(ty)
        #out_1 = torch.Tensor(out[0])

        #out_2 = torch.Tensor(out[1])
        #out = torch.cat((out_1, out_2), dim=1)
        #out = np.asarray(out, dtype=np.float64)
        #out = torch.from_numpy(out)
        #out = np.array(out, dtype=np.float64)
        #out = torch.Tensor(out)
        #print('size of out1 :', list(out1.size()))
        # flatten
        #out = torch.flatten(out)
        #print(list(out.view(out.size(0), -1).size()))
        return out.view(out.size(0), -1)

#####################################################################################################################
#
# Predictor
#

class Context_extractor00(nn.Module):

    def __init__(self, seq_size):
        super(Context_extractor00, self).__init__()
        self.net = nn.Sequential(
            Conv1dSame(20, 256, 3),
            nn.LeakyReLU(),
            Conv1dSame(256, 64, 3),
            nn.LeakyReLU(),
        )
        self.out_vector_dim = 64 * seq_size

    def forward(self, list_tensors):
        out = torch.cat(list_tensors, dim=1)
        #print('size of out 1:', list(out.size()))
        out = self.net(out)
        # flatten
        #print('size of out :', list(out.size()))
        #print(list(out.view(out.size(0), -1).size()))
        return out.view(out.size(0), -1)

##############
class Predictor(nn.Module):

    def __init__(self, input_size):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 200),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            # nn.Linear(1024, 512),
            # nn.LeakyReLU(),
            # nn.Dropout(0.5),

            nn.Linear(200, 1)
        )
        self.out_act = nn.Tanh()

    def forward(self, context_vector):
        out = self.net(context_vector)
        #print('here out is : ', out)
        #print('sadadd', list(self.out_act(out).size()))
        #print('out', out)
        return self.out_act(out)

#####################################################################################################################
#
# Model
#


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.encoder_hla_a2 = CNN_HLA_Encoder(23)

        self.encoder_peptide2 = CNN_Peptide_Encoder(23)
        
        self.context_extractor2 = Context_extractor(11)
        self.predictor = Predictor(self.context_extractor2.out_vector_dim)

    def forward(self, hla_a_seqs, hla_a_mask, hla_a_seqs2, hla_a_mask2,peptides, pep_mask,peptides2,pep_mask2):

        hla_out2  = self.encoder_hla_a2(hla_a_seqs2)
        pep_out2  = self.encoder_peptide2(peptides)
        #print(hla_out2.size())
        #print(pep_out2.size())
        #print(([hla_out2, pep_out2]))
        context2  = self.context_extractor2([hla_out2, pep_out2])
        
        #print(list(context2.size()))#(list(context2.size()))
        #print(context2.size())
        #context2 = torch.flatten(context2)
        #print(list(context2.size()))
        #context2 = torch.reshape(context2, (704))
        #print(len(context2))
        
        ic50 = self.predictor(context2)
        #print('ic50', ic50)
        return ic50


def main():
    test()

if __name__ == '__main__':
    main()
    pass