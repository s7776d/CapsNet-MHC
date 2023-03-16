import sys
import datetime
import os, math
import pandas as pd

from ray.train.torch import TorchCheckpoint
from ray.air import session
import argparse
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.schedulers import ASHAScheduler
import ray
from ray import air, tune

import torch
from torch.optim import lr_scheduler
from torch import optim
import torch.nn as nn

from sklearn.metrics import confusion_matrix
import numpy as np
from collections import Counter
from seq_encoding import ENCODING_METHOD_MAP
from config_parser import Config
from model import (
    Model,
    weight_initial,
    count_parameters,
)
from data_provider import DataProvider
from logger import (
    setup_logging,
    log_to_file,
)
from callbacks import (
    ModelCheckPointCallBack,
    EarlyStopCallBack,
)
from result_writer import (
    weeekly_result_writer
)

import numbers
import numpy as np

import numbers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#############################################################################################
#
# Write result
#
#############################################################################################

def weeekly_result_writer(result_dict, config):
    """Write prediction result as an additional column
    out [weekly_result.txt]
    """
    out_file_path = os.path.join(config.working_dir, 'weekly_result.txt')
    out_file = open(out_file_path, 'w')

    with open(config.test_file) as in_file:
        for line_num, line in enumerate(in_file):
            info = line.strip('\n').split('\t')
            if line_num == 0:
                # title
                out_str = '{}\t{}\n'.format(
                    '\t'.join(info[:4]),
                    '\t'.join(info[4:])
                )
                out_file.write(out_str)
            else:
                alleles = info[0]
                peptide = info[1]

                hla_a = alleles

                uid = '{hla_a}-{peptide}'.format(
                    hla_a=hla_a,
                    peptide=peptide,
                )

                if uid not in result_dict:
                    value = '-'
                else:
                    value = result_dict[uid]
                out_str = '{}\t{}\t{}\n'.format(
                    '\t'.join(info[:4]),
                    value,
                    '\t'.join(info[4:])
                )
                out_file.write(out_str)

    return out_file_path

#############################################################################################
#
# Write metrics
#
#############################################################################################

METHOD_LIST = [
    'our_method_ic50',
    
]
from os import readlink

def get_acc(result_file):
    
    f = open(result_file,"r")
    lines = f.readlines()
    y_prob = []
    y_true = []
    lines.pop(0)
    #lines.pop(-1)
    for x in lines:
      #if (x.split('\t')[3]) == '12':
        y_prob.append(x.split('\t')[4])
        y_true.append(x.split('\t')[2])
    f.close()
    y_prob = [float(num) for num in y_prob]
    y_true = [float(num) for num in y_true]
    y_pred = [round(num) for num in y_prob]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels = [0, 1]).ravel().tolist()
    accuracy = (tp+tn)/(tn+fp+fn+tp)

    return accuracy

def get_weekly_result_info_dict(result_file):

    result_info = {}
    with open(result_file) as in_file:
        for line_num, line in enumerate(in_file):
            if line_num == 0:
                continue

            info = line.strip('\n').split('\t')
            #date = info[0]
            #iedb_id = info[0]
            full_allele = info[0]
            #measure_type = info[4]
            pep_len = len(info[3])
            measure_value = float(info[2])

            record_id = '{}-{}-{}'.format(full_allele, pep_len)

            if record_id not in result_info:
                result_info[record_id] = {}
                result_info[record_id]['full_allele'] = full_allele
                #result_info[record_id]['date'] = date
                result_info[record_id]['pep_length'] = pep_len
                #result_info[record_id]['iedb_id'] = iedb_id
                #result_info[record_id]['measure_type'] = measure_type
                result_info[record_id]['label_values'] = []
                result_info[record_id]['method_values'] = {}
                for method in METHOD_LIST:
                    result_info[record_id]['method_values'][method] = []

            # fill real value
            result_info[record_id]['label_values'].append(measure_value)

            # fill prediction values, if no result, do not fill
            for method_index, method_name in enumerate(METHOD_LIST):
                col_index = method_index + 4
                val = info[col_index]
                try:
                    val = float(val)
                    result_info[record_id]['method_values'][method_name].append(val)
                except:
                    pass

    return result_info
	
def batch_train(model, data, conf):
    hla_a, hla_mask, hla_a2, hla_mask2,  pep, pep_mask, pep2, pep_mask2, ic50,samples = data

    pred_ic50= model(hla_a.to(device), hla_mask.to(device), hla_a2.to(device), hla_mask2.to(device), pep.to(device), pep_mask.to(device),pep2.to(device), pep_mask2.to(device))
    loss = nn.MSELoss()(pred_ic50.to(conf.cpu_device), ic50.view(ic50.size(0), 1))

    return loss


def batch_validation(model, data, conf):
    hla_a, hla_mask, hla_a2, hla_mask2,  pep, pep_mask, pep2, pep_mask2, ic50,samples = data
    with torch.no_grad():
    	 # validation_call
         pred_ic50= model(hla_a.to(device), hla_mask.to(device), hla_a2.to(device), hla_mask2.to(device), pep.to(device), pep_mask.to(device),pep2.to(device), pep_mask2.to(device))
         loss = nn.MSELoss()( pred_ic50.to(conf.cpu_device), ic50.view(ic50.size(0), 1))
         pred_ic50=pred_ic50.view(len(pred_ic50)).tolist()

         return loss,pred_ic50,samples


def train(conf, model, optimizer,data_provider,p):
    # skip training if test mode
    if not conf.do_train:
       log_to_file('Skip train', 'Not enabled training')
       return
    device = conf.device
    log_to_file('Device', device)
        # log pytorch version
    log_to_file('PyTorch version', torch.__version__)
        # prepare model
    log_to_file('based on base_model #', p)

    for i in range(conf.model_count):
        log_to_file('begin training model #',i)
        log_to_file('Trainable params count', count_parameters(model))
        print(model.parameters())
        log_to_file("Optimizer", "Adam")
        # call backs
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, threshold=conf.loss_delta, patience=4,
                                               cooldown=4, verbose=True, min_lr=conf.min_lr, factor=0.2)
        model_check_callback = ModelCheckPointCallBack(
           model,
           conf.model_save_path(p*conf.model_count+i),
           period=1,
           delta=conf.loss_delta,
        )
        early_stop_callback = EarlyStopCallBack(patience=5, delta=conf.loss_delta)

        # some vars
        epoch_loss = 0
        validation_loss = 0
        data_provider.new_epoch()
                
        steps = data_provider.train_steps()
        log_to_file('Start training1', datetime.datetime.now())
    

        for epoch in range(conf.epochs):
            epoch_start_time = datetime.datetime.now()
            # train batches
            print(steps)
            model.train(True)
            for _ in range(steps):
                data = data_provider.batch_train(i)
                # print("***")
                loss = batch_train(model, data, conf)
                # print("loss:",loss)
                # exit()
                loss.backward()
                # clip grads
                nn.utils.clip_grad_value_(model.parameters(), conf.grad_clip)
                # update params
                optimizer.step()
                # record loss
                epoch_loss += loss.item()
                # reset grad
                optimizer.zero_grad()
                # time compute
            time_delta = datetime.datetime.now() - epoch_start_time
                # validation on epoch end

            model.eval()
            val_sample=[]
            val_pred=[]
            for _ in range(data_provider.val_steps()):
                data = data_provider.batch_val(i)
                t_loss,t_pred,t_samples=batch_validation(model,  data, conf)
                val_sample.append(t_samples)
                val_pred.append(t_pred)
                validation_loss += t_loss
            # log
            log_to_file("Training process", "[base_model{0:1d}]-[model{1:1d}]-[Epoch {2:04d}] - time: {3:4d} s, train_loss: {4:0.5f}, val_loss: {5:0.5f}".format(
               p,i, epoch, time_delta.seconds, epoch_loss / steps, validation_loss / data_provider.val_steps()))
            # call back
            model_check_callback.check(epoch, validation_loss / data_provider.val_steps())
            if early_stop_callback.check(epoch, validation_loss / data_provider.val_steps()):
                break
            epoch_loss = 0
            validation_loss = 0
            # reset data provider
            data_provider.new_epoch()
            # save last epoch model
            torch.save(model.state_dict(), os.path.join(conf.working_dir, 'last_epoch_model_{}.pytorch'.format(p*conf.model_count+i)))

def batch_test(model,  data, conf):
    hla_a, hla_mask, hla_a2, hla_mask2, pep, pep_mask, pep2, pep_mask2, uid_list = data
    pred_ic50 = model(hla_a.to(device), hla_mask.to(device), hla_a2.to(device), hla_mask2.to(device), pep.to(device), pep_mask.to(device), pep2.to(device), pep_mask2.to(device))
    return pred_ic50, uid_list

def test(conf, model, data_provider):

    if not conf.do_test:
        log_to_file('Skip testing', 'Not enabled testing')
        return

    device = conf.device
    
    temp_list=[] 
    for p in range(conf.base_model_count):
        for k in range(conf.model_count):
    # load and prepare model
             state_dict = torch.load(conf.model_save_path(p*conf.model_count+k))
            #  model = Model(conf)
             model.load_state_dict(state_dict)
            #  model.to(device)
             model.eval()
             temp_dict={}
             data_provider.new_epoch()
             for _ in range(data_provider.test_steps()):
                   data = data_provider.batch_test()
                   with torch.no_grad():
                        pred_ic50, uid_list= batch_test(model, data, conf)
                        for i, uid in enumerate(uid_list):
                            temp_dict[uid] = pred_ic50[i].item()
             temp_list.append(temp_dict)

    # average score of the emsemble model
    result_dict=temp_list[0]
    if conf.model_count>1:
       for k in range(1,conf.model_count):
           for j in result_dict.keys():
                result_dict[j]+=temp_list[k][j]

    if conf.base_model_count>1:
       for p in range(1,conf.base_model_count):
           for k in range(conf.model_count):
               for j in result_dict.keys():
                   result_dict[j]+=temp_list[p*conf.model_count+k][j]

    for j in result_dict.keys():
    	result_dict[j]=result_dict[j]/(conf.model_count*conf.base_model_count)

    # print(result_dict)
    result_file = weeekly_result_writer(result_dict, conf)
    log_to_file('Testing result file', result_file)

    acc = get_acc(result_file)
    return acc



# if __name__ == '__main__':
#     main()



def train_capsules(config):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(conf, config["dropout"])
    model.to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=config["lr"]
    )
    
    should_checkpoint = True
    encoding_func = ENCODING_METHOD_MAP[conf.encoding_method]
    encoding_func2= ENCODING_METHOD_MAP[conf.encoding_method2]
    data_provider=[]
    for p in range(conf.base_model_count):
        temp_provider = DataProvider(
             encoding_func,
             encoding_func2,
             conf.data_file,
             conf.test_file,
             config["batch_size"],
             max_len_hla=conf.max_len_hla,
             max_len_pep=conf.max_len_pep,
             model_count=conf.model_count
        )
        data_provider.append(temp_provider)
    while True:
        train(conf, model, optimizer, data_provider[p],p)
        acc = test(conf, model, data_provider[0])
        checkpoint = None
        if should_checkpoint:
            checkpoint = TorchCheckpoint.from_state_dict(model.state_dict())
        session.report({"mean_accuracy": acc}, checkpoint=checkpoint)


if __name__ == "__main__":
    conf_file = sys.argv[1]
    conf = Config(conf_file)
    

    # for early stopping
    sched = AsyncHyperBandScheduler()

    resources_per_trial = {"cpu": 4, "gpu": 1}  # set this for GPUs
    tuner = tune.Tuner(
        tune.with_resources(train_capsules, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            metric="mean_accuracy",
            mode="max",
            scheduler=sched,
            num_samples=10 
        ),
        run_config=air.RunConfig(
            name="tune_capsules",
            stop={
                "mean_accuracy": 0.98,
                "training_iteration": 10 
            },
        ),
        param_space={
            "lr": tune.choice([1e-1, 1e-2, 1e-3]),
            "batch_size": tune.choice([40, 100, 1024]),
            "dropout": tune.choice([0.2, 0.5, 0.75])
        },
    )
    results = tuner.fit()
    
    print("Best conf is:", results.get_best_result().config)
