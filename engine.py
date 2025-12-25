import math
import sys
import os
import datetime
import json
from turtle import undo
from typing import Iterable
from pathlib import Path

import torch

import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
import copy
import utils
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score


class Engine():
    def __init__(self, model=None,device=None,class_mask=[], domain_list= [], args=None):
        self.current_task=0
        self.current_classes=[]
        #! distillation
        self.class_group_num = 5
        self.classifier_pool = [None for _ in range(self.class_group_num)]
        self.class_group_train_count = [0 for _ in range(self.class_group_num)]
        
        self.task_num = len(class_mask)
        self.class_group_size = len(class_mask[0])
        self.distill_head= None
        self.model = model
        
        self.num_classes= max([item for mask in class_mask for item in mask])+1
        self.labels_in_head = np.arange(self.num_classes)
        self.added_classes_in_cur_task = set()
        self.head_timestamps = np.zeros_like(self.labels_in_head)
        self.args=args
        
        self.class_mask=class_mask
        self.domain_list=domain_list

        self.task_type="initial"
        self.args=args
        
        self.adapter_vec=[]
        self.task_type_list=[]
        self.class_group_list=[]
        self.adapter_vec_label=[]
        self.device=device
        
        if self.args.d_threshold:
            self.acc_per_label = np.zeros((self.args.class_num, self.args.domain_num))
            self.label_train_count = np.zeros((self.args.class_num))
            self.tanh = torch.nn.Tanh()
            
        self.cs=torch.nn.CosineSimilarity(dim=1,eps=1e-6)

    def update_ema_adapter(self, ema_adapter, current_adapter, decay):
        """Manually update EMA adapter parameters"""
        with torch.no_grad():
            # Iterate over each adapter module in the ModuleList
            for ema_module, current_module in zip(ema_adapter, current_adapter):
                for ema_param, current_param in zip(ema_module.parameters(), current_module.parameters()):
                    ema_param.data.mul_(decay).add_(current_param.data, alpha=1.0 - decay)

    def create_ema_adapter(self, adapter):
        """Create a deep copy of adapter for EMA"""
        return copy.deepcopy(adapter)

    def kl_div(self,p,q):
        p=F.softmax(p,dim=1)
        q=F.softmax(q,dim=1)
        kl = torch.mean(torch.sum(p * torch.log(p / q),dim=1))
        return kl
  
    def set_new_head(self, model, labels_to_be_added,task_id):
        len_new_nodes = len(labels_to_be_added)
        self.labels_in_head = np.concatenate((self.labels_in_head, labels_to_be_added))
        self.added_classes_in_cur_task.update(labels_to_be_added)
        self.head_timestamps = np.concatenate((self.head_timestamps, [task_id]*len_new_nodes))
        prev_weight, prev_bias = model.head.weight, model.head.bias
        prev_shape = prev_weight.shape # (class, dim)
        new_head = torch.nn.Linear(prev_shape[-1], prev_shape[0] + len_new_nodes)
    
        new_head.weight[:prev_weight.shape[0]].data.copy_(prev_weight)
        new_head.weight[prev_weight.shape[0]:].data.copy_(prev_weight[labels_to_be_added])
        new_head.bias[:prev_weight.shape[0]].data.copy_(prev_bias)
        new_head.bias[prev_weight.shape[0]:].data.copy_(prev_bias[labels_to_be_added])
        
        print(f"Added {len_new_nodes} nodes with label ({labels_to_be_added})")
        return new_head
    
    
    def inference_acc(self,model,data_loader,device):
        print("Start detecting labels to be added...")
        accuracy_per_label = []
        correct_pred_per_label = [0 for i in range(len(self.current_classes))]
        num_instance_per_label = [0 for i in range(len(self.current_classes))]
        
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(data_loader):
                if self.args.develop:
                    if batch_idx>200:
                        break
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                output = model(input)
                
                if output.shape[-1] > self.num_classes: # there are already added nodes till now
                    output,_,_ = self.get_max_label_logits(output, self.current_classes) # there are added nodes previously, but not in current task -> get maximum value and use it
                mask = self.current_classes
                not_mask = np.setdiff1d(np.arange(self.num_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = output.index_fill(dim=1, index=not_mask, value=float('-inf'))
                _, pred = torch.max(logits, 1)
                
                correct_predictions = (pred == target)
                for i, label in enumerate(self.current_classes):
                    mask = (target == label)
                    num_correct_pred = torch.sum(correct_predictions[mask])
                    correct_pred_per_label[i] += num_correct_pred.item()
                    num_instance_per_label[i] += sum(mask).item()
        for correct, num in zip (correct_pred_per_label, num_instance_per_label):
            accuracy_per_label.append(round(correct/num,2))
        return accuracy_per_label
    
    def detect_labels_to_be_added(self,inference_acc, thresholds=[]):
        labels_with_low_accuracy = []
        
        if self.args.d_threshold:
            for label,acc,thre in zip(self.current_classes, inference_acc,thresholds):
                if acc <= thre:
                    labels_with_low_accuracy.append(label)
        else: # static threshold
            for label,acc in zip(self.current_classes, inference_acc):
                if acc <= self.args.thre:
                    labels_with_low_accuracy.append(label)
                
        print(f"Labels whose node to be increased: {labels_with_low_accuracy}")
        return labels_with_low_accuracy
    
    def find_same_cluster_items(self,vec):
        if self.kmeans.n_clusters == 1:
            other_cluster_vecs = self.adapter_vec_array
            other_cluster_vecs = torch.tensor(other_cluster_vecs,dtype=torch.float32).to(self.device)
            same_cluster_vecs = None
        else:
            predicted_cluster = self.kmeans.predict(vec.unsqueeze(0).detach().cpu())[0]
            same_cluster_vecs = self.adapter_vec_array[self.cluster_assignments == predicted_cluster]
            other_cluster_vecs = self.adapter_vec_array[self.cluster_assignments != predicted_cluster]
            same_cluster_vecs = torch.tensor(same_cluster_vecs,dtype=torch.float32).to(self.device)
            other_cluster_vecs = torch.tensor(other_cluster_vecs,dtype=torch.float32).to(self.device)
        return same_cluster_vecs, other_cluster_vecs
    
    def calculate_l2_distance(self,diff_adapter, other):
        weights=[]
        for o in other:
            l2_distance = torch.norm(diff_adapter - o, p=2)
            weights.append(l2_distance.item())
        weights = torch.tensor(weights)
        weights = weights / torch.sum(weights) # summation-> 1
        return weights
    
    def train_one_epoch(self,model: torch.nn.Module, 
                        criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                        device: torch.device, epoch: int, max_norm: float = 0,
                        set_training_mode=True, task_id=-1, class_mask=None, ema_model = None, args = None,):

        model.train(set_training_mode)

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
        
        for batch_idx, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            if self.args.develop:
                if batch_idx>20:
                    break
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(input) # (bs, class + n)
            distill_loss=0
            if self.distill_head != None:
                feature = model.forward_features(input)[:,0]
                output_distill = self.distill_head(feature) 
                #! exclude added nodes in current task during distillation
                mask = torch.isin(torch.tensor(self.labels_in_head), torch.tensor(self.current_classes))
                cur_class_nodes = torch.where(mask)[0]#[:-len(self.added_classes_in_cur_task)] #! to be fixed
                m=torch.isin(torch.tensor(self.labels_in_head[cur_class_nodes]), torch.tensor(list(self.added_classes_in_cur_task)))
                distill_node_indices = self.labels_in_head[cur_class_nodes][~m]
                distill_loss = self.kl_div(output[:,distill_node_indices], output_distill[:,distill_node_indices])
               
        
            if output.shape[-1] > self.num_classes: # there are already added nodes till now 
                output,_,_ = self.get_max_label_logits(output, class_mask[task_id],slice=False)
                if len(self.added_classes_in_cur_task) > 0: # there are added nodes in current task
                    for added_class in self.added_classes_in_cur_task:
                        cur_node = np.where(self.labels_in_head == added_class)[0][-1] # the latest appended node
                        output[:, added_class] = output[:,cur_node]# replace logit value of added label
                    
                output = output[:, :self.num_classes]       
                
            # here is the trick to mask out classes of non-current tasks
            if args.train_mask and class_mask is not None:
                mask = class_mask[task_id]
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = output.index_fill(dim=1, index=not_mask, value=float('-inf'))

            loss = criterion(logits, target) # (bs, class), (bs)
            
           
            if self.args.use_cast_loss:
                if len(self.adapter_vec)> args.k: 
                    cur_adapters = model.get_adapter()
                    self.cur_adapters = self.flatten_parameters(cur_adapters)
                    diff_adapter = self.cur_adapters-self.prev_adapters
                    _, other = self.find_same_cluster_items(diff_adapter)
                    sim = 0
                    
                    # if self.args.ws:
                    weights = self.calculate_l2_distance(diff_adapter,other)
                    for o,w in zip(other,weights):
                        if self.args.norm_cast:
                            sim += w * torch.matmul(diff_adapter, o) / (torch.norm(diff_adapter)*torch.norm(o))
                        else:
                            sim += w * torch.matmul(diff_adapter, o)
                    # else:
                        # for o in other:
                            # sim += torch.matmul(diff_adapter, o)
                        # sim /= len(other)
                    orth_loss = args.beta * torch.abs(sim)
                    if self.args.use_cast_loss:  
                        if orth_loss>0:
                            loss += orth_loss
                    
            if self.args.IC:
                if distill_loss > 0:
                    loss += distill_loss
           
            acc1, acc3 = accuracy(logits, target, topk=(1, 3))

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            optimizer.zero_grad()
            
            loss.backward(retain_graph=True) 
            optimizer.step()
            torch.cuda.synchronize()
            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@3'].update(acc3.item(), n=input.shape[0])

            if ema_model is not None:
                current_adapter = model.get_adapter()
                self.update_ema_adapter(ema_model, current_adapter, self.args.ema_decay)
            
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def get_max_label_logits(self,output, class_mask,task_id=None, slice=True,target=None):
        #! Get max value for each label output
        correct=0
        total=0
        for label in range(self.num_classes): 
            label_nodes = np.where(self.labels_in_head == label)[0]
            output[:,label],max_index = torch.max(output[:,label_nodes],dim=1)
        if slice:
            output = output[:, :self.num_classes] # discard logits of added nodes
            
        return output,correct,total

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, data_loader,
                 device, task_id=-1, class_mask=None, ema_model=None, args=None, ):
        criterion = torch.nn.CrossEntropyLoss()

        # 1. Initialize lists to collect all predictions for F1/Precision/Recall
        all_targets = []
        all_preds = []

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test: [Task {}]'.format(task_id + 1)

        # Switch to evaluation mode
        model.eval()

        correct_sum, total_sum = 0, 0
        label_correct, label_total = np.zeros((self.class_group_size)), np.zeros((self.class_group_size))

        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
                if args.develop:
                    if batch_idx > 20:
                        break

                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # Compute output
                output = model(input)

                # Logic to handle added nodes/masking
                output, correct, total = self.get_max_label_logits(output, class_mask[task_id], task_id=task_id,
                                                                   target=target, slice=True)

                correct_sum += correct
                total_sum += total

                # Handle EMA Model
                if ema_model is not None:
                    output_ema = [output.softmax(dim=1)]
                    tmp_adapter = model.get_adapter()
                    model.put_adapter(ema_model)
                    output_ema_forward = model(input)
                    output_ema_forward, _, _ = self.get_max_label_logits(output_ema_forward, class_mask[task_id],
                                                                         slice=True)
                    output_ema.append(output_ema_forward.softmax(dim=1))
                    model.put_adapter(tmp_adapter)
                    output = torch.stack(output_ema, dim=-1).max(dim=-1)[0]

                loss = criterion(output, target)

                # Update specific label accuracy for d_threshold logic
                if self.args.d_threshold and self.current_task + 1 != self.args.num_tasks and self.current_task == task_id:
                    label_correct, label_total = self.update_acc_per_label(label_correct, label_total, output, target)

                acc1, acc3 = accuracy(output, target, topk=(1, 3))

                metric_logger.meters['Loss'].update(loss.item())
                metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@3'].update(acc3.item(), n=input.shape[0])

                # 2. Collect predictions for global metrics
                _, pred = output.max(1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

            if total_sum > 0:
                print(f"Max Pooling acc: {correct_sum / total_sum}")

            if self.args.d_threshold and task_id == self.current_task:
                domain_idx = int(self.label_train_count[self.current_classes][0])
                self.acc_per_label[self.current_classes, domain_idx] += np.round(label_correct / label_total,
                                                                                 decimals=3)
                print(self.label_train_count)
                print(self.acc_per_label)

        # 3. Compute Global Scikit-Learn Metrics
        macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        macro_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        macro_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)

        # Gather the stats from all processes
        metric_logger.synchronize_between_processes()

        print('* Acc@1 {top1.global_avg:.3f} Acc@3 {top3.global_avg:.3f} Loss {losses.global_avg:.3f} F1 {f1:.3f}'
              .format(top1=metric_logger.meters['Acc@1'], top3=metric_logger.meters['Acc@3'],
                      losses=metric_logger.meters['Loss'], f1=macro_f1))

        # 4. Return extended dictionary
        return {
            'Acc@1': metric_logger.meters['Acc@1'].global_avg,
            'Acc@3': metric_logger.meters['Acc@3'].global_avg,
            'Loss': metric_logger.meters['Loss'].global_avg,
            'F1': macro_f1,
            'Precision': macro_precision,
            'Recall': macro_recall
        }

    @torch.no_grad()
    def evaluate_till_now(self, model: torch.nn.Module, data_loader,
                          device, task_id=-1, class_mask=None, acc_matrix=None, ema_model=None, args=None, ):
        # 5. Initialize matrix to store [Acc@1, Acc@3, Loss, F1] for every task
        stat_matrix = np.zeros((4, args.num_tasks))

        for i in range(task_id + 1):
            test_stats = self.evaluate(model=model, data_loader=data_loader[i]['val'],
                                       device=device, task_id=i, class_mask=class_mask, ema_model=ema_model, args=args)

            stat_matrix[0, i] = test_stats['Acc@1']
            stat_matrix[1, i] = test_stats['Acc@3']
            stat_matrix[2, i] = test_stats['Loss']
            stat_matrix[3, i] = test_stats['F1']

            acc_matrix[i, task_id] = test_stats['Acc@1']

        # Compute averages over all tasks seen so far
        avg_stat = np.sum(stat_matrix, axis=1) / (task_id + 1)

        diagonal = np.diag(acc_matrix)

        result_str = "[Average metrics till task {}]\tAcc@1: {:.4f}\tAcc@3: {:.4f}\tLoss: {:.4f}\tF1: {:.4f}".format(
            task_id + 1, avg_stat[0], avg_stat[1], avg_stat[2], avg_stat[3])

        # 6. Compute Continual Learning Metrics (Forgetting, Backward Transfer)
        if task_id > 0:
            forgetting = np.mean((np.max(acc_matrix, axis=1) -
                                acc_matrix[:, task_id])[:task_id])
            backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

            result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
        print(result_str)
        return test_stats
    
    def flatten_parameters(self,modules):
        flattened_params = []
       
        for m in modules:
            params = list(m.parameters())
            flattened_params.extend(params) 
        return torch.cat([param.view(-1) for param in flattened_params])
    
    def cluster_adapters(self):
        k = self.args.k
        if len(self.adapter_vec) > k:
              
            self.adapter_vec_array = torch.stack(self.adapter_vec).detach().cpu().numpy().astype(float)
            self.kmeans = KMeans(n_clusters=k,n_init=10)
            self.kmeans.fit(self.adapter_vec_array)
            self.cluster_assignments = self.kmeans.labels_
            print("Cluster(shifts) Assignments:", self.cluster_assignments)
    
    
    def pre_train_epoch(self, model: torch.nn.Module, epoch: int = 0, task_id: int = 0, args = None,):
        if task_id == 0 or args.num_freeze_epochs < 1:
            return model
        
        if epoch == 0:
            for n, p in model.named_parameters():
                if 'adapter' in n:
                    p.requires_grad = False
            print('Freezing adapter parameters for {} epochs'.format(args.num_freeze_epochs))

        if epoch == args.num_freeze_epochs:
            for n, p in model.named_parameters():
                if 'adapter' in n:
                    p.requires_grad = True
            print('Unfreezing adapter parameters')        
        return model
    
    
    def pre_train_task(self, model, data_loader, device, task_id, args):
        self.current_task += 1
        self.current_class_group = int(min(self.class_mask[task_id])/self.class_group_size)
        self.class_group_list.append(self.current_class_group)
        self.current_classes = self.class_mask[task_id]
        
        print(f"\n\nTASK : {task_id}")
        self.added_classes_in_cur_task = set()  
        #! distillation
        if self.class_group_train_count[self.current_class_group]==0:
            self.distill_head=None
        else: # already seen classes
            if self.args.IC:
                self.distill_head = self.classifier_pool[self.current_class_group]
                inf_acc = self.inference_acc(model, data_loader, device)
                thresholds=[]
                if self.args.d_threshold:
                    count = self.class_group_train_count[self.current_class_group]
                    if count > 0:
                        average_accs = np.sum(self.acc_per_label[self.current_classes, :count], axis=1) / count
                    thresholds = self.args.gamma*(average_accs - inf_acc) / average_accs
                    thresholds = self.tanh(torch.tensor(thresholds)).tolist()
                    thresholds = [round(t,2) if t>self.args.thre else self.args.thre for t in thresholds]
                    print(f"Thresholds for class {self.current_classes[0]}~{self.current_classes[-1]} : {thresholds}")
                labels_to_be_added = self.detect_labels_to_be_added(inf_acc, thresholds)
                
                
                if len(labels_to_be_added) > 0: #! Add node to the classifier if needed
                    new_head = self.set_new_head(model, labels_to_be_added,task_id).to(device)
                    model.head = new_head
        optimizer = create_optimizer(args, model)

        with torch.no_grad():
            prev_adapters = model.get_adapter()
            self.prev_adapters = self.flatten_parameters(prev_adapters)
            self.prev_adapters.requires_grad=False
    
        if task_id==0:
            self.task_type_list.append("Initial")
            return model, optimizer
        
        prev_class = self.class_mask[task_id-1]
        prev_domain = self.domain_list[task_id-1]
        cur_class = self.class_mask[task_id]
        self.cur_domain = self.domain_list[task_id]
        
        if prev_class == cur_class:
            self.task_type = "DIL"
        else:
            self.task_type = "CIL"
        
        self.task_type_list.append(self.task_type)
        print(f"Current task : {self.task_type}")
        
        return model, optimizer


    def post_train_task(self,model: torch.nn.Module,task_id=-1):
        #! update classifier pool
        self.class_group_train_count[self.current_class_group]+=1
        self.classifier_pool[self.current_class_group]=copy.deepcopy(model.head)
        for c in self.classifier_pool:
                if c != None:
                    for p in c.parameters():
                        p.requires_grad=False
      
        cur_adapters = model.get_adapter()
        self.cur_adapters = self.flatten_parameters(cur_adapters)
        vector=self.cur_adapters - self.prev_adapters
        # if task_id>0: #? 1
        self.adapter_vec.append(vector)
        self.adapter_vec_label.append(self.task_type)
        self.cluster_adapters()
                 
    def train_and_evaluate(self, model: torch.nn.Module, criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                        lr_scheduler, device: torch.device, class_mask=None, args = None,):

        # create matrix to save end-of-task accuracies 
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        
        ema_model = None
        
        for task_id in range(args.num_tasks):
            # Create new optimizer for each task to clear optimizer status
            if task_id > 0 and args.reinit_optimizer:
                optimizer = create_optimizer(args, model)
            
            if task_id == 1 and len(args.adapt_blocks) > 0:
                # Create manual EMA adapter
                ema_model = self.create_ema_adapter(model.get_adapter())
                ema_model = ema_model.to(device)
                for param in ema_model.parameters():
                    param.requires_grad = False
            model, optimizer = self.pre_train_task(model, data_loader[task_id]['train'], device, task_id,args)
            for epoch in range(args.epochs):
                model = self.pre_train_epoch(model=model, epoch=epoch, task_id=task_id, args=args,)
                train_stats = self.train_one_epoch(model=model, criterion=criterion, 
                                            data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                            device=device, epoch=epoch, max_norm=args.clip_grad, 
                                            set_training_mode=True, task_id=task_id, class_mask=class_mask, ema_model=ema_model, args=args,)
              
                if lr_scheduler:
                    lr_scheduler.step(epoch)
                    
            self.post_train_task(model,task_id=task_id)
            if self.args.d_threshold:
                self.label_train_count[self.current_classes] += 1 
            test_stats = self.evaluate_till_now(model=model, data_loader=data_loader, device=device, 
                                        task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, ema_model=ema_model, args=args)
            if args.output_dir and utils.is_main_process():
                Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
                
                checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
                state_dict = {
                        'model': model.state_dict(),
                        'ema_model': {k: v.cpu() for k, v in ema_model.state_dict().items()} if ema_model is not None else None,
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }
                if args.sched is not None and args.sched != 'constant':
                    state_dict['lr_scheduler'] = lr_scheduler.state_dict()
                
                utils.save_on_master(state_dict, checkpoint_path)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,}

            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                    f.write(json.dumps(log_stats) + '\n')