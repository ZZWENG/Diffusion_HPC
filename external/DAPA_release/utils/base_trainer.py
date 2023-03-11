from __future__ import division
import sys
import time

import torch
import numpy as np
from tqdm import tqdm
tqdm.monitor_interval = 0
# from torch.utils.tensorboard import SummaryWriter

from utils import CheckpointDataLoader, CheckpointSaver

class BaseTrainer(object):
    """Base class for Trainer objects.
    Takes care of checkpointing/logging/resuming training.
    """
    def __init__(self, options):
        self.options = options
        self.endtime = time.time() + self.options.time_to_run
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # override this function to define your model, optimizers etc.
        self.init_fn()
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)
        # self.summary_writer = SummaryWriter(self.options.summary_dir)

        self.checkpoint = None
        if self.options.resume and self.saver.exists_checkpoint(self.options.checkpoint):
            self.checkpoint = self.saver.load_checkpoint(self.models_dict, self.optimizers_dict, checkpoint_file=self.options.checkpoint)

        if self.checkpoint is None:
            self.epoch_count = 0
            self.step_count = 0
        else:
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['total_step_count']

    def load_pretrained(self, checkpoint_file=None):
        """Load a pretrained checkpoint.
        This is different from resuming training using --resume.
        """
        if checkpoint_file is not None:
            checkpoint = torch.load(checkpoint_file)
            for model in self.models_dict:
                if model in checkpoint:
                    self.models_dict[model].load_state_dict(checkpoint[model], strict=False)
                    print('Checkpoint loaded')

    def train(self):
        """Training process."""
        # Run training for num_epochs epochs
        for epoch in tqdm(range(self.epoch_count, self.options.num_epochs), total=self.options.num_epochs, initial=self.epoch_count):
            # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
 
            self.adapt_mode = self.options.ft_dataset != ''
            self.test()

            if self.adapt_mode:
                 self.checkpoint = None  # for adaptation, load the entire dataset.
            train_data_loader = CheckpointDataLoader(self.train_ds,checkpoint=self.checkpoint,
                                                     batch_size=self.options.batch_size,
                                                     num_workers=self.options.num_workers,
                                                     pin_memory=self.options.pin_memory,
                                                     shuffle=self.options.shuffle_train)
            # Iterate over all batches in an epoch
            for step, batch in enumerate(tqdm(train_data_loader, desc='Epoch '+str(epoch),
                                              total=len(self.train_ds) // self.options.batch_size,
                                              initial=train_data_loader.checkpoint_batch_idx),
                                         train_data_loader.checkpoint_batch_idx):
                if time.time() < self.endtime:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                    out = self.train_step(batch)
                    self.step_count += 1
                    # Tensorboard logging every summary_steps steps
                    if self.step_count % self.options.summary_steps == 0:
                        self.train_summaries(batch, *out)
                    # Save checkpoint every checkpoint_steps steps
                    if self.step_count % self.options.checkpoint_steps == 0:
                        self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step+1, self.options.batch_size, train_data_loader.sampler.dataset_perm, self.step_count, "")
                        tqdm.write('Checkpoint saved')
                        
                    if self.options.pretrain:
                        fake_batch = self.get_fake(out[0])
                        if self.step_count % self.options.summary_steps == 0:
                            fake_batch = {k: v.to(self.device).detach() if isinstance(v, torch.Tensor) else v for k,v in fake_batch.items()}
                            fake_out = self.train_step(fake_batch, False)
                            self.fake_summaries(fake_batch, *fake_out)
                    
                    elif self.adapt_mode and not self.options.adapt_baseline:
                        fake_batch = self.get_fake(out[0])
                        fake_batch = {k: v.to(self.device).detach() if isinstance(v, torch.Tensor) else v for k,v in fake_batch.items()}
                        fake_out = self.train_step(fake_batch, False)

                        if self.step_count % self.options.summary_steps == 0:
                            self.fake_summaries(fake_batch, *fake_out)
                            
                    elif self.adapt_mode and self.options.adapt_baseline:
                        # as baseline, do not generate fake data at all. Simply train on real images.
                        pass
                            

                    # Run validation every test_steps steps
                    if self.step_count % self.options.test_steps == 0:
                        self.test()
                else:
                    tqdm.write('Timeout reached')
                    self.finalize()
                    self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step, self.options.batch_size, train_data_loader.sampler.dataset_perm, self.step_count, "") 
                    tqdm.write('Checkpoint saved')
                    sys.exit(0)

            # load a checkpoint only on startup, for the next epochs
            # just iterate over the dataset as usual
            self.checkpoint=None
            # save checkpoint after each epoch
            if (epoch+1) % 10 == 0:
                # self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.step_count)
                self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.options.batch_size, None, self.step_count, "")
        return

    # The following methods (with the possible exception of test) have to be implemented in the derived classes
    def init_fn(self):
        raise NotImplementedError('You need to provide an _init_fn method')

    def train_step(self, input_batch):
        raise NotImplementedError('You need to provide a _train_step method')

    def train_summaries(self, input_batch):
        raise NotImplementedError('You need to provide a _train_summaries method')

    def test(self):
        self.model.eval()
        import eval_diffusion as eval_
        
        ft_datasets = self.options.ft_dataset
        eval_dataset = [ds for ds in ft_datasets.split(',') if ds.endswith('real')]

        if len(eval_dataset) == 0:
            if ft_datasets.endswith('zs_syn'):
                eval_dataset.append(ft_datasets.replace('zs_syn', '_real'))  # turn `skizs_syn` to `ski_real`
            if ft_datasets.endswith('_syn'):
                eval_dataset.append(ft_datasets.replace('_syn', '_real'))  # turn `skizs_syn` to `ski_real`
        # eval_dataset = ['polevault_real']

        assert len(eval_dataset) == 1
        eval_types = ['2d']

        if eval_dataset[0].startswith('ski'):
            eval_dataset *= 2
            eval_types.append('3d_keypoints')

        print('!!! Running evaluation on ', eval_dataset)
        for ds_name, eval_type in zip(eval_dataset, eval_types):
            print(f'Evaluating on {ds_name} with eval_type {eval_type}')
            dataset = eval_.BaseDataset(None, ds_name, is_train=False)
            print('Dataset length:', len(dataset))
            metric = eval_.run_evaluation(self.model, dataset, None,
                batch_size=self.options.batch_size, eval_type=eval_type,
                shuffle=False)
            self.summary_writer.log({f'metrics/{eval_type}_{ds_name}': metric})

        # if 'pole_vault' in self.options.ft_dataset:
        #     ds_names = ['pole_vault_sportscap', 'vault_sportscap']
        #     eval_types = ['2d', '2d']
        #     for ds_name, eval_type in zip(ds_names, eval_types):
        #         print('Evaluating on ', ds_name, '...')
        #         dataset = eval_.BaseDataset(None, ds_name, is_train=False)
        #         print('Dataset length:', len(dataset))
        #         metric = eval_.run_evaluation(self.model, dataset, None,
        #             batch_size=self.options.batch_size, eval_type=eval_type,
        #             shuffle=False)
        #         self.summary_writer.log({f'metrics/{ds_name}': metric})
        
        # if 'ski' in self.options.ft_dataset:
        #     ds_names = ['ski_test', 'ski_test']
        #     eval_types = ['2d', '3d']
        #     for ds_name, eval_type in zip(ds_names, eval_types):
        #         print('Evaluating on ', ds_name, '...')
        #         dataset = eval_.BaseDataset(None, ds_name, is_train=False)
        #         print('Dataset length:', len(dataset))
        #         metric = eval_.run_evaluation(self.model, dataset, None,
        #             batch_size=self.options.batch_size, eval_type=eval_type,
        #             shuffle=False)
        #         self.summary_writer.log({f'metrics/{ds_name}': metric})

        self.model.train()
