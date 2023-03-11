from utils import TrainOptions
from train import Trainer, PreTrainer, AdaptTrainer

if __name__ == '__main__':
    options = TrainOptions().parse_args()
    
    if options.pretrain and (options.ft_dataset != ''):
        raise Exception('Either set --pretrain or the target dataset, do not set both!')
    
    if options.ft_dataset != '' and not options.adapt_baseline:
        trainer = AdaptTrainer(options)
    elif options.pretrain:
        trainer = PreTrainer(options)
    else:
        trainer = Trainer(options)
    trainer.train()
    