python preprocess.py -train_src data/adv/train_src.txt \
                     -train_tgt data/adv/train_tgt.txt \
                     -valid_src data/adv/valid_src.txt \
                     -valid_tgt data/adv/valid_tgt.txt \
                     -save_data data/adv/AD \
                     -src_seq_length_trunc 15 \
                     -tgt_seq_length_trunc 15 \
                     -dynamic_dict \
                     -share_vocab \
                     -shard_size 100000

python -u train.py -data data/adv/AD \
                   -save_model models/adv \
                   -layers 6 \
                   -rnn_size 512 \
                   -word_vec_size 512 \
                   -max_grad_norm 0 \
                   -optim adam \
                   -encoder_type transformer \
                   -decoder_type transformer \
                   -position_encoding \
                   -dropout 0\.2 \
                   -param_init 0 \
                   -warmup_steps 8000 \
                   -valid_steps 1000 \
                   -save_checkpoint_steps 1000 \
                   -learning_rate 2 \
                   -decay_method noam \
                   -label_smoothing 0.1 \
                   -adam_beta2 0.998 \
                   -batch_size 256 \
                   -normalization tokens \
                   -max_generator_batches 2 \
                   -train_steps 20000 \
                   -accum_count 4 \
                   -share_embeddings \
                   -copy_attn \
                   -param_init_glorot \
                   -world_size 1 \
                   -gpu_ranks 0 \
                   -train_from models/adv_step_2000.pt
                   
python translate.py -gpu 0 \
                    -batch_size 20 \
                    -beam_size 10 \
                    -model models/adv_step_25000.pt \
                    -src data/adv/valid_src.txt \
                    -output data/adv_out.txt \
                    -min_length 35 \
                    -verbose \
                    -stepwise_penalty \
                    -coverage_penalty summary \
                    -beta 5 \
                    -length_penalty wu \
                    -alpha 0.9 \
                    -verbose \
                    -block_ngram_repeat 3 \
                    -ignore_when_blocking "." "</t>" "<t>"    
                    
      
