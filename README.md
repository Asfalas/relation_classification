# relation_classification
  Attention based Bi-LSTM model   &  bert
## Dataset
  SemEval 2010 Task 8
## environment:
    python==3.6
    pytorch==1.5.0
    transformers==2.1.0
    tqdm
    nltk
## glove embedding
  place pretrained glove embedding in folder "code/vector_cache/" for att_bilstm model.
  example:   "relation_classification/code/vector_cache/glove.6B.200d.txt"

## run 
### att_bilstm:
  python main.py  --model att_bilstm 
                  --nums_train_epochs 50 
                  --train_batch_size 64
                  --learning_rate 0.005
                  --embedding_dim 200
                  --hidden_dim 400
                  --save_model ./model/
                  --output_dir ./out/
### bert:
  python main.py  --model bert 
                  --num_train_epochs 10
                  --learning_rate 0.00005 
                  --train_batch_size 64
                  --save_model ./model/
                  --output_dir ./out/
## metrics
### att_bilstm
  Micro-averaged result (excluding Other):
  P = 1822/2281 =  79.88%     R = 1822/2263 =  80.51%     F1 =  80.19%

  MACRO-averaged result (excluding Other):
  P =  79.43%	R =  79.25%	F1 =  79.10%
### bert
  Micro-averaged result (excluding Other):
  P = 2014/2339 =  86.11%     R = 2014/2263 =  89.00%     F1 =  87.53%

  MACRO-averaged result (excluding Other):
  P =  85.45% 		R =  88.93%		 F1 =  87.02%


