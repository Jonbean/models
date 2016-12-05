batch_size = [300, 350]
Adam_LR_ls = [0.001, 0.0005]
sample_strat = ['plot_cos', 'end_cos', 'combine_cos']
for i in range(len(batch_size)):
    for j in range(len(Adam_LR_ls)):
        for k in range(len(sample_strat)):
            f = open('../hierachical_rnn/HierBGRU_simiCosine'+str(i)+str(j)+str(k)+'.job','w')
            f.write('#!/bin/bash\n')
            f.write('#$ -S /bin/bash\n')
            f.write('#$ -M jontsai@uchicago.edu\n')
            f.write('#$ -N HierBGRU_simiCosine'+str(i)+str(j)+str(k)+'\n')
            f.write('#$ -m beasn\n')
            f.write('#$ -o HierBGRU_simiCosine'+str(i)+str(j)+str(k)+'.out\n')
            f.write('#$ -e HierBGRU_simiCosine'+str(i)+str(j)+str(k)+'.err\n')
            f.write('#$ -r n\n')
            f.write('#$ -cwd\n')
            f.write('SETTING1="300"\n')
            f.write('SETTING2="'+str(batch_size[i])+'"\n')
            f.write('SETTING3="0"\n')
            f.write('SETTING4="'+str(Adam_LR_ls[j])+'"\n')
            f.write('SETTING5="1.0"\n')
            f.write('SETTING6="last"\n')
            f.write('SETTING7="'+sample_strat[k]+'"\n')
            f.write('export OMP_NUM_THREADS=1\n')
            f.write('export OPENBLAS_NUM_THREADS=1\n')
            f.write('echo "Start - `date`"\n')
            f.write('/home-nfs/jontsai/anaconda/bin/python HierBGRU_simiCosine.py \\\n')
            f.write('$SETTING1 $SETTING2 $SETTING3 $SETTING4 $SETTING5 $SETTING6 $SETTING7\n')
            f.write('echo "End - `date`"\n')
            f.close()

