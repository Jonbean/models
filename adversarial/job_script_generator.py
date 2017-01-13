discrim_dnn = ['300x150x70x35x1', '400x200x50x1', '512x256x1']
discrim_lr = ['0.0002', '0.0005','0.001']
generat_lr = ['0.0004', '0.001', '0.0015']
index = ['8E-3', '5E-3']

for j in range(len(discrim_dnn)):
    for l in range(len(gener_dnn)):
        for m in range(len(index)):
            f = open('../adversarial/FeatureNoiseAdv_DNN'+str(j)+str(l)+str(m)+'.job','w')
            f.write('#!/bin/bash\n')
            f.write('#$ -S /bin/bash\n')
            f.write('#$ -M jontsai@uchicago.edu\n')
            f.write('#$ -N feature_noiseAdv_DNN'+str(j)+str(l)+str(m)+'\n')
            f.write('#$ -m beasn\n')
            f.write('#$ -o feature_noiseAdv_DNN'+str(j)+str(l)+str(m)+'.out\n')
            f.write('#$ -e feature_noiseAdv_DNN'+str(j)+str(l)+str(m)+'.err\n')
            f.write('#$ -r n\n')
            f.write('#$ -cwd\n')
            f.write('SETTING1="300"\n')
            f.write('SETTING2="300"\n')
            f.write('SETTING3="50"\n')
            f.write('SETTING4="300x300"\n')
            f.write('SETTING5="concatenate"\n')
            f.write('SETTING6="0"\n')
            f.write('SETTING7="'+discrim_lr[l]+'"\n')            
            f.write('SETTING8="'+generat_lr[l]+'"\n')
            f.write('SETTING9="1.0"\n')
            f.write('SETTING10="last"\n')
            f.write('SETTING11="default"\n')
            f.write('SETTING12="DNN"\n')
            f.write('SETTING13="hinge"\n')
            f.write('SETTING14="normal"\n')
            f.write('SETTING15="'+discrim_dnn[j]+'"\n')
            f.write('SETTING16="1"\n')
            f.write('SETTING17="1"\n')
            f.write('SETTING18="'+index[m]+'"\n')
            f.write('SETTING19="feature_noiseAdv_DNN'+str(j)+str(l)+str(m)+'"\n')
            f.write('export OMP_NUM_THREADS=1\n')
            f.write('export OPENBLAS_NUM_THREADS=1\n')
            f.write('echo "Start - `date`"\n')
            f.write('/home-nfs/jontsai/anaconda/bin/python HierBLSTM_FeatureDiffNoiseAdv.py \\\n')
            f.write('$SETTING1 $SETTING2 $SETTING3 $SETTING4 $SETTING5 $SETTING6 $SETTING7 $SETTING8 $SETTING9 \
                     $SETTING10 $SETTING11 $SETTING12 $SETTING13 $SETTING14 $SETTING15 $SETTING16 $SETTING17 $SETTING18 $SETTING19\n')
            f.write('echo "End - `date`"\n')
            f.close()