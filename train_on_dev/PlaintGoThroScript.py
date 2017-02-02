
for l in range(5):
    f = open('./PlaintGT'+str(l)+'.job','w')
    f.write('#!/bin/bash\n')
    f.write('#$ -S /bin/bash\n')
    f.write('#$ -M jontsai@uchicago.edu\n')
    f.write('#$ -N PlaintGT'+str(l)+'\n')
    f.write('#$ -m beasn\n')
    f.write('#$ -o PlaintGT'+str(l)+'.out\n')
    f.write('#$ -e PlaintGT'+str(l)+'.err\n')
    f.write('#$ -r n\n')
    f.write('#$ -cwd\n')
    f.write('SETTING1="300"\n')
    f.write('SETTING3="50"\n')
    f.write('SETTING4="'+str(l)+'"\n')          

    f.write('export OMP_NUM_THREADS=1\n')
    f.write('export OPENBLAS_NUM_THREADS=1\n')
    f.write('echo "Start - `date`"\n')
    f.write('/home-nfs/jontsai/anaconda/bin/python plaint_rnn.py \\\n')
    f.write('$SETTING1 $SETTING2 $SETTING3 $SETTING4\n')
    f.write('echo "End - `date`"\n')
    f.close()

