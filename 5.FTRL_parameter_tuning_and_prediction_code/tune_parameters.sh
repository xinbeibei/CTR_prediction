
data_dir="/home/rcf-47/qjl_001/panfs/mini-project/new_data"
for alpha in 1 0.8 1.2
do
  for beta in 1 0.8 1.2
      do
	 for L1 in 0.001 1 #1000
	     do 
		for L2 in 0.001 #1 1000
		    do
        		JOB=alpha${alpha}_beta${beta}_L1${L1}_L2${L2}_tuning.pbs
        		cp job.sh $JOB
        		echo "cd $data_dir" >> $JOB
        		echo "matlab -nodisplay -nojvm -r alpha=$alpha\;beta=$beta\;L1=$L1\;L2=$L2\;< run_ftlr_full_crossvalidation.m" >> $JOB
        		qsub -b 1000 $JOB
        		sleep 1

			#matlab -nodisplay -nojvm -r alpha=$alpha\;beta=$beta\;L1=$L1\;L2=$L2\;<ftlr_full_crossvalidation.m> 3anda.sh.log
		    done
		done
	 done
done

