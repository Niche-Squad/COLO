

# post
for i in 0 1 2 3
do
    sbatch \
    --output=trial_0_$i.log\
    --error=trial_0_$i.err\
    --export=suffix=run$i\
    trial_0.sh
done