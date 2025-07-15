
ssh ault "rsync -av /scratch/dcamenis/measurements/ /users/dcamenis/measurements/"
rsync -av dcamenis@ault:/users/dcamenis/measurements ./cscs_measurements/
rsync -av --exclude='cifar-10-batches-py' --exclude='*.gz' --exclude='*.ncu-rep' --exclude='*.so' --exclude='*.pyc' ./code dcamenis@ault:/users/dcamenis
rsync -av ./job.sh dcamenis@ault:/users/dcamenis
rsync -av ./job2.sh dcamenis@ault:/users/dcamenis
rsync -av ./jobgpt.sh dcamenis@ault:/users/dcamenis