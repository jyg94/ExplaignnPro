out_path=$1
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=out/test.out
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -t 0-06:00:00
#SBATCH -o out/$out_path
#SBATCH --mem 256G

python -u run_on_turn.py config/convmix/explaignn.yml dev_turn
EOT
