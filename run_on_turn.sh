i=${2:-0}
j=${3:-none}
config=$1
gnn_string=$(python find_path.py config/convmix/$config.yml)
out_path=$(python find_path.py config/convmix/$config.yml $i)
echo out/$gnn_string.out
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=$gnn_string
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -t 0-06:00:00
#SBATCH -o out/$gnn_string.out
#SBATCH --mem 256G

python -u run_on_turn.py config/convmix/$config.yml $out_path $i $j
EOT