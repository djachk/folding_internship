#!/usr/bin/sh
echo "hello world"
for E in 500 1000 3000
do
	for nu in 0.35 0.41 0.45
	do
	    python simul_beam_MPM_script_deux_cotes.py "${E}" "${nu}" 15	    
	done
done


