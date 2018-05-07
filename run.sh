

# Table 1

python main.py --algo nstepQlearning --game pong --total-steps 1000000 | tee -a log.txt
python main.py --algo A3C --game pong --total-steps 1000000 | tee -a log.txt

python main.py --algo nstepQlearning --game seaquest | tee -a log.txt
python main.py --algo A3C --game seaquest | tee -a log.txt

python main.py --algo nstepQlearning --game spaceinvaders | tee -a log.txt
python main.py --algo A3C --game spaceinvaders | tee -a log.txt

python main.py --algo nstepQlearning --game frosbite | tee -a log.txt
python main.py --algo A3C --game frosbite | tee -a log.txt

python main.py --algo nstepQlearning --game beamrider | tee -a log.txt
python main.py --algo A3C --game beamrider | tee -a log.txt

# Figure 2

python main.py --algo nstepQlearning --game pong --num-steps 1 --total-steps 1000000 | tee -a log.txt
python main.py --algo nstepQlearning --game pong --num-steps 10 --total-steps 1000000 | tee -a log.txt
python main.py --algo nstepQlearning --game pong --num-steps 50 --total-steps 1000000 | tee -a log.txt
python main.py --algo nstepQlearning --game pong --num-steps 1000000000 --total-steps 1000000 | tee -a log.txt

python main.py --algo A3C --game pong --num-steps 1 --total-steps 1000000 | tee -a log.txt
python main.py --algo A3C --game pong --num-steps 10 --total-steps 1000000 | tee -a log.txt
python main.py --algo A3C --game pong --num-steps 50 --total-steps 1000000 | tee -a log.txt
python main.py --algo A3C --game pong --num-steps 1000000000 --total-steps 1000000 | tee -a log.txt

python main.py --algo nstepQlearning --game seaquest --num-steps 1 | tee -a log.txt
python main.py --algo nstepQlearning --game seaquest --num-steps 10 | tee -a log.txt
python main.py --algo nstepQlearning --game seaquest --num-steps 50 | tee -a log.txt
python main.py --algo nstepQlearning --game seaquest --num-steps 1000000000 | tee -a log.txt

python main.py --algo A3C --game seaquest --num-steps 1 | tee -a log.txt
python main.py --algo A3C --game seaquest --num-steps 10 | tee -a log.txt
python main.py --algo A3C --game seaquest --num-steps 50 | tee -a log.txt
python main.py --algo A3C --game seaquest --num-steps 1000000000 | tee -a log.txt

python main.py --algo nstepQlearning --game spaceinvaders --num-steps 1 | tee -a log.txt
python main.py --algo nstepQlearning --game spaceinvaders --num-steps 10 | tee -a log.txt
python main.py --algo nstepQlearning --game spaceinvaders --num-steps 50 | tee -a log.txt
python main.py --algo nstepQlearning --game spaceinvaders --num-steps 1000000000 | tee -a log.txt

python main.py --algo A3C --game spaceinvaders --num-steps 1 | tee -a log.txt
python main.py --algo A3C --game spaceinvaders --num-steps 10 | tee -a log.txt
python main.py --algo A3C --game spaceinvaders --num-steps 50 | tee -a log.txt
python main.py --algo A3C --game spaceinvaders --num-steps 1000000000 | tee -a log.txt

