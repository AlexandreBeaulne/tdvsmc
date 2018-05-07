

# Table 1

echo "`date` n-step Q-learning pong 20"
python -u main.py --algo nstepQlearning --game pong --total-steps 2500000 --seed 3443 | tee -a log.txt

echo "`date` A3C pong 20"
python -u main.py --algo A3C --game pong --total-steps 2500000 | tee -a log.txt

echo "`date` n-step Q-learning seaquest 20"
python -u main.py --algo nstepQlearning --game seaquest | tee -a log.txt

echo "`date` A3C seaquest 20"
python -u main.py --algo A3C --game seaquest | tee -a log.txt

echo "`date` n-step Q-learning spaceinvaders 20"
python -u main.py --algo nstepQlearning --game spaceinvaders | tee -a log.txt

echo "`date` A3C spaceinvaders 20"
python -u main.py --algo A3C --game spaceinvaders | tee -a log.txt

echo "`date` n-step Q-learning frostbite 20"
python -u main.py --algo nstepQlearning --game frosbite | tee -a log.txt

echo "`date` A3C frostbite 20"
python -u main.py --algo A3C --game frosbite | tee -a log.txt

echo "`date` n-step Q-learning beamrider 20"
python -u main.py --algo nstepQlearning --game beamrider | tee -a log.txt

echo "`date` A3C beamrider 20"
python -u main.py --algo A3C --game beamrider | tee -a log.txt

# Figure 2

echo "`date` n-step Q-learning pong 1" | tee -a log.txt
python -u main.py --algo nstepQlearning --game pong --num-steps 2 --total-steps 3000000 | tee -a log.txt
echo "`date` n-step Q-learning pong 10" | tee -a log.txt
python -u main.py --algo nstepQlearning --game pong --num-steps 10 --total-steps 3000000 | tee -a log.txt
echo "`date` n-step Q-learning pong 50" | tee -a log.txt
python -u main.py --algo nstepQlearning --game pong --num-steps 50 --total-steps 3000000 | tee -a log.txt
echo "`date` n-step Q-learning pong inf" | tee -a log.txt
python -u main.py --algo nstepQlearning --game pong --num-steps 1000000000 --total-steps 3000000 | tee -a log.txt

echo "`date` A3C pong 1" | tee -a log.txt
python -u main.py --algo A3C --game pong --num-steps 2 --total-steps 3000000 | tee -a log.txt
echo "`date` A3C pong 10" | tee -a log.txt
python -u main.py --algo A3C --game pong --num-steps 10 --total-steps 3000000 | tee -a log.txt
echo "`date` A3C pong 50" | tee -a log.txt
python -u main.py --algo A3C --game pong --num-steps 50 --total-steps 3000000 | tee -a log.txt
echo "`date` A3C pong inf" | tee -a log.txt
python -u main.py --algo A3C --game pong --num-steps 1000000000 --total-steps 3000000 | tee -a log.txt

echo "`date` n-step Q-learning seaquest 1"
python -u main.py --algo nstepQlearning --game seaquest --num-steps 1 | tee -a log.txt
echo "`date` n-step Q-learning seaquest 10"
python -u main.py --algo nstepQlearning --game seaquest --num-steps 10 | tee -a log.txt
echo "`date` n-step Q-learning seaquest 50"
python -u main.py --algo nstepQlearning --game seaquest --num-steps 50 | tee -a log.txt
echo "`date` n-step Q-learning seaquest inf"
python -u main.py --algo nstepQlearning --game seaquest --num-steps 1000000000 | tee -a log.txt

echo "`date` A3C seaquest 1"
python -u main.py --algo A3C --game seaquest --num-steps 1 | tee -a log.txt
echo "`date` A3C seaquest 10"
python -u main.py --algo A3C --game seaquest --num-steps 10 | tee -a log.txt
echo "`date` A3C seaquest 50"
python -u main.py --algo A3C --game seaquest --num-steps 50 | tee -a log.txt
echo "`date` A3C seaquest inf"
python -u main.py --algo A3C --game seaquest --num-steps 1000000000 | tee -a log.txt

echo "`date` n-step Q-learning spaceinvaders 1"
python -u main.py --algo nstepQlearning --game spaceinvaders --num-steps 1 | tee -a log.txt
echo "`date` n-step Q-learning spaceinvaders 10"
python -u main.py --algo nstepQlearning --game spaceinvaders --num-steps 10 | tee -a log.txt
echo "`date` n-step Q-learning spaceinvaders 50"
python -u main.py --algo nstepQlearning --game spaceinvaders --num-steps 50 | tee -a log.txt
echo "`date` n-step Q-learning spaceinvaders inf"
python -u main.py --algo nstepQlearning --game spaceinvaders --num-steps 1000000000 | tee -a log.txt

echo "`date` A3C spaceinvaders 1"
python -u main.py --algo A3C --game spaceinvaders --num-steps 1 | tee -a log.txt
echo "`date` A3C spaceinvaders 10"
python -u main.py --algo A3C --game spaceinvaders --num-steps 10 | tee -a log.txt
echo "`date` A3C spaceinvaders 50"
python -u main.py --algo A3C --game spaceinvaders --num-steps 50 | tee -a log.txt
echo "`date` A3C spaceinvaders inf"
python -u main.py --algo A3C --game spaceinvaders --num-steps 1000000000 | tee -a log.txt

