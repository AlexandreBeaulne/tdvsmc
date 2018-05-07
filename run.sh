

# Table 1

python main.py --algo nstepQlearning --game pong
python main.py --algo nstepQlearning --game seaquest
python main.py --algo nstepQlearning --game spaceinvaders
python main.py --algo nstepQlearning --game frosbite
python main.py --algo nstepQlearning --game beamrider

python main.py --algo A3C --game pong
python main.py --algo A3C --game seaquest
python main.py --algo A3C --game spaceinvaders
python main.py --algo A3C --game frosbite
python main.py --algo A3C --game beamrider

# Figure 2

python main.py --algo nstepQlearning --game pong --num-steps 1
python main.py --algo nstepQlearning --game pong --num-steps 10
python main.py --algo nstepQlearning --game pong --num-steps 50
python main.py --algo nstepQlearning --game pong --num-steps 1000000000

python main.py --algo A3C --game pong --num-steps 1
python main.py --algo A3C --game pong --num-steps 10
python main.py --algo A3C --game pong --num-steps 50
python main.py --algo A3C --game pong --num-steps 1000000000

python main.py --algo nstepQlearning --game seaquest --num-steps 1
python main.py --algo nstepQlearning --game seaquest --num-steps 10
python main.py --algo nstepQlearning --game seaquest --num-steps 50
python main.py --algo nstepQlearning --game seaquest --num-steps 1000000000

python main.py --algo A3C --game seaquest --num-steps 1
python main.py --algo A3C --game seaquest --num-steps 10
python main.py --algo A3C --game seaquest --num-steps 50
python main.py --algo A3C --game seaquest --num-steps 1000000000

python main.py --algo nstepQlearning --game spaceinvaders --num-steps 1
python main.py --algo nstepQlearning --game spaceinvaders --num-steps 10
python main.py --algo nstepQlearning --game spaceinvaders --num-steps 50
python main.py --algo nstepQlearning --game spaceinvaders --num-steps 1000000000

python main.py --algo A3C --game spaceinvaders --num-steps 1
python main.py --algo A3C --game spaceinvaders --num-steps 10
python main.py --algo A3C --game spaceinvaders --num-steps 50
python main.py --algo A3C --game spaceinvaders --num-steps 1000000000

