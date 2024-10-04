# Cart Pole
This repository contains the Model Predicitve Controller (MPC) to swing-up and balance a cart pole. [Casadi](https://web.casadi.org/) is used for solving the nonlinear optimization problem.

## Modeling
You can find a detailed modeling of the system in this [PDF](./cart_pole.pdf).

## Installation
First clone the repository,

`
git clone https://github.com/asarmadi/cart_pole.git
`

move to the directory

`
cd cart_pole
`

Make a python virtual environment

`
python -m venv env
`

Activate the environment. For windows

`
.\env\Scripts\activate
`

for linux

`
source ./env/bin/activate
`

Install the requirements

`
pip install -r requirements.txt
`

## Runing the experiment
To run the experiments

`
python main.py
`