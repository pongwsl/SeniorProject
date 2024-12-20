# playground.py
# created by pongwsl on dec 20, 2024
# This is only a playground, not save something to be permanent here.

import torch
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")