import os
# Parameters connected to spout sender
SENDER_NAME = os.getenv("SENDER_NAME", "Python Spout Sender")

WITH_SPOUT = bool(int(os.getenv("WITH_SPOUT", 1)))