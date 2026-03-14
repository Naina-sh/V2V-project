import socket
import threading
import time
import json
import random
import uuid

from crypto_aes import encrypt, decrypt

PORT = 5000
BROADCAST = ("255.255.255.255", PORT)

car_id = str(uuid.uuid4())[:4]

# Static vehicle info
vehicle_type = random.choice(["Car", "Truck", "Bike", "Ambulance"])
priority_map = {
    "Car": 2,
    "Truck": 2,
    "Bike": 1,
    "Ambulance": 3
}
priority = priority_map[vehicle_type]

print("CAR ID:", car_id, vehicle_type, "Priority:", priority)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
sock.bind(("", PORT))

last_speed = 60


# -------- SEND LOOP --------
def send_loop():
    global last_speed

    while True:

        # Position
        pos_x = random.randint(0, 1000)
        pos_y = random.randint(0, 1000)

        # Weather
        weather = random.choice(["Clear", "Rain", "Fog"])

        # Speed
        speed = random.randint(40, 90)

        # Brake
        brake = random.choice([0, 1])

        # Network features
        acc = random.uniform(-3, 3)
        delay = random.uniform(0.1, 1.5)
        msg_rate = random.randint(5, 40)
        duplicate = random.randint(0, 3)

        # Attack simulation
        is_attack = 0

        if random.random() < 0.2:

            is_attack = 1

            attack = random.choice(["spoof", "dos", "replay"])

            if attack == "spoof":
                speed = random.randint(200, 350)

            elif attack == "dos":
                delay = random.uniform(2, 4)

            elif attack == "replay":
                speed = last_speed

        last_speed = speed


        # Data Packet
        data = {
            "car": car_id,
            "type": vehicle_type,
            "priority": priority,

            "x": pos_x,
            "y": pos_y,

            "weather": weather,

            "speed": speed,
            "brake": brake,

            "acc": acc,
            "delay": delay,
            "msg_rate": msg_rate,
            "duplicate": duplicate,

            "attack": is_attack
        }

        print("SENDING:", data)

        msg = json.dumps(data)
        enc = encrypt(msg)

        sock.sendto(enc, BROADCAST)

        time.sleep(1)


# -------- RECEIVE LOOP --------
def listen_loop():
    while True:

        data, addr = sock.recvfrom(4096)

        try:
            plain = decrypt(data)
            obj = json.loads(plain)

            if obj["car"] != car_id:

                print(f"[{car_id}] Received:", obj)

                with open("v2v_smart_dataset.csv", "a") as f:

                    f.write(
                        f"{obj['type']},{obj['priority']},"
                        f"{obj['x']},{obj['y']},"
                        f"{obj['weather']},"
                        f"{obj['speed']},{obj['brake']},"
                        f"{obj['acc']},{obj['delay']},"
                        f"{obj['msg_rate']},{obj['duplicate']},"
                        f"{obj['attack']}\n"
                    )

        except Exception as e:
            print("ERROR:", e)



# -------- START --------
threading.Thread(target=send_loop, daemon=True).start()
threading.Thread(target=listen_loop, daemon=True).start()

while True:
    time.sleep(1)