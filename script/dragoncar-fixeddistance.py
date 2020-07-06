from gpiozero import DistanceSensor, Robot
from time import sleep



robot = Robot(left=(5, 6), right=(23, 24))
chaoshengbo = DistanceSensor(27, 22)

while True:
    if(chaoshengbo.distance > 0.3):
        print("qian")
        robot.forward(speed=0.3)
    elif(chaoshengbo.distance < 0.2):
        print("hou")
        robot.backward(speed=0.3)
    else:
        print("ting")
        robot.stop()

    sleep(0.1)
