import sys
import RPi.GPIO as GPIO
import time

triggerPIN = 13
GPIO.setmode(GPIO.BCM)
GPIO.setup(triggerPIN, GPIO.OUT)
buzzer = GPIO.PWM(triggerPIN, 1000)
buzzer.start(10)
time.sleep(0.2)
GPIO.cleanup()
sys.exit()