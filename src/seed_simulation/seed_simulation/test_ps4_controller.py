
#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file presents an interface for interacting with the Playstation 4 Controller
# in Python. Simply plug your PS4 controller into your computer using USB and run this
# script!
#
# NOTE: I assume in this script that the only joystick plugged in is the PS4 controller.
#       if this is not the case, you will need to change the class accordingly.
#
# Copyright © 2015 Clay L. McLeod <clay.l.mcleod@gmail.com>
#
# Distributed under terms of the MIT license.

import os
import pprint
import pygame
import json
import pathlib

path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/params.json')
# Opening JSON file
with open(path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

max_steer = json_object["CBF_simple"]["max_steer"]  # [rad] max steering angle
max_speed = json_object["Car_model"]["max_speed"] # [m/s]
min_speed = json_object["Car_model"]["min_speed"] # [m/s]
max_acc = json_object["CBF_simple"]["max_acc"] 
min_acc = json_object["CBF_simple"]["min_acc"] 
car_max_acc = json_object["Controller"]["max_acc"]
car_min_acc = json_object["Controller"]["min_acc"]


class PS4Controller(object):
    """Class representing the PS4 controller. Pretty straightforward functionality."""

    controller = None
    axis_data = None
    button_data = None
    hat_data = None

    def init(self):
        """Initialize the joystick components"""
        
        pygame.init()
        pygame.joystick.init()
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()

    def listen(self):
        """Listen for events to happen"""
        
        if not self.axis_data:
            self.axis_data = {}

        if not self.button_data:
            self.button_data = {}
            for i in range(self.controller.get_numbuttons()):
                self.button_data[i] = False

        if not self.hat_data:
            self.hat_data = {}
            for i in range(self.controller.get_numhats()):
                self.hat_data[i] = (0, 0)

        # while True:
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                self.axis_data[event.axis] = round(event.value,2)
            elif event.type == pygame.JOYBUTTONDOWN:
                self.button_data[event.button] = True
            elif event.type == pygame.JOYBUTTONUP:
                self.button_data[event.button] = False
            elif event.type == pygame.JOYHATMOTION:
                self.hat_data[event.hat] = event.value

            # Insert your code on what you would like to happen for each event here!
            # In the current setup, I have the state simply printing out to the screen.
            
            # os.system('clear')
            # pprint.pprint(self.button_data)
            # pprint.pprint(self.axis_data)
            # pprint.pprint(self.hat_data)

    def mapping_ranges(self, input, input_start, input_end, output_start, output_end):
        output = output_start + ((output_end - output_start) / (input_end - input_start)) * (input - input_start)
        return output

    def control_input(self):
        self.listen()

        if len(self.axis_data)==0:
            throttle = 0.0
            delta = 0.0
        else:
            # Map left analog to throttle input
            if 1 in list(self.axis_data.keys()):
                throttle = self.mapping_ranges(-self.axis_data[1], -1.0, 1.0, min_acc, max_acc)
            else:
                throttle = 0.0

            # Map right analog to steering input
            if 3 in list(self.axis_data.keys()):
                delta = self.mapping_ranges(-self.axis_data[3], -1.0, 1.0, -max_steer, max_steer)
            else:
                delta = 0.0
        
        # pprint.pprint(f'Throttle: {throttle}, steering: {delta}\n')

        return throttle, delta

        


if __name__ == "__main__":
    ps4 = PS4Controller()
    ps4.init()
    ps4.listen()