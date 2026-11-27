#!/usr/bin/env python3

from configparser import ConfigParser

import logging
import os

''' 
util module to read config file
''' 
CONFIG_FILE = 'config.ini'

def create_config(file):
    config = ConfigParser()
    config.read(file)
    return config

CONFIG = create_config(CONFIG_FILE)

def set_config(sectie, item, value):
    global CONFIG
    config = ConfigParser()
    config.read(CONFIG_FILE)
    config.set(sectie, item, value)
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)
    CONFIG = create_config(CONFIG_FILE)