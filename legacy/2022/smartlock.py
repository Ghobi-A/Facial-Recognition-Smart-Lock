#! /usr/bin/env python3

'''
Program smartlock.py
Rev. 0
14-04-2022, first release

Description:

Face recognition with door lock
'''

# Import required packages
import logging
from logging import handlers
import configparser
import os
from view import View
from tkinter import *



class Controller():
    def __init__(self,parent):
        self.parent = parent

        self.view = View(self)      # initializes the view
  
        #start hardware self test in thread
        logging.info('start program')
        
        #repeat face recognition every 100 msec
        self.view.frame1.after(100, self.update)

    def update(self):
        '''
        read frames and analyse faces
        '''
        self.view.update()
        self.view.frame1.after(100, self.update)


def main(): 
    #setup logging
    rfh =  handlers.RotatingFileHandler(
        filename = 'logger.log',
        maxBytes = 100*1024,
        backupCount = 0
    )   
    logging.basicConfig( 
        level=logging.INFO,
        format=' %(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[ 
            rfh,
            logging.StreamHandler()
        ]   
    )
    logging.info('Start Application')

    #read configuration data
    config = configparser.ConfigParser()
    try: 
        config.read('config.ini')
    except Exception as e:
        logging.error('cannot read configuration file', e)

    #select display for output
    os.environ["DISPLAY"] = ":0.0"
    root = Tk()
    root.withdraw()
    root.title('Smart Lock')
    #full screen mode
    root.attributes("-fullscreen", True)

    app = Controller(root)

    root.bind("q", quit)

    root.mainloop()

if __name__ == "__main__":
    main()