#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import time

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")
        elapsed_time = (time.perf_counter() - self._start_time)/60
        elapsed_min = np.floor(elapsed_time)
        elapsed_sec = (elapsed_time - elapsed_min)*60
        
        self._start_time = None
        print(f"Elapsed time: {elapsed_min:0.0f}min, {elapsed_sec: 0.0f}seconds")


# In[3]:





# https://realpython.com/python-timer/

# In[ ]:




