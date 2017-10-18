"""
I, Miguel Tasende, didn't write the code on this file. It didn't have any license, so I will use it for Udacity's
Machine Learning Nanodegree capstone project, with this note added.
This code was written by Marc Liyanage.
It was downloaded from https://github.com/liyanage/python-modules
I found it while searching for an efficient way to calculate the standard deviation, a sample at a time
(in an "online" fashion).
"""

# Based on http://www.johndcook.com/standard_deviation.html
import math


class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0
    
    def clear(self):
        self.n = 0
        
    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)
            
            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0
    
    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0
        
    def standard_deviation(self):
        return math.sqrt(self.variance())
