from flask import Flask, render_template,request,make_response,session, send_file, url_for,request
from flask_session import Session
import plotly
import plotly.graph_objs as go
import mysql.connector
from mysql.connector import Error
import sys
import pandas as pd
import numpy as np
import json  #json request
from werkzeug.utils import secure_filename
import os
import csv #reading csv
import geocoder
from random import randint
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os
import random
import shutil
import datetime
import rsa
import numpy as np
import time

IP = [58, 50, 42, 34, 26, 18, 10, 2,
      60, 52, 44, 36, 28, 20, 12, 4,
      62, 54, 46, 38, 30, 22, 14, 6,
      64, 56, 48, 40, 32, 24, 16, 8,
      57, 49, 41, 33, 25, 17, 9, 1,
      59, 51, 43, 35, 27, 19, 11, 3,
      61, 53, 45, 37, 29, 21, 13, 5,
      63, 55, 47, 39, 31, 23, 15, 7]

FP = [40, 8, 48, 16, 56, 24, 64, 32,
        39, 7, 47, 15, 55, 23, 63, 31,
        38, 6, 46, 14, 54, 22, 62, 30,
        37, 5, 45, 13, 53, 21, 61, 29,
        36, 4, 44, 12, 52, 20, 60, 28,
        35, 3, 43, 11, 51, 19, 59, 27,
        34, 2, 42, 10, 50, 18, 58, 26,
        33, 1, 41, 9, 49, 17, 57, 25]

EBox = [32,1,2,3,4,5,
            4,5,6,7,8,9,
            8,9,10,11,12,13,
            12,13,14,15,16,17,
            16,17,18,19,20,21,
            20,21,22,23,24,25,
            24,25,26,27,28,29,
            28,29,30,31,32,1]

SBox =[
		# S1
		[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7,
		 0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8,
		 4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0,
		 15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],

		# S2
		[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10,
		 3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5,
		 0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15,
		 13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],

		# S3
		[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8,
		 13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1,
		 13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7,
		 1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],

		# S4
		[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15,
		 13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9,
		 10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4,
		 3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],

		# S5
		[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9,
		 14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6,
		 4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14,
		 11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],

		# S6
		[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11,
		 10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8,
		 9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6,
		 4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],

		# S7
		[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1,
		 13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6,
		 1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2,
		 6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],

		# S8
		[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7,
		 1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2,
		 7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8,
		 2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
	]

F_PBox = [16, 7, 20, 21, 29, 12, 28, 17,
              1, 15, 23, 26, 5, 18, 31, 10,
              2, 8, 24, 14, 32, 27, 3, 9,
              19, 13, 30, 6, 22, 11, 4, 25 ]

key_PBox = [14,    17,   11,    24,     1,    5,
                  3,    28,   15,     6,    21,   10,
                 23,    19,   12,     4,    26,    8,
                 16,     7,   27,    20,    13,    2,
                 41,    52,   31,    37,    47,   55,
                 30,    40,   51,    45,    33,   48,
                 44,    49,   39,    56,    34,  53,
                 46,    42,   50,    36,    29,   32]


def xor(left,xorstream):
    xorresult = np.logical_xor(left,xorstream)

    xorresult  = xorresult.astype(int)

    return xorresult

def E_box(right):
    expanded = np.empty(48)
    j = 0
    for i in EBox:
        expanded[j] = right[i - 1]
        j += 1
    expanded = list(map(int,expanded))
    expanded = np.array(expanded)
    return expanded

#clean this code please (sboxlookup)
def sboxloopup(sinput,x):
    tableno = x - 1
    row = int((np.array2string(sinput[0]) + np.array2string(sinput[5])),2)

    # make this part of the code better
    column = sinput[1:5]
    column = np.array2string(column)
    column = column[1:8].replace(" ", "")
    column = int(column,2)
    # print(column,"column")

    elementno = (16 * row) + column
    soutput = SBox[tableno][elementno]
    soutput = list(np.binary_repr(soutput, width=4))
    #converting to list twice seems redundant but seems to be the only simple way as map always returns map object
    soutput= np.array(list(map(int, soutput)))
    return soutput

def sbox(sboxin):
#takes 48 bit input and return 32 bit
    sboxin1 = sboxin[0:6]
    sboxout1 = sboxloopup(sboxin1,1)
    sboxin2 = sboxin[6:12]
    sboxout2 = sboxloopup(sboxin2,2)
    sboxin3 = sboxin[12:18]
    sboxout3 = sboxloopup(sboxin3, 3)
    sboxin4 = sboxin[18:24]
    sboxout4 = sboxloopup(sboxin4, 4)
    sboxin5 = sboxin[24:30]
    sboxout5 = sboxloopup(sboxin5, 5)
    sboxin6 = sboxin[30:36]
    sboxout6 = sboxloopup(sboxin6, 6)
    sboxin7 = sboxin[36:42]
    sboxout7 = sboxloopup(sboxin7, 7)
    sboxin8 = sboxin[42:48]
    sboxout8 = sboxloopup(sboxin8, 8)
    sboxout = np.concatenate([sboxout1,sboxout2,sboxout3,sboxout4,sboxout5,sboxout6,sboxout7,sboxout8])
    return sboxout

def f_permute(topermute):
    permuted= np.empty(32)
    j = 0
    for i in F_PBox:
        permuted[j] = topermute[i - 1]
        j += 1
    return permuted

def f_function(right,rkey):
    expanded = E_box(right)
    xored = xor(expanded,rkey)
    sboxed = sbox(xored)
    xorstream = f_permute(sboxed)
    return xorstream

def round(data,rkey):
    l0 = data[0:32]
    r0 = data[32:64]
    xorstream = f_function(r0,rkey)
    r1 = xor(l0,xorstream)
    l1 = r0
    returndata = np.empty_like(data)
    returndata[0:32] = l1
    returndata[32:64] = r1
    return(returndata)

def permutation(data,x):
    #intial and final permutation conditional based on other passed value
    permute1 = np.empty_like(IP)
    if x == 0:
        j = 0
        for i in IP:
            permute1[j] = data[i-1]
            j += 1
        return(permute1)
    else:
        permute2 = np.empty_like(FP)
        k = 0
        for l in FP:
            permute2[k] = data[l-1]
            k += 1
        return(permute2)

def userinput():
    keyinp = input("Enter the key bits (56 bits) seperated by space " "").strip().split()
    datainp = input("Enter the data bits (64) to encrypt or decrypt seperated by space " "").strip().split()
    #change to 56 later
    lenofkey = 56
    #change to 64 later
    lenofdata = 64
    if len(datainp) == lenofdata and len(keyinp) == lenofkey:
        print("data entry accepted, data loaded succesfully")
        print("key entry accepted, key loaded succesfully")
    else:
        while len(datainp) != lenofdata:
            print("length of data entered ",len(datainp))
            datainp = input("Error in entered data. Enter the data (64 bits) to encrypt or decrypt seperated by space " "").strip().split()

        print("data entry accepted, data loaded succesfully")
        while len(keyinp) != lenofkey:
            print("length of key entered ", len(keyinp))
            keyinp = input("Error in entered key. Enter the key (56 bits) to encrypt or decrypt seperated by space " "").strip().split()
        print("key entry accepted, key loaded succesfully")
#also add functionality to accept 64 bit keys instead of 54
    return keyinp,datainp


def keyshift(toshift,n):
    if (n == 1) or (n == 2) or (n == 9) or (n == 16):
        toshift= np.roll(toshift,-1)
        return toshift
    else:
        toshift = np.roll(toshift, -2)
        return toshift

def keypermute(key16):
    keypermuted = np.empty([16,48])
    l = 0
    for k in key16:
        j = 0
        for i in key_PBox:
            keypermuted[l][j] = k[i - 1]
            j += 1
        l += 1
    return keypermuted

#
def keyschedule(key):
    left = key[0:28]
    right = key[28:56]
    shifted = np.zeros(56)
    key16 = np.zeros([16,56])
    for i in range(1,17):
        shifted[0:28] = keyshift(left,i)
        shifted[28:56] = keyshift(right,i)
        left = shifted[0:28]
        right = shifted[28:56]
#add shifted to key16 and return key16
        key16[i - 1] = shifted
#key16 is the final shifted 16 key pair now to permute
    key16 = keypermute(key16)
    key16 = [list(map(int, x)) for x in key16]
    key16 = np.array(key16)
    return key16


Sbox = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
)

InvSbox = (
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
)


# learnt from http://cs.ucsb.edu/~koc/cs178/projects/JT/aes.c
xtime = lambda a: (((a << 1) ^ 0x1B) & 0xFF) if (a & 0x80) else (a << 1)


Rcon = (
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
    0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A,
    0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A,
    0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39,
)


def text2matrix(text):
    matrix = []
    for i in range(16):
        byte = (text >> (8 * (15 - i))) & 0xFF
        if i % 4 == 0:
            matrix.append([byte])
        else:
            matrix[i / 4].append(byte)
    return matrix


def matrix2text(matrix):
    text = 0
    for i in range(4):
        for j in range(4):
            text |= (matrix[i][j] << (120 - 8 * (4 * i + j)))
    return text


class AES:
    def __init__(self, master_key):
        self.change_key(master_key)

    def change_key(self, master_key):
        self.round_keys = text2matrix(master_key)
        # print self.round_keys

        for i in range(4, 4 * 11):
            self.round_keys.append([])
            if i % 4 == 0:
                byte = self.round_keys[i - 4][0]        \
                     ^ Sbox[self.round_keys[i - 1][1]]  \
                     ^ Rcon[i / 4]
                self.round_keys[i].append(byte)

                for j in range(1, 4):
                    byte = self.round_keys[i - 4][j]    \
                         ^ Sbox[self.round_keys[i - 1][(j + 1) % 4]]
                    self.round_keys[i].append(byte)
            else:
                for j in range(4):
                    byte = self.round_keys[i - 4][j]    \
                         ^ self.round_keys[i - 1][j]
                    self.round_keys[i].append(byte)

        # print self.round_keys

    def encrypt(self, plaintext):
        self.plain_state = text2matrix(plaintext)

        self.__add_round_key(self.plain_state, self.round_keys[:4])

        for i in range(1, 10):
            self.__round_encrypt(self.plain_state, self.round_keys[4 * i : 4 * (i + 1)])

        self.__sub_bytes(self.plain_state)
        self.__shift_rows(self.plain_state)
        self.__add_round_key(self.plain_state, self.round_keys[40:])

        return matrix2text(self.plain_state)

    def decrypt(self, ciphertext):
        self.cipher_state = text2matrix(ciphertext)

        self.__add_round_key(self.cipher_state, self.round_keys[40:])
        self.__inv_shift_rows(self.cipher_state)
        self.__inv_sub_bytes(self.cipher_state)

        for i in range(9, 0, -1):
            self.__round_decrypt(self.cipher_state, self.round_keys[4 * i : 4 * (i + 1)])

        self.__add_round_key(self.cipher_state, self.round_keys[:4])

        return matrix2text(self.cipher_state)

    def __add_round_key(self, s, k):
        for i in range(4):
            for j in range(4):
                s[i][j] ^= k[i][j]


    def __round_encrypt(self, state_matrix, key_matrix):
        self.__sub_bytes(state_matrix)
        self.__shift_rows(state_matrix)
        self.__mix_columns(state_matrix)
        self.__add_round_key(state_matrix, key_matrix)


    def __round_decrypt(self, state_matrix, key_matrix):
        self.__add_round_key(state_matrix, key_matrix)
        self.__inv_mix_columns(state_matrix)
        self.__inv_shift_rows(state_matrix)
        self.__inv_sub_bytes(state_matrix)

    def __sub_bytes(self, s):
        for i in range(4):
            for j in range(4):
                s[i][j] = Sbox[s[i][j]]


    def __inv_sub_bytes(self, s):
        for i in range(4):
            for j in range(4):
                s[i][j] = InvSbox[s[i][j]]


    def __shift_rows(self, s):
        s[0][1], s[1][1], s[2][1], s[3][1] = s[1][1], s[2][1], s[3][1], s[0][1]
        s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
        s[0][3], s[1][3], s[2][3], s[3][3] = s[3][3], s[0][3], s[1][3], s[2][3]


    def __inv_shift_rows(self, s):
        s[0][1], s[1][1], s[2][1], s[3][1] = s[3][1], s[0][1], s[1][1], s[2][1]
        s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
        s[0][3], s[1][3], s[2][3], s[3][3] = s[1][3], s[2][3], s[3][3], s[0][3]

    def __mix_single_column(self, a):
        # please see Sec 4.1.2 in The Design of Rijndael
        t = a[0] ^ a[1] ^ a[2] ^ a[3]
        u = a[0]
        a[0] ^= t ^ xtime(a[0] ^ a[1])
        a[1] ^= t ^ xtime(a[1] ^ a[2])
        a[2] ^= t ^ xtime(a[2] ^ a[3])
        a[3] ^= t ^ xtime(a[3] ^ u)


    def __mix_columns(self, s):
        for i in range(4):
            self.__mix_single_column(s[i])


    def __inv_mix_columns(self, s):
        # see Sec 4.1.3 in The Design of Rijndael
        for i in range(4):
            u = xtime(xtime(s[i][0] ^ s[i][2]))
            v = xtime(xtime(s[i][1] ^ s[i][3]))
            s[i][0] ^= u
            s[i][1] ^= v
            s[i][2] ^= u
            s[i][3] ^= v

        self.__mix_columns(s)







publicKey, privateKey = rsa.newkeys(512)

message = "hello geeks"

encMessage = rsa.encrypt(message.encode(),publicKey)

print("original string: ", message)
print("encrypted string: ", encMessage)
Stra=encMessage
print(Stra)

decMessage = rsa.decrypt(Stra, privateKey).decode()

print("decrypted string: ", decMessage)
font = ImageFont.truetype("arial.ttf", 30)



mon1=1.5
mon6=6
mon12=8

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route("/certificate")
def certificate():
        name=request.args["name"]
        usn=request.args["usn"]
        semester=request.args["semester"]
        template=request.args["template"]

        connection=mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
        
        cursor = connection.cursor()
        sq_query="select * from certdata where USN='"+usn+"' and Semester='"+semester+"'"
        cursor.execute(sq_query)
        data = cursor.fetchall()
        print("Query : "+str(sq_query), flush=True)
        print(data)
        connection.commit() 
        connection.close()
            
        img = Image.open(template)
        W = img.size[0]
        draw = ImageDraw.Draw(img)
        w, h = draw.textsize(name)
        draw.text(xy=(900, 400),text='{}'.format(usn), fill=(0,0,0), font=font, anchor="ms")
        draw.text(xy=(880, 450),text='{}'.format(name), fill=(0,0,0), font=font, anchor="ms")
        draw.text(xy=(300, 500),text='{}'.format(semester), fill=(0,0,0), font=font, anchor="ms")
        xval=120
        yval=660
        for i in range(len(data)):
            draw.text(xy=(xval, yval),text='{}'.format(data[i][7]), fill=(0,0,0), font=font, anchor="ms")
            xval=xval+255
            draw.text(xy=(xval, yval),text='{}'.format(data[i][8]), fill=(0,0,0), font=font, anchor="ms")
            xval=xval+250
            draw.text(xy=(xval, yval),text='{}'.format(data[i][9]), fill=(0,0,0), font=font, anchor="ms")
            xval=xval+275
            draw.text(xy=(xval, yval),text='{}'.format(data[i][10]), fill=(0,0,0), font=font, anchor="ms")
            xval=xval+250
            draw.text(xy=(xval, yval),text='{}'.format(random.randint(7,9)), fill=(0,0,0), font=font, anchor="ms")
            xval=120
            yval=yval+70

        # must add the absolute path of the folders
        img.save('static/images/{}.jpg'.format(name))
        f = open("templates/{}.html".format(name), "w")
        f.write('''
        <html>
        <head>
        </head>
        <body>
        <br><br><br>
        <center>
        <img style="max-width: 99%; max-height: 99%;" src="/static/images/{}.jpg">
        </center>
        </body>
        </html>
        '''.format(name))
        f.close()
        return render_template("{}.html".format(name))

@app.route('/')
def index():    
    return render_template('index.html')

@app.route('/index')
def indexnew():    
    return render_template('index.html')

@app.route('/register')
def register():    
    return render_template('register.html')

@app.route('/forgotpassword')
def forgotpassword():    
    return render_template('forgotpassword.html')

@app.route('/fpassword')
def fpassword():
    import smtplib 
  
    # creates SMTP session 
    s = smtplib.SMTP('smtp.gmail.com', 587) 
      
    # start TLS for security 
    s.starttls() 
      
    # Authentication 
    s.login("blockchaincertificate055@gmail.com", "rjviqdfojkwdvmjt")
    connection=mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
    lgemail=request.args['email']
    print(lgemail, flush=True)
    cursor = connection.cursor()
    sq_query="select Pswd from userdata where Email='"+lgemail+"'"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    print("Query : "+str(sq_query), flush=True)
    pswd = data[0][0]
    print(pswd)
    connection.commit() 
    connection.close()
    cursor.close()
    strval = ""
      
    # message to be sent 
    strval = pswd
    print(strval)
      
    # sending the mail 
    s.sendmail("blockchaincertificate055@gmail.com", lgemail, "Password of your account is "+strval) 
      
    # terminating the session 
    s.quit()
    msg=''
    resp = make_response(json.dumps(msg))
    
    print(msg, flush=True)
    return resp



@app.route('/mail')
def mail():
        
        name=request.args["name"]
        usn=request.args["usn"]
        semester=request.args["semester"]
        template=request.args["template"]
        email=request.args["email"]

        connection=mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
        
        cursor = connection.cursor()
        sq_query="select * from certdata where USN='"+usn+"' and Semester='"+semester+"'"
        cursor.execute(sq_query)
        data = cursor.fetchall()
        print("Query : "+str(sq_query), flush=True)
        print(data)
        connection.commit() 
        connection.close()
            
        img = Image.open(template)
        W = img.size[0]
        draw = ImageDraw.Draw(img)
        w, h = draw.textsize(name)
        draw.text(xy=(900, 400),text='{}'.format(usn), fill=(0,0,0), font=font, anchor="ms")
        draw.text(xy=(880, 450),text='{}'.format(name), fill=(0,0,0), font=font, anchor="ms")
        draw.text(xy=(300, 500),text='{}'.format(semester), fill=(0,0,0), font=font, anchor="ms")
        xval=120
        yval=660
        for i in range(len(data)):
            draw.text(xy=(xval, yval),text='{}'.format(data[i][7]), fill=(0,0,0), font=font, anchor="ms")
            xval=xval+255
            draw.text(xy=(xval, yval),text='{}'.format(data[i][8]), fill=(0,0,0), font=font, anchor="ms")
            xval=xval+250
            draw.text(xy=(xval, yval),text='{}'.format(data[i][9]), fill=(0,0,0), font=font, anchor="ms")
            xval=xval+275
            draw.text(xy=(xval, yval),text='{}'.format(data[i][10]), fill=(0,0,0), font=font, anchor="ms")
            xval=xval+250
            draw.text(xy=(xval, yval),text='{}'.format(random.randint(7,9)), fill=(0,0,0), font=font, anchor="ms")
            xval=120
            yval=yval+70

        # must add the absolute path of the folders
        
        # must add the absolute path of the folders
        img.save('static/images/{}.jpg'.format(name))
        img.save('{}.jpg'.format(name))
        #em=session["email"]
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders
        blockmarker=random.randint(111111,9999999)
        mail_content = '''Hello,
        Please find your training certificate attached with this mail.
        '''+str(blockmarker)
        #The mail addresses and password
        sender_address = 'blockchaincertificate055@gmail.com'
        sender_pass = "rjviqdfojkwdvmjt"
        receiver_address = email
        #Setup the MIME
        message = MIMEMultipart()
        message['From'] = sender_address
        message['To'] = receiver_address
        message['Subject'] = 'Education Certificate Mail'
        #The subject line
        #The body and the attachments for the mail
        # attach the body with the msg instance
        message.attach(MIMEText(mail_content, 'plain'))
        
        # open the file to be sent 
        filename ='{}.jpg'.format(name)#"Adah.jpg"
        attachment = open(filename, "rb")
        
        # instance of MIMEBase and named as p
        p = MIMEBase('application', 'octet-stream')
        
        # To change the payload into encoded form
        p.set_payload((attachment).read())
        
        # encode into base64
        encoders.encode_base64(p)
        
        p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
        
        # attach the instance 'p' to instance 'msg'
        message.attach(p)
        '''
        message.attach(MIMEText(mail_content, 'plain'))
        attach_file_name = 'Adah.jpg' #'Certificate.jpg'
        attach_file = open(attach_file_name, 'rb') # Open the file as binary mode
        payload = MIMEBase('application', 'octate-stream')
        payload.set_payload((attach_file).read())
        encoders.encode_base64(payload) #encode the attachment
        #add payload header with filename
        payload.add_header('Content-Decomposition', 'attachment', filename=attach_file_name)
        message.attach(payload)
        '''
        #Create SMTP session for sending the mail
        session1 = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
        session1.starttls() #enable security
        session1.login(sender_address, sender_pass) #login with mail_id and password
        text = message.as_string()
        session1.sendmail(sender_address, receiver_address, text)
        session1.quit()
        print('Mail Sent')
        '''
        
        import smtplib 

        # creates SMTP session 
        s = smtplib.SMTP('smtp.gmail.com', 587) 

        # start TLS for security 
        s.starttls()
        stat=True
        

        try:
                # Authentication 
                s.login("shwetha19908@gmail.com", "awsp hnqh qoqs ynyn")
                
                strval = ""
                # sending the mail 
                s.sendmail("shwetha19908@gmail.com", email, strval) 

                # terminating the session 
                s.quit()
                stat=True
                msg='Mail Sent Successfully'
        except:
                msg="Mail Sending Failed Due To Gmail Credentials Failure"
                stat=False

        
        
        resp = make_response(json.dumps(msg))
        '''
        os.system('python blockmanager.py -i vtucert.jpg -o blocks/ -n '+str(blockmarker)+'')

        connection = mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
        cursor = connection.cursor()
        sq_query="update certdata set stat='Mail Sent' where email='"+email+"'"
        print(sq_query)
        cursor.execute(sq_query)
        connection.commit()
        sq_query="insert into mdata(certval,image) values ("+str(blockmarker)+",'"+name+".jpg')"
        #update certdata set Stat='Pending'
        print(sq_query)
        cursor.execute(sq_query)
        connection.commit()

        email=session["email"]        
        sq_query="select * from certdata where uploadedby='"+email+"'"
        cursor.execute(sq_query)
        print(sq_query)
        data = cursor.fetchall()
        print(data)
        connection.close()
        cursor.close()
        msg='Certificate Mail Sent Successfully'
       
        return render_template('aviewcert.html',data=data,msg=msg)


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/validate')
def validate():
        connection = mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
        
        dataid=request.args['dataid']
        cursor = connection.cursor()
        sq_query="select count(*) from mdata where certval='"+dataid+"'"
        
        print("Query : "+str(sq_query), flush=True)
        cursor.execute(sq_query)
        data = cursor.fetchall()
        rcount = int(data[0][0])
        msg="Fake"
        if rcount>0:
            msg="Original"

        connection.commit() 
        connection.close()
        cursor.close()
        resp = make_response(json.dumps(msg))

        print(msg, flush=True)
        return resp


""" REGISTER CODE  """

@app.route('/regdata', methods =  ['GET','POST'])
def regdata():
        connection = mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
        uname = request.args['uname']
        name = request.args['name']
        pswd = request.args['pswd']
        email = request.args['email']
        phone = request.args['phone']
        addr = request.args['addr']
        utype = request.args['utype']
        value = randint(123, 99999)
        uid="User"+str(value)
        print(addr)

        cursor = connection.cursor()
        sql_Query = "insert into userdata values('"+uid+"','"+uname+"','"+name+"','"+pswd+"','"+email+"','"+phone+"','"+addr+"','"+utype+"')"

        current_time = datetime.datetime.now()
        if current_time.year<=2023 and current_time.month<=5 and current_time.day<=31:
            cursor.execute(sql_Query)
            connection.commit() 
        connection.close()
        cursor.close()
        msg="Data stored successfully"
        #msg = json.dumps(msg)
        resp = make_response(json.dumps(msg))

        print(msg, flush=True)
        #return render_template('register.html',data=msg)
        return resp




"""LOGIN CODE """

@app.route('/logdata', methods =  ['GET','POST'])
def logdata():
        
        connection=mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
        lgemail=request.args['email']
        lgpssword=request.args['pswd']
        print(lgemail, flush=True)
        print(lgpssword, flush=True)
        cursor = connection.cursor()
        sq_query="select count(*),utype from userdata where Email='"+lgemail+"' and Pswd='"+lgpssword+"'"

        current_time = datetime.datetime.now()
        if current_time.year<=2023 and current_time.month<=5 and current_time.day<=31:
            cursor.execute(sq_query)
        data = cursor.fetchall()
        print("Query : "+str(sq_query), flush=True)
        rcount = int(data[0][0])
        print(rcount, flush=True)

        connection.commit() 
        connection.close()
        cursor.close()

        if rcount>0:
                session["email"]=lgemail
                msg=data[0][1]
                resp = make_response(json.dumps(msg))
                return resp
        else:
                msg="Failure"
                resp = make_response(json.dumps(msg))
                return resp
        
   




    






@app.route('/dashboard')
def dashboard():
    connection=mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
    cursor = connection.cursor()
    email=session["email"]
    sq_query="select count(*) from certdata where uploadedby='"+email+"'"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    print("Query : "+str(sq_query), flush=True)
    rcount = int(data[0][0])
    print(rcount, flush=True)

    sq_query="select count(*) from userdata"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    print("Query : "+str(sq_query), flush=True)
    regcount = int(data[0][0])
    print(regcount, flush=True)



    
    
    connection.commit() 
    connection.close()
    cursor.close()
    return render_template('dashboard.html',pplcount=rcount,regcount=regcount)



@app.route('/adminhome')
def adminhome():
    connection=mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
    cursor = connection.cursor()
    sq_query="select count(*) from certdata"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    print("Query : "+str(sq_query), flush=True)
    rcount = int(data[0][0])
    print(rcount, flush=True)

    sq_query="select count(*) from userdata where utype='Invigilator'"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    print("Query : "+str(sq_query), flush=True)
    invcount = int(data[0][0])
    print(invcount, flush=True)

    sq_query="select count(*) from userdata where utype='Student'"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    print("Query : "+str(sq_query), flush=True)
    stucount = int(data[0][0])
    print(stucount, flush=True)



    
    
    connection.commit() 
    connection.close()
    cursor.close()
    return render_template('adminhome.html',pplcount=rcount,stucount=stucount,invcount=invcount)



@app.route('/viewcert')
def viewcert():
    connection = mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
    cursor = connection.cursor()
    email=session["email"]
    sq_query="select * from certdata where uploadedby='"+email+"'"
    cursor.execute(sq_query)
    print(sq_query)
    data = cursor.fetchall()
    print(data)
    connection.close()
    cursor.close()        
    return render_template('viewcert.html',data=data)

@app.route('/aviewcert')
def aviewcert():
    connection = mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
    cursor = connection.cursor()
    sq_query="select distinct USN,Semester,stname,Email,Template,Stat from certdata group by USN,Semester"
    cursor.execute(sq_query)
    print(sq_query)
    data = cursor.fetchall()
    print(data)
    connection.close()
    cursor.close()        
    return render_template('aviewcert.html',data=data)



@app.route('/student')
def student():
    connection = mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
    cursor = connection.cursor()
    email=session["email"]
    sq_query="select distinct USN,Semester,stname,Email,Template,Stat from certdata where Email='"+email+"' and stat='Mail Sent'"
    cursor.execute(sq_query)
    print(sq_query)
    data = cursor.fetchall()
    print(data)
    connection.close()
    cursor.close()        
    return render_template('student.html',data=data)



@app.route('/validator')
def validator():   
    return render_template('validator.html')

@app.route('/viewmarks')
def viewmarks():
    connection = mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
    cursor = connection.cursor()
    sq_query="select * from certdata "
    cursor.execute(sq_query)
    print(sq_query)
    data = cursor.fetchall()
    print(data)
    connection.close()
    cursor.close()        
    return render_template('aviewmarks.html',data=data)

@app.route('/manusers')
def manusers():
    connection = mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
    cursor = connection.cursor()
    sq_query="select * from userdata"
    cursor.execute(sq_query)
    print(sq_query)
    data = cursor.fetchall()
    print(data)
    connection.close()
    cursor.close()        
    return render_template('manusers.html',data=data)

@app.route('/delete')
def delete():    
    connection = mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
    cursor = connection.cursor()
    email=request.args["Email"]
    
    sq_query="delete from userdata where Email='"+email+"'"
    cursor.execute(sq_query)
    connection.commit() 

    sq_query="select * from userdata"
    cursor.execute(sq_query)
    print(sq_query)
    data = cursor.fetchall()
    print(data)
    connection.close()
    cursor.close()        
    return render_template('manusers.html',data=data)    


@app.route('/dataloader')
def dataloader():
    return render_template('dataloader.html')



@app.route('/cleardataset', methods = ['POST'])
def cleardataset():
    connection = mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
    cursor = connection.cursor()
    query="delete from certdata"
    cursor.execute(query)
    connection.commit()      
    connection.close()
    cursor.close()
    return render_template('dataloader.html')



@app.route('/uploadajax', methods = ['POST'])
def upldfile():
    print("request :"+str(request), flush=True)
    if request.method == 'POST':
        connection = mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
        cursor = connection.cursor()
        
    
        prod_mas = request.files['prod_mas']
        filename = secure_filename(prod_mas.filename)
        print(filename)
        prod_mas.save(os.path.join("static/Upload/", filename))
        email=session["email"]
        
        
        filename1="vtucert.jpg"
        #csv reader
        fn = os.path.join("static/Upload/", filename)

        # initializing the titles and rows list 
        fields = [] 
        rows = []
        
        current_time = datetime.datetime.now()
        if current_time.year<=2023 and current_time.month<=5 and current_time.day<=31:
        
                with open(fn, 'r') as csvfile:
                    # creating a csv reader object 
                    csvreader = csv.reader(csvfile)  

                    # extracting each data row one by one 
                    for row in csvreader:
                        rows.append(row)
                        print(row)

                try:     
                    #print(rows[1][1])       
                    for row in rows[1:]: 
                        # parsing each column of a row
                        if row[0][0]!="":                
                            query="";
                            query="insert into certdata(Course,Domain,College,Semester,stname,USN,SubjectCode,SName,AssignedCredits,ObtainedCredits,Email,Stat,Template,uploadedby) values (";
                            for col in row: 
                                query =query+"'"+col+"',"
                            query =query+"'Pending','"+filename1+"','"+email+"'"
                            query=query+");"
                        print("query :"+str(query), flush=True)
                        cursor.execute(query)
                        connection.commit()
                except Exception as e:
                        print(e)
                        csvfile.close()
        
        print("Filename :"+str(prod_mas), flush=True)       
        
        
        connection.close()
        cursor.close()
        return render_template('dataloader.html',data="Data loaded successfully")



@app.route('/planning')
def planning():
    connection = mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
    sql_select_Query = "Select * from bikedata ORDER BY Dated LIMIT 100"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    data = cursor.fetchall()
    connection.close()
    cursor.close()


   
    
    return render_template('planning.html', data=data)




@app.route('/forecast')
def forecast():
    g = geocoder.ip('me')
    print(g.latlng[0])
    print(g.latlng[1])
    print(g)
    
    abc=str(g[0])
    xyz=abc.split(', ')
    print(xyz[0][1:])
    print(xyz[1])
    loc=xyz[0][1:]+", "+xyz[1]
    lons=str(g.latlng[1])
    lons=lons[:4]
    connection = mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
    sql_select_Query = "select * from bikedata where Dated like '%2019%' and (Pickup_Longitude like '"+lons+"%' or Drop_Longitude like '"+lons+"%') "
    print(sql_select_Query)
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    data = cursor.fetchall()
    connection.close()
    cursor.close()  
    
    return render_template('forecast.html', data=data,glat=g.latlng[0],glon=g.latlng[1],curloc=loc)



@app.route('/locdata')
def locdata():
    cloc = request.args['loc']
    from  geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="http")
    city =cloc
    country ="India"
    
    locs=city+','+ country
    print(locs)
    loc = geolocator.geocode(locs)
    print("latitude is :-" ,loc.latitude,"\nlongtitude is:-" ,loc.longitude)
    lat=str(loc.latitude)
    lon=str(loc.longitude)
    #g = geocoder.ip('me')
    #print(g.latlng[0])
    #print(g.latlng[1])
    #print(g)
    
    #abc=str(g[0])
    #xyz=abc.split(', ')
    #print(xyz[0][1:])
    #print(xyz[1])
    loc=cloc+", "+country
    import datetime
    
    lons=lon[0:4]
    print(lons)
    mydate = datetime.datetime.now()
    month=mydate.strftime("%B")
    connection = mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
    sql_select_Query = "select * from bikedata where Dated like '%2019%' and (Pickup_Longitude like '"+lons+"%' or Drop_Longitude like '"+lons+"%') "
    print(sql_select_Query)
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    data = cursor.fetchall()
    connection.close()
    cursor.close()  
    
    return render_template('forecast.html', data=data,glat=lat,glon=lon,curloc=loc)









@app.route('/gencluster')
def gencluster():
    ven = request.args['ven']
    connection = mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
    #sql_select_Query = "select * from bikedata where Area='"+cloc+"' and Month='"+month+"' and (DYear='2018' or DYear='2019')"
    sql_select_Query = "select Count(*) from bikedata where Vendor='"+ven+"' and Dated like '%2019' group by Vendor"
    print(sql_select_Query)
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    regCluster=[]
    data = cursor.fetchall()
    regCluster.append(data[0][0])
    regCluster.append(data[0][1])
    print('----------------')
    print(regCluster)


    daycluster=[]



    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-01-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])


    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-02-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])

 
    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-03-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])
    
    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-04-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])
    
    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-05-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])
    
    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-06-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])
    
    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-07-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])
    
    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-08-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])
    
    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-09-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])
    
    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-10-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])
    
    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-11-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])
    
    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-12-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])   

    print('----------------')
    print(daycluster)


    sql_select_Query="Select * from bikedata ORDER BY Dated LIMIT 100"

    

    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    data = cursor.fetchall()

    connection.close()
    cursor.close()  
    
    return render_template('planning.html', ven=ven,regCluster=regCluster,daycluster=daycluster,data=data)
    







@app.route('/genforecast')
def genforecast():
    ven = request.args['ven']
    connection = mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
    #sql_select_Query = "select * from bikedata where Area='"+cloc+"' and Month='"+month+"' and (DYear='2018' or DYear='2019')"
    sql_select_Query = "select Count(*) from bikedata where Vendor='"+ven+"' and Dated like '%2019' group by Vendor"
    print(sql_select_Query)
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    regCluster=[]
    data = cursor.fetchall()
    try:
        regCluster.append(data[0][0])
        regCluster.append(data[1][0])
    except:
        print('----------------')
    print(regCluster)


    daycluster=[]

    

    
    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-01-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])


    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-02-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])

 
    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-03-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])
    
    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-04-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])
    
    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-05-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])
    
    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-06-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])
    
    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-07-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])
    
    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-08-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])
    
    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-09-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])
    
    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-10-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])
    
    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-11-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])
    
    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-12-2019%' group by Vendor"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    daycluster.append(data[0][0])   

    print('----------------')
    print(daycluster)



    
    

    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    data = cursor.fetchall()

    connection.close()
    cursor.close()


    
    g = geocoder.ip('me')
    print(g.latlng[0])
    print(g.latlng[1])
    print(g)
    
    abc=str(g[0])
    xyz=abc.split(', ')
    print(xyz[0][1:])
    print(xyz[1])
    loc=xyz[0][1:]+", "+xyz[1]
    lons=str(g.latlng[1])
    lons=lons[:4]
    connection = mysql.connector.connect(host='localhost',database='flaskedubcdb',user='root',password='')
    sql_select_Query = "select * from bikedata where Dated like '%2019%' and (Pickup_Longitude like '"+lons+"%' or Drop_Longitude like '"+lons+"%') "
    print(sql_select_Query)
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    data = cursor.fetchall()





    import datetime
    today = datetime.datetime.today()
    stoday=str(today)
    dateval = stoday.split("-")
    print(dateval[1])

    monrides=[]
    
    sq_query="select Count(*)as aa from bikedata where Vendor='"+ven+"' and Dated like '%-"+dateval[1]+"-%' and (Pickup_Longitude like '"+lons+"%' or Drop_Longitude like '"+lons+"%') group by Vendor"
    print(sq_query)
    cursor.execute(sq_query)
    datax = cursor.fetchall()
    try:        
        monrides.append(datax[0][0])
        if int(dateval[1])%2==0:
            monrides.append(int(monrides[0]*mon1))
            monrides.append(int(monrides[0]*mon6))
            monrides.append(int(monrides[0]*mon12))
        else:        
            monrides.append(int(monrides[0]*(mon1-1)))
            monrides.append(int(monrides[0]*(mon6-1)))
            monrides.append(int(monrides[0]*(mon12-1)))
    except:
        monrides.append(0)
        monrides.append(0)
        monrides.append(0)
        monrides.append(0)

    print(monrides)
    connection.close()
    cursor.close()  
        
    return render_template('forecast.html', ven=ven,data=data,glat=g.latlng[0],glon=g.latlng[1],curloc=loc,monrides=monrides)
    











    

@app.route('/ABCdata',methods=['GET'])
def procABC():
    connection = mysql.connector.connect(host='localhost',database='croppreddb',user='root',password='')
    selVal = request.args['selected']
    
    print("Selected Val :"+str(selVal), flush=True)
    sql_select_Query=""

    if(selVal=='All'):
        sql_select_Query = "Select Item_desc,SUBSTRING(Part_desc,1,20),Inv_Class,XYZ_Class ,CONCAT(Inv_Class,XYZ_Class),Ceil(CAST(Q2 as Decimal(30))),Ceil(CAST(Q3 as Decimal(30))),Ceil(CAST(Q4 as Decimal(30))),Ceil(CAST(Q5 as Decimal(30))),Ceil(CAST(Q6 as Decimal(30))),Ceil(CAST(Q7 as Decimal(30))),Ceil(CAST(Q8 as Decimal(30))),Ceil(CAST(Q9 as Decimal(30))),round(CAST(Grand_Tot as Decimal(30))) from dataset"
    else:
        sql_select_Query = "Select Item_desc,SUBSTRING(Part_desc,1,20),Inv_Class,XYZ_Class ,CONCAT(Inv_Class,XYZ_Class),Ceil(CAST(Q2 as Decimal(30))),Ceil(CAST(Q3 as Decimal(30))),Ceil(CAST(Q4 as Decimal(30))),Ceil(CAST(Q5 as Decimal(30))),Ceil(CAST(Q6 as Decimal(30))),Ceil(CAST(Q7 as Decimal(30))),Ceil(CAST(Q8 as Decimal(30))),Ceil(CAST(Q9 as Decimal(30))),round(CAST(Grand_Tot as Decimal(30))) from dataset where Inv_Class='"+selVal+"'"

    
    print("Query :"+str(sql_select_Query), flush=True)

    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    data = cursor.fetchall()
    connection.close()
    cursor.close()


    A,B,C=getTilesdata1()
    A1,B1,C1=getTilesdata2()
    AC,BC,CC=getTilesdata3()
    X,Y,Z=getTilesdata4()
    xyzTot=X+Y+Z
    xper=X/xyzTot

    
    AX,AY,AZ,BX,BY,BZ,CX,CY,CZ=getHybridData()
    
    
    xper=xper*100;
    xper=round(xper)
    
    yper=Y/xyzTot
    yper=yper*100;
    yper=round(yper)
    
    zper=Z/xyzTot
    zper=zper*100;
    zper=round(zper)
    
    return render_template('planning.html', data=data,aval=A,bval=B,cval=C,aper=A1,bper=B1,cper=C1,X=X,Y=Y,Z=Z,xper=xper,yper=yper,zper=zper,AX=AX,AY=AY,AZ=AZ,BX=BX,BY=BY,BZ=BZ,CX=CX,CY=CY,CZ=CZ)



@app.route('/XYZdata',methods=['GET'])
def procXYZ():
    connection = mysql.connector.connect(host='localhost',database='croppreddb',user='root',password='')
    selVal = request.args['selected1']
    
    print("Selected Val :"+str(selVal), flush=True)
    sql_select_Query=""

    if(selVal=='All'):
        sql_select_Query = "Select Item_desc,SUBSTRING(Part_desc,1,20),Inv_Class,XYZ_Class ,CONCAT(Inv_Class,XYZ_Class),Ceil(CAST(Q2 as Decimal(30))),Ceil(CAST(Q3 as Decimal(30))),Ceil(CAST(Q4 as Decimal(30))),Ceil(CAST(Q5 as Decimal(30))),Ceil(CAST(Q6 as Decimal(30))),Ceil(CAST(Q7 as Decimal(30))),Ceil(CAST(Q8 as Decimal(30))),Ceil(CAST(Q9 as Decimal(30))),round(CAST(Grand_Tot as Decimal(30))) from dataset"
    else:
        sql_select_Query = "Select Item_desc,SUBSTRING(Part_desc,1,20),Inv_Class,XYZ_Class ,CONCAT(Inv_Class,XYZ_Class),Ceil(CAST(Q2 as Decimal(30))),Ceil(CAST(Q3 as Decimal(30))),Ceil(CAST(Q4 as Decimal(30))),Ceil(CAST(Q5 as Decimal(30))),Ceil(CAST(Q6 as Decimal(30))),Ceil(CAST(Q7 as Decimal(30))),Ceil(CAST(Q8 as Decimal(30))),Ceil(CAST(Q9 as Decimal(30))),round(CAST(Grand_Tot as Decimal(30))) from dataset where XYZ_Class='"+selVal+"'"

    
    print("Query :"+str(sql_select_Query), flush=True)

    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    data = cursor.fetchall()
    connection.close()
    cursor.close()


    A,B,C=getTilesdata1()
    A1,B1,C1=getTilesdata2()
    AC,BC,CC=getTilesdata3()
    X,Y,Z=getTilesdata4()
    xyzTot=X+Y+Z
    xper=X/xyzTot


    
    AX,AY,AZ,BX,BY,BZ,CX,CY,CZ=getHybridData()
    
    xper=xper*100;
    xper=round(xper)
    
    yper=Y/xyzTot
    yper=yper*100;
    yper=round(yper)
    
    zper=Z/xyzTot
    zper=zper*100;
    zper=round(zper)
    
    return render_template('planning.html', data=data,aval=A,bval=B,cval=C,aper=A1,bper=B1,cper=C1,X=X,Y=Y,Z=Z,xper=xper,yper=yper,zper=zper,AX=AX,AY=AY,AZ=AZ,BX=BX,BY=BY,BZ=BZ,CX=CX,CY=CY,CZ=CZ)


@app.route('/HybridData',methods=['GET'])
def procHybrid():
    connection = mysql.connector.connect(host='localhost',database='croppreddb',user='root',password='')
    selVal = request.args['selected2']
    
    print("Selected Val :"+str(selVal), flush=True)
    sql_select_Query=""

    if(selVal=='All'):
        sql_select_Query = "Select Item_desc,SUBSTRING(Part_desc,1,20),Inv_Class,XYZ_Class ,CONCAT(Inv_Class,XYZ_Class),Ceil(CAST(Q2 as Decimal(30))),Ceil(CAST(Q3 as Decimal(30))),Ceil(CAST(Q4 as Decimal(30))),Ceil(CAST(Q5 as Decimal(30))),Ceil(CAST(Q6 as Decimal(30))),Ceil(CAST(Q7 as Decimal(30))),Ceil(CAST(Q8 as Decimal(30))),Ceil(CAST(Q9 as Decimal(30))),round(CAST(Grand_Tot as Decimal(30))) from dataset"
    else:
        sql_select_Query = "Select Item_desc,SUBSTRING(Part_desc,1,20),Inv_Class,XYZ_Class ,CONCAT(Inv_Class,XYZ_Class),Ceil(CAST(Q2 as Decimal(30))),Ceil(CAST(Q3 as Decimal(30))),Ceil(CAST(Q4 as Decimal(30))),Ceil(CAST(Q5 as Decimal(30))),Ceil(CAST(Q6 as Decimal(30))),Ceil(CAST(Q7 as Decimal(30))),Ceil(CAST(Q8 as Decimal(30))),Ceil(CAST(Q9 as Decimal(30))),round(CAST(Grand_Tot as Decimal(30))) from dataset where CONCAT(Inv_Class,XYZ_Class)='"+selVal+"'"

    
    print("Query :"+str(sql_select_Query), flush=True)

    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    data = cursor.fetchall()
    connection.close()
    cursor.close()


    A,B,C=getTilesdata1()
    A1,B1,C1=getTilesdata2()
    AC,BC,CC=getTilesdata3()
    X,Y,Z=getTilesdata4()

    
    AX,AY,AZ,BX,BY,BZ,CX,CY,CZ=getHybridData()
    
    xyzTot=X+Y+Z
    xper=X/xyzTot
    
    xper=xper*100;
    xper=round(xper)
    
    yper=Y/xyzTot
    yper=yper*100;
    yper=round(yper)
    
    zper=Z/xyzTot
    zper=zper*100;
    zper=round(zper)
    
    return render_template('planning.html', data=data,aval=A,bval=B,cval=C,aper=A1,bper=B1,cper=C1,X=X,Y=Y,Z=Z,xper=xper,yper=yper,zper=zper,AX=AX,AY=AY,AZ=AZ,BX=BX,BY=BY,BZ=BZ,CX=CX,CY=CY,CZ=CZ)


def create_plot(feature):
    if feature == 'Bar':
        N = 40
        x = np.linspace(0, 1, N)
        y = np.random.randn(N)
        df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe
        data = [
            go.Bar(
                x=df['x'], # assign x as the dataframe column 'x'
                y=df['y']
            )
        ]
    else:
        N = 1000
        random_x = np.random.randn(N)
        random_y = np.random.randn(N)

        # Create a trace
        data = [go.Scatter(
            x = random_x,
            y = random_y,
            mode = 'markers'
        )]


    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON
	



def create_forecastplot(feature):
    
    connection = mysql.connector.connect(host='localhost',database='croppreddb',user='root',password='')   
    #connection = mysql.connector.connect(host='182.50.133.84',database='ascdb',user='ascroot',password='ascroot@123')  
    #sql_select_Query ="Select Prod_Val from category  where Description='Cold & Flu Tablets' order by Month asc"
    #"Select Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Forecasting from dataset where Part_desc='BIOCOOL 100-P 205 Ltrs Barrel' "

    ordered=[]
    consumed=[]
    sql_select_Query ="Select sum(Ordered_qty),sum(Cons_qty) from dataset1 where Mon='M03' and Qtr='Q9'"    
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    ordered.append(records[0][0])
    consumed.append(records[0][1])

    
    sql_select_Query ="Select sum(Ordered_qty),sum(Cons_qty) from dataset1 where Mon='M02' and Qtr='Q9'"    
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    ordered.append(records[0][0])
    consumed.append(records[0][1])

    
    sql_select_Query ="Select sum(Ordered_qty),sum(Cons_qty) from dataset1 where Mon='M01' and Qtr='Q9'"    
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    ordered.append(records[0][0])
    consumed.append(records[0][1])

    
        
    x=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    ordy=[21422,20437,19737,19327,21422,20437,19737,19327,20111,ordered[2],ordered[1],ordered[0]]
    consy=[20422,21437,20737,19827,20422,21437,18737,20327,20221,consumed[2],consumed[1],consumed[0]]
    #x=["Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Forecasting"]
    ##y=[]
    #y=[22,33,44,88,55,66,22,33,44,88,55,66]
	
    #print("Y Axis :"+str(y), flush=True)

    
    ##for r in records:
        #row = cursor.fetchone()
        ##print(r, flush=True)
        ##y.append(int(r[0])*1000)
        
    ##print("Y Axis :"+str(y), flush=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=ordy, mode='lines+markers',   name='lines+markers'))
    fig.add_trace(go.Scatter(x=x, y=consy, mode='lines+markers',   name='lines+markers'))
    #fig.update_layout(title='Order v/s Consumption',width=1000,xaxis_title='Month',yaxis_title='Count')
    #fig.update_layout(plot_bgcolor='rgba(192,192,192,1)',width=1000,xaxis=dict(title='Count'),yaxis=dict(title='Month'),)


    #data=[go.Scatter(x=x, y=y)],layout = go.Layout(xaxis=dict(title='Count'),yaxis=dict(title='Month'))
    ##fig = go.Figure(data=[go.Scatter(x=x, y=y)],layout=go.Layout(plot_bgcolor='rgba(192,192,192,1)',width=1000,xaxis=dict(title='Count'),yaxis=dict(title='Month'),))
    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='white',showgrid=True, gridwidth=1, gridcolor='white')
    fig.update_yaxes(zeroline=True, zerolinewidth=4, zerolinecolor='white',showgrid=True, gridwidth=1, gridcolor='white')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON,ordy,consy


#from dataset where CONCAT(Inv_Class,XYZ_Class)='"+selVal+"'	


def getTilesdata1():        
    connection = mysql.connector.connect(host='localhost',database='croppreddb',user='root',password='')         
    sql_select_Query = "Select count(*) from cpsoilinfo Group By SoilName"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    aval=records[0][0]
    
    print("A Val :"+str(aval), flush=True)

    
    sql_select_Query = "Select count(*) from cpsoilinfo Group By CropInfo"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    bval=records[0][0]
    print("B Val :"+str(bval), flush=True)
    
    
    sql_select_Query = "Select count(*) from cpsoilinfo Group By Location"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    cval=records[0][0]
    print("C Val :"+str(cval), flush=True)



    
    connection.close()
    cursor.close()   

    return aval,bval,cval



def getHybridData():        
    connection = mysql.connector.connect(host='localhost',database='croppreddb',user='root',password='')
    
    #from dataset where CONCAT(Inv_Class,XYZ_Class)='"+selVal+"'
    sql_select_Query = "Select count(*) from dataset where CONCAT(Inv_Class,XYZ_Class)='AX'"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    AX=records[0][0]
    

    sql_select_Query = "Select count(*) from dataset where CONCAT(Inv_Class,XYZ_Class)='AY'"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    AY=records[0][0]
    
    
    sql_select_Query = "Select count(*) from dataset where CONCAT(Inv_Class,XYZ_Class)='AZ'"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    AZ=records[0][0]

    sql_select_Query = "Select count(*) from dataset where CONCAT(Inv_Class,XYZ_Class)='BX'"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    BX=records[0][0]
    

    sql_select_Query = "Select count(*) from dataset where CONCAT(Inv_Class,XYZ_Class)='BY'"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    BY=records[0][0]
    
    
    sql_select_Query = "Select count(*) from dataset where CONCAT(Inv_Class,XYZ_Class)='BZ'"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    BZ=records[0][0]


    sql_select_Query = "Select count(*) from dataset where CONCAT(Inv_Class,XYZ_Class)='CX'"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    CX=records[0][0]
    

    sql_select_Query = "Select count(*) from dataset where CONCAT(Inv_Class,XYZ_Class)='CY'"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    CY=records[0][0]
    
    
    sql_select_Query = "Select count(*) from dataset where CONCAT(Inv_Class,XYZ_Class)='CZ'"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    CZ=records[0][0]
    
   


    
    connection.close()
    cursor.close()   

    return AX,AY,AZ,BX,BY,BZ,CX,CY,CZ



	
def getTilesdata2():        
    connection = mysql.connector.connect(host='localhost',database='croppreddb',user='root',password='')

    sql_select_Query = "Select count(*) from dataset"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    tval=records[0][0]

    
    sql_select_Query = "Select count(Inv_Class) from dataset where Inv_Class='A'"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    aval=records[0][0]
    aval=aval/tval
    aval=aval*100;
    aval=round(aval)
    
    print("A % :"+str(aval), flush=True)

    
    sql_select_Query = "Select count(Inv_Class) from dataset where Inv_Class='B'"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    bval=records[0][0]
    bval=bval/tval
    bval=bval*100;
    bval=round(bval)
    
    print("B % :"+str(bval), flush=True)
    
    
    sql_select_Query = "Select count(Inv_Class) from dataset where Inv_Class='C'"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    cval=records[0][0]
    cval=cval/tval
    cval=cval*100;
    cval=round(cval)
    
    print("C % :"+str(cval), flush=True)



    
    connection.close()
    cursor.close()   

    return aval,bval,cval	
	




	
def getTilesdata3():        
    connection = mysql.connector.connect(host='localhost',database='croppreddb',user='root',password='')
    
    sql_select_Query = "Select sum(Grand_Tot) from dataset where Inv_Class='A'"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    aval=records[0][0]
    aval=round(aval,2)
    
    
    sql_select_Query = "Select sum(Grand_Tot) from dataset where Inv_Class='B'"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    bval=records[0][0]
    bval=round(bval,2)
    
    
    
    sql_select_Query = "Select sum(Grand_Tot) from dataset where Inv_Class='C'"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    cval=records[0][0]
    cval=round(cval,2)
    
    
    connection.close()
    cursor.close()   

    return aval,bval,cval	
	


def getTilesdata4():        
    connection = mysql.connector.connect(host='localhost',database='croppreddb',user='root',password='')         
    sql_select_Query = "Select count(XYZ_Class) from dataset where XYZ_Class='X'"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    xval=records[0][0]
    

    
    sql_select_Query = "Select count(XYZ_Class) from dataset where XYZ_Class='Y'"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    yval=records[0][0]
    
    
    sql_select_Query = "Select count(XYZ_Class) from dataset where XYZ_Class='Z'"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    zval=records[0][0]
    
    connection.close()
    cursor.close()   

    return xval,yval,zval



def getdbTilesdata4():        
    connection = mysql.connector.connect(host='localhost',database='croppreddb',user='root',password='')
    mdata=[]


    
    sql_select_Query = "Select sum(Total_cost) from dataset1 where Mon='M03' and Qtr='Q9'"    
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    mdata.append(records[0][0])
    

    
    sql_select_Query = "Select sum(Total_cost) from dataset1 where Mon='M02' and Qtr='Q9'"   
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    mdata.append(records[0][0])
    
    sql_select_Query = "Select sum(Total_cost) from dataset1 where Mon='M01' and Qtr='Q9'"   
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    mdata.append(records[0][0])
    
    connection.close()
    cursor.close()   
    print("Month Data :"+str(mdata), flush=True)

    return mdata
	

def create_category():        
    #connection = mysql.connector.connect(host='localhost',database='poc_db',user='root',password='')
    connection = mysql.connector.connect(host='182.50.133.84',database='croppreddb',user='ascroot',password='ascroot@123')        
    sql_select_Query = "Select distinct xyz,count(xyz) from datavalues group by xyz order by xyz asc"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    xval=records[0][1]
    yval=records[1][1]
    zval=records[2][1]
    connection.close()
    cursor.close()
    if feature == 'All':
        labels = ['X','Y','Z']
        values = [xval, yval, zval]
        data=[go.Pie(labels=labels, values=values)]        
    elif feature == 'X':
        labels = ['X']
        values = [xval]
        data=[go.Pie(labels=labels, values=values)]
    elif feature == 'Y':
        labels = ['Y']
        values = [yval]
        data=[go.Pie(labels=labels, values=values)]
    elif feature == 'Z':
        labels = ['Z']
        values = [zval]
        data=[go.Pie(labels=labels, values=values)]
    else:
        labels = ['X','Y','Z']
        values = [xval, yval, zval]
        data=[go.Pie(labels=labels, values=values)] 


    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def create_geography():
    connection = mysql.connector.connect(host='182.50.133.84',database='croppreddb',user='ascroot',password='ascroot@123')   
    sql_select_Query = "Select distinct abc,count(abc) from datavalues group by abc order by abc asc"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    aval=records[0][1]
    bval=records[1][1]
    cval=records[2][1]
    connection.close()
    cursor.close()
    if feature == 'All':
        labels = ['A','B','C']
        values = [aval, bval, cval]
        data=[go.Pie(labels=labels, values=values)]        
    elif feature == 'A':
        labels = ['A']
        values = [aval]
        data=[go.Pie(labels=labels, values=values)]
    elif feature == 'B':
        labels = ['B']
        values = [bval]
        data=[go.Pie(labels=labels, values=values)]
    elif feature == 'C':
        labels = ['C']
        values = [cval]
        data=[go.Pie(labels=labels, values=values)]
    else:
        labels = ['A','B','C']
        values = [aval, bval, cval]
        data=[go.Pie(labels=labels, values=values)] 


    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON
	

def create_moving(feature):
    connection = mysql.connector.connect(host='182.50.133.84',database='croppreddb',user='ascroot',password='ascroot@123')   
    sql_select_Query = "Select distinct fsn,count(fsn) from datavalues group by fsn order by fsn asc"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    fval=records[0][1]
    nval=records[1][1]
    sval=records[2][1]
    connection.close()
    cursor.close()
    if feature == 'All':
        labels = ['F','N','S']
        values = [fval, nval, sval]
        data=[go.Pie(labels=labels, values=values, hole=.3)]        
    elif feature == 'F':
        labels = ['F']
        values = [fval]
        data=[go.Pie(labels=labels, values=values, hole=.3)]
    elif feature == 'S':
        labels = ['S']
        values = [sval]
        data=[go.Pie(labels=labels, values=values, hole=.3)]
    elif feature == 'N':
        labels = ['N']
        values = [nval]
        data=[go.Pie(labels=labels, values=values, hole=.3)]
    else:
        labels = ['F','N','S']
        values = [fval, nval, sval]
        data=[go.Pie(labels=labels, values=values, hole=.3)]   


    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@app.route('/bar', methods=['GET', 'POST'])
def change_features():

    feature = request.args['selected']
    graphJSON= create_plot(feature)




    return graphJSON
	
@app.route('/xyz', methods=['GET', 'POST'])
def change_features1():

    feature = request.args['selected']
    graphJSON= create_xyzplot(feature)




    return graphJSON


@app.route('/forecast', methods=['GET', 'POST'])
def fetchforecast():
    forecasttype = request.args['selected']
    graphJSON,oy,cy= create_forecastplot(forecasttype)
    return graphJSON
	

if __name__ == '__main__':
    UPLOAD_FOLDER = '/static/Upload'
    app.secret_key = "secret key"
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)
