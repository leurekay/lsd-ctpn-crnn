#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 11:08:48 2018

@author: dirac
"""

class Solution:
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        
        row,col=len(matrix),len(matrix[0])
        table=[[0]*col for _ in range(row)]
        def select(pre,cur):
            for i in range(4):
                vector=[pre[0]-cur[0],pre[1]-cur[1]]
                nex=[cur[0]+vector[1],cur[1]-vector[0]]
                
                if nex[0]<row and nex[0]>=0 and nex[1]<col and nex[1]>=0:
                    if table[nex[0]][nex[1]]==0:
                        return nex
                pre=nex
            
            if i==3:
                return [-1,-1]
        
        pre=[-1,0]
        cur=[0,0]
        ret=[]
        while cur[0]>=0 and cur[1]>=0 :
            ret.append(matrix[cur[0]][cur[1]])
            table[cur[0]][cur[1]]=1
            nex=select(pre,cur)
            
            pre=cur
            cur=nex
        return ret
            
            
a=[[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]]

s=Solution()
d=s.spiralOrder(a)