#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2023 Fisher. All rights reserved.
#   
#   文件名称：kmp.py
#   创 建 者：YuLianghua
#   创建日期：2023年06月15日
#   描    述：
#
#================================================================

def build_next(patternString):
    '''生成next数组'''
    next_list = [0]
    prefix_len = 0
    i = 1
    while i<len(patternString):
        if patternString[i]==patternString[prefix_len]:
            prefix_len += 1
            next_list.append(prefix_len)
            i+=1
        else:
            if prefix_len == 0:
                next_list.append(0)
                i += 1
            else:
                # 这里不更改i值
                prefix_len = next_list[prefix_len-1]

    return next_list

def match(source, patternString):
    '''子串匹配'''
    next_list = build_next(patternString)
    print(next_list)
    if not patternString:
        return ''
    m = len(source)
    n = len(patternString)

    i,j = 0, 0
    while i<m:
        if source[i]==patternString[j]:
            i+=1
            j+=1
        elif j>0:
            j = next_list[j-1]
        else:
            i += 1

        if j==n:
            return i-j

    return -1

if __name__ == "__main__":
    source = "aabcadfgedggsadfg"
    patternString = 'cadca'
    print(match(source, patternString))
