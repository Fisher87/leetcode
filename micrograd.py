#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2024 Fisher. All rights reserved.
#   
#   文件名称：micrograd.py
#   创 建 者：YuLianghua
#   创建日期：2024年03月08日
#   描    述：
#
#================================================================

class Node:
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)

    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data+other.data, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data*other.data, (self, other))
        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data ** other.data, (self, other))
        def _backward():
            self.grad += (other * self.data * (other-1)) * out.grad
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()
