

#================================================================
#   Copyright (C) 2024 Fisher. All rights reserved.
#   
#   文件名称：makefile
#   创 建 者：YuLianghua
#   创建日期：2024年01月17日
#   描    述：
#
#================================================================

TARGET := gemm.cubin

all: $(TARGET)

$(TARGET): gemm.cu
	nvcc -arch sm_75 -maxrregcount 128 --ptxas-options=-v -cubin gemm.cu

clean:
	rm -f $(TARGET)

.PHONY: all clean 
