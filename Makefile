#GCC_FLAGS  =-std=gnu99 -Wall -O3
#NVCC_FLAGS =-O3 -arch=sm_35 -maxrregcount=144 --restrict
NVCC_FLAGS =-O3 -g -G --gpu-architecture=compute_70 --gpu-code=compute_70 -lcublas -lcusparse -lcurand
INCLUDE = -I/usr/local/cuda/include
CFLAGS =-std=c99 -O3 -Wall
CC = gcc

all: main

main: main.o go.o go_solve.o go_ion.o gr.o gr_solve.o gr_ion.o pkj.o pkj_solve.o pkj_ion.o io.o io_solve.o io_ion.o  solve_bem.o solve_cnm.o solve_rkc.o syn.o gap.o output.o reduction.o
	nvcc $(INCLUDE) ${NVCC_FLAGS} -o $@ $^

main.o: main.cu param.h go.cuh go_solve.cuh go_ion.cuh gr.cuh gr_solve.cuh gr_ion.cuh pkj.cuh pkj_solve.cuh pkj_ion.cuh solve_bem.cuh solve_cnm.cuh solve_rkc.cuh syn.cuh gap.cuh output.cuh 
	nvcc $(INCLUDE) ${NVCC_FLAGS} -c $<

go.o: go.cu go.cuh param.h go_ion.cuh
	nvcc $(INCLUDE) ${NVCC_FLAGS} -c $<

go_solve.o: go_solve.cu go_solve.cuh param.h go.cuh go_ion.cuh solve_bem.cuh solve_cnm.cuh solve_rkc.cuh syn.cuh
	nvcc $(INCLUDE) ${NVCC_FLAGS} -c $<

go_ion.o: go_ion.cu go_ion.cuh param.h go.cuh go_solve.cuh
	nvcc $(INCLUDE) ${NVCC_FLAGS} -c $<

gr.o: gr.cu gr.cuh param.h gr_ion.cuh
	nvcc $(INCLUDE) ${NVCC_FLAGS} -c $<

gr_solve.o: gr_solve.cu gr_solve.cuh param.h gr.cuh gr_ion.cuh solve_bem.cuh solve_cnm.cuh solve_rkc.cuh syn.cuh
	nvcc $(INCLUDE) ${NVCC_FLAGS} -c $<

gr_ion.o: gr_ion.cu gr_ion.cuh param.h gr.cuh gr_solve.cuh
	nvcc $(INCLUDE) ${NVCC_FLAGS} -c $<

pkj.o: pkj.cu pkj.cuh param.h pkj_ion.cuh
	nvcc $(INCLUDE) ${NVCC_FLAGS} -c $<

pkj_solve.o: pkj_solve.cu pkj_solve.cuh param.h pkj.cuh pkj_ion.cuh solve_bem.cuh syn.cuh
	nvcc $(INCLUDE) ${NVCC_FLAGS} -c $<

pkj_ion.o: pkj_ion.cu pkj_ion.cuh param.h pkj.cuh pkj_solve.cuh
	nvcc $(INCLUDE) ${NVCC_FLAGS} -c $<

io.o: io.cu io.cuh param.h io_ion.cuh
	nvcc $(INCLUDE) ${NVCC_FLAGS} -c $<

io_solve.o: io_solve.cu io_solve.cuh param.h io.cuh io_ion.cuh solve_bem.cuh solve_cnm.cuh solve_rkc.cuh syn.cuh
	nvcc $(INCLUDE) ${NVCC_FLAGS} -c $<

io_ion.o: io_ion.cu io_ion.cuh param.h io.cuh io_solve.cuh
	nvcc $(INCLUDE) ${NVCC_FLAGS} -c $<

solve_bem.o: solve_bem.cu solve_bem.cuh param.h go.cuh go_solve.cuh go_ion.cuh gr.cuh gr_solve.cuh gr_ion.cuh pkj.cuh pkj_solve.cuh pkj_ion.cuh io.cuh io_solve.cuh io_ion.cuh syn.cuh gap.cuh
	nvcc $(INCLUDE) ${NVCC_FLAGS} -c $<

solve_cnm.o: solve_cnm.cu solve_cnm.cuh param.h go.cuh go_solve.cuh go_ion.cuh gr.cuh gr_solve.cuh gr_ion.cuh pkj.cuh pkj_solve.cuh pkj_ion.cuh io.cuh io_solve.cuh io_ion.cuh syn.cuh gap.cuh
	nvcc $(INCLUDE) ${NVCC_FLAGS} -c $<

solve_rkc.o: solve_rkc.cu solve_rkc.cuh param.h go.cuh go_solve.cuh go_ion.cuh gr.cuh gr_solve.cuh gr_ion.cuh pkj.cuh pkj_solve.cuh pkj_ion.cuh io.cuh io_solve.cuh io_ion.cuh syn.cuh gap.cuh
	nvcc $(INCLUDE) ${NVCC_FLAGS} -c $<

reduction.o: reduction.cu reduction.h
	nvcc $(INCLUDE) ${NVCC_FLAGS} -c $<

output.o: output.cu output.cuh
	nvcc $(INCLUDE) ${NVCC_FLAGS} -c $<

syn.o: syn.cu syn.cuh gr.cuh go.cuh pkj.cuh param.h
	nvcc $(INCLUDE) ${NVCC_FLAGS} -c $<

gap.o: gap.cu gap.cuh io.cuh param.h
	nvcc $(INCLUDE) ${NVCC_FLAGS} -c $<

clean:
	rm -f *.o
