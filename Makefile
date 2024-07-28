CUDA_CXXFLAGS := -I${CUDA_HOME}/include -Icub -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 --expt-extended-lambda --expt-relaxed-constexpr

CUDA_LIBS := -L${CUDA_HOME}/lib64 -lcuda -lcudart -lcurand -lnvToolsExt

CUDF_CXXFLAGS := -I/path/to/cudf/include -I/path/to/rmm/include
CUDF_LIBS := -L/path/to/cudf/lib -L/path/to/rmm/lib -lcudf -lrmm

CXXFLAGS := -std=c++14 -O3 -Xcompiler -fopenmp ${CUDA_CXXFLAGS} ${CUDF_CXXFLAGS}
LDFLAGS := ${CUDA_LIBS} ${CUDF_LIBS}

# use long long (8B) for key_type instead of int (4B)
# CXXFLAGS += -DUSE_8B_KEYS

all: scripts/partitioned_join

partitioned_join: main.cu
	nvcc $(CXXFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -rf partitioned_join