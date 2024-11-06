CPPSRC = main.cpp kernel.cu kernel.cpp ../m5iface/m5iface.c
EXECUTABLE = bs.gem5
# BC has a compilation error in host code with ROCm 4.0, so O2 and O3 do not
# pass even on real GPUs
OPTS = -O1

HIP_PATH ?= /opt/rocm/hip
HIPCC = $(HIP_PATH)/bin/hipcc

# these are needed for m5ops
GEM5_PATH ?= $(shell pwd)/../.gem5
CFLAGS += -I$(GEM5_PATH)/include -I../m5iface -O1
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -L$(GEM5_PATH)/m5iface/build/x86_64 -l:libm5.a -l:libm5iface.a -lz -lm -lpthread
TARGET =gfx801,gfx803,gfx902
BIN_DIR ?= ./bin

all: $(BIN_DIR)/$(EXECUTABLE)

$(BIN_DIR)/$(EXECUTABLE): $(CPPSRC) $(BIN_DIR)
	$(HIPCC) $(OPTS) --amdgpu-target=$(TARGET) $(CXXFLAGS) $(CPPSRC) -o $(BIN_DIR)/$(EXECUTABLE) $(CFLAGS) $(LDFLAGS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)

.PHONY: clean
