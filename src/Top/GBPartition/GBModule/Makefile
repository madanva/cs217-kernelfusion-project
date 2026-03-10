# Copyright (c) 2016-2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

LOGFILE = build.log
CFLAGS = -DHLS_ALGORITHMICC
DEBUG_FLAG = -DDEBUG_LEVEL=5
HLS_SCRIPTS ?= $(REPO_TOP)/scripts/hls/

include $(HLS_SCRIPTS)/Makefile_src

.PHONY: all run clean_attn

all: clean sim_test run

run:
	./sim_test

# Original GBModule testbench (NMP/GBControl tests)
sim_test: testbench.cpp $(wildcard *.h)
	$(CC) -o sim_test $(CFLAGS) $(USER_FLAGS) testbench.cpp $(BOOSTLIBS) $(LIBS) > $(LOGFILE) 2>&1

sim_test_debug: testbench.cpp $(wildcard *.h)
	$(CC) -o sim_test $(CFLAGS) $(DEBUG_FLAG) $(USER_FLAGS) testbench.cpp $(BOOSTLIBS) $(LIBS) > $(LOGFILE) 2>&1

# Attention integration testbench
# Usage:
#   make sim_test_attn                              # default: fully fused
#   make sim_test_attn USER_FLAGS="-DATTN_UNFUSED"  # unfused variant
#   make sim_test_attn USER_FLAGS="-DATTN_PARTIAL_FUSED"  # partial fused
sim_test_attn: testbench_attn.cpp $(wildcard *.h)
	$(CC) -o sim_test_attn $(CFLAGS) $(USER_FLAGS) testbench_attn.cpp $(BOOSTLIBS) $(LIBS) > $(LOGFILE) 2>&1

clean_attn:
	rm -f sim_test_attn
