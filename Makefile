DESIGN_NAME ?= GBModule
AWS_DESIGN_NAME ?= design_top
SUBMISSION_NAME ?= lab3-submission

REPO_TOP := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SRC_HOME := $(REPO_TOP)/src
HLS_HOME := $(REPO_TOP)/hls
REPORT_HOME := $(REPO_TOP)/reports
AWS_HOME := $(REPO_TOP)/$(AWS_DESIGN_NAME)
AWS_LOGS := $(AWS_HOME)/logs

.PHONY: systemc_sim hls_sim hls_sim_debug clean submission copy_rtl hls

# Copy the generated RTL to AWS design folder
copy_rtl:
	mkdir -p $(AWS_HOME)/design
	mkdir -p $(REPORT_HOME)
	mkdir -p $(REPORT_HOME)/hls
	cp $(HLS_HOME)/$(DESIGN_NAME)/Catapult/$(DESIGN_NAME).v1/concat_$(DESIGN_NAME).v $(AWS_HOME)/design
	cp $(HLS_HOME)/$(DESIGN_NAME)/Catapult/$(DESIGN_NAME).v1/$(DESIGN_NAME).rpt $(REPORT_HOME)/hls/
	cp $(HLS_HOME)/$(DESIGN_NAME)/Catapult/$(DESIGN_NAME).v1/scverify/concat_sim_$(DESIGN_NAME)_v_vcs/sim.log $(REPORT_HOME)/hls/$(DESIGN_NAME)_hls_sim.log

# Run SystemC simulation
systemc_sim:
	cd $(SRC_HOME)/$(DESIGN_NAME) && make sim_test && make run

# Run HLS and copy RTL
hls: hls_sim copy_rtl

# Run HLS simulation
hls_sim:
	cd $(HLS_HOME)/$(DESIGN_NAME) && make

# Run HLS simulation with debug
hls_sim_debug:
	cd $(HLS_HOME)/$(DESIGN_NAME) && make vcs_debug
	verdi -ssf $(HLS_HOME)/$(DESIGN_NAME)/default.fsdb \
		-dbdir $(HLS_HOME)/$(DESIGN_NAME)/Catapult/$(DESIGN_NAME).v1/scverify/concat_sim_$(DESIGN_NAME)_v_vcs/sc_main.daidir/

# Clean all generated files
clean:
	cd $(SRC_HOME)/$(DESIGN_NAME) && make clean
	cd $(HLS_HOME)/$(DESIGN_NAME) && make clean
	rm -rf design_top/build/checkpoints/
	rm -rf design_top/build/constraints/generated_cl_clocks_aws.xdc
	rm -rf design_top/build/reports/
	rm -rf design_top/build/src_post_encryption
	rm -rf design_top/build/scripts/hd_visual/
	rm -rf design_top/build/scripts/.Xil/
	rm -rf design_top/build/scripts/*.jou
	rm -rf design_top/build/scripts/*.vivado.log
	rm -rf design_top/build/scripts/*.txt
	rm -rf design_top/verif/sim/
	rm -rf design_top/software/runtime/design_top
	rm -rf ./*~
	rm -rf ./*.key
	rm -rf ./core*
	rm -rf ./Catapult*
	rm -rf ./catapult*
	rm -rf ./*.log
	rm -rf ./design_checker_*.tcl
	rm -rf ./DVE*
	rm -rf ./verdi*
	rm -rf ./slec*
	rm -rf ./novas*
	rm -rf ./*.fsdb
	rm -rf ./*.saif*
	rm -rf ./*.vpd

# Create submission archive
submission:
	zip -r $(SUBMISSION_NAME).zip \
		src/$(DESIGN_NAME)/GBCore/GBCore.h \
		src/$(DESIGN_NAME)/NMP/NMP.h \
		design_top/design/concat_$(DESIGN_NAME).v \
		$(REPORT_HOME) \
