
BASE_DIR := $(shell pwd)
BIN_DIR := $(BASE_DIR)/bin

all: 	
	mkdir -p $(BIN_DIR)
	cd l1_bw_32f &&		make &&	cp l1_bw_32f $(BIN_DIR)
	cd l1_bw_64f && 		make &&	cp l1_bw_64f $(BIN_DIR)
	cd l1_bw_128 && 		make &&	cp l1_bw_128 $(BIN_DIR)
	cd l1_lat && 		make &&	cp l1_lat $(BIN_DIR)
	cd l2_bw_32f &&  		make &&	cp l2_bw_32f $(BIN_DIR)
	cd l2_bw_64f && 		make &&	cp l2_bw_64f $(BIN_DIR)
	cd l2_bw_128 && 		make &&	cp l2_bw_128 $(BIN_DIR)
	cd l2_lat && 		make &&	cp l2_lat $(BIN_DIR)
	cd mem_bw && 		make &&	cp mem_bw $(BIN_DIR)
	cd mem_lat && 		make &&	cp mem_lat $(BIN_DIR)
	cd shared_bw && 		make &&	cp shared_bw $(BIN_DIR)
	cd shared_lat && 		make &&	cp shared_lat $(BIN_DIR)
	cd MaxFlops && 		make &&	cp MaxFlops $(BIN_DIR)
	cd l1_shared_bw && 		make &&	cp l1_shared_bw $(BIN_DIR)
	cd shared_bank_conflicts && 		make &&	cp shared_bank_conflicts $(BIN_DIR)
	cd l1_bw_32f_unroll && 		make &&	cp l1_bw_32f_unroll $(BIN_DIR)
	cd l1_bw_32f_unroll_large && 		make &&	cp l1_bw_32f_unroll_large $(BIN_DIR)
	cd Atomic_ubench/Atomic_add/Atomic_add_bw &&            make && cp atomic_add_bw $(BIN_DIR)
	cd Atomic_ubench/Atomic_add/Atomic_add_bw_conflict &&            make && cp atomic_add_bw_conflict $(BIN_DIR)
	cd Atomic_ubench/Atomic_add/Atomic_add_bw_profile &&            make && cp atomic_add_bw_profile $(BIN_DIR)
	cd Atomic_ubench/Atomic_add/Atomic_add_bw_diverge &&            make && cp atomic_add_bw_diverge $(BIN_DIR)
	cd Atomic_ubench/Atomic_add/Atomic_add_lat &&            make && cp atomic_add_lat $(BIN_DIR)


clean:
	cd $(BIN_DIR) && rm -f *
