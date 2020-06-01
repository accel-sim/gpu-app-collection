echo "Running all benchnarks..."

if [ ! -n "$POWER_BIN_DIR" ]; then
	echo "Please set env var "POWER_BIN_DIR" to directory containing binaries for the power benchmarks"
	exit 1
fi

cd add_mem_1
rm -f powerdatafile
$POWER_BIN_DIR/add_mem_1
echo "finished add_mem_1..."
cd ..

cd add_mem_2
rm -f powerdatafile
$POWER_BIN_DIR/add_mem_2
echo "finished add_mem_2..."
cd ..

cd BE_L1D_HIT
rm -f powerdatafile
$POWER_BIN_DIR/BE_L1D_HIT
echo "finished BE_L1D_HIT..."
cd ..

cd BE_L1D_MISS
rm -f powerdatafile
$POWER_BIN_DIR/BE_L1D_MISS
echo "finished BE_L1D_MISS..."
cd ..

cd BE_MEM_ACCESS
rm -f powerdatafile
$POWER_BIN_DIR/BE_MEM_ACCESS
echo "finished BE_MEM_ACCESS..."
cd ..

cd BE_MEM_CNST_Acss
rm -f powerdatafile
$POWER_BIN_DIR/BE_MEM_CNST_Acss
echo "finished BE_MEM_CNST_Acss..."
cd ..

cd BE_MEM_SHRD_Acss
rm -f powerdatafile
$POWER_BIN_DIR/BE_MEM_SHRD_Acss
echo "finished BE_MEM_SHRD_Acss..."
cd ..

cd BE_SFU
rm -f powerdatafile
$POWER_BIN_DIR/BE_SFU
echo "finished BE_SFU..."
cd ..

cd BE_SFU_EXP
rm -f powerdatafile
$POWER_BIN_DIR/BE_SFU_EXP
echo "finished BE_SFU_EXP..."
cd ..

cd BE_SFU_LG2
rm -f powerdatafile
$POWER_BIN_DIR/BE_SFU_LG2
echo "finished BE_SFU_LG2..."
cd ..

cd BE_SFU_SIN
rm -f powerdatafile
$POWER_BIN_DIR/BE_SFU_SIN
echo "finished BE_SFU_SIN..."
cd ..

cd BE_SFU_SQRT
rm -f powerdatafile
$POWER_BIN_DIR/BE_SFU_SQRT
echo "finished BE_SFU_SQRT..."
cd ..

cd BE_SP_FP
rm -f powerdatafile
$POWER_BIN_DIR/BE_SP_FP
echo "finished BE_SP_FP..."
cd ..

cd BE_SP_FP_ADD
rm -f powerdatafile
$POWER_BIN_DIR/BE_SP_FP_ADD
echo "finished BE_SP_FP_ADD..."
cd ..

cd BE_SP_FP_DIV
rm -f powerdatafile
$POWER_BIN_DIR/BE_SP_FP_DIV
echo "finished BE_SP_FP_DIV..."
cd ..

cd BE_SP_FP_MUL
rm -f powerdatafile
$POWER_BIN_DIR/BE_SP_FP_DIV
echo "finished BE_SP_FP_DIV..."
cd ..

cd BE_SP_INT
rm -f powerdatafile
$POWER_BIN_DIR/BE_SP_INT
echo "finished BE_SP_INT..."
cd ..

cd BE_SP_INT_ADD
rm -f powerdatafile
$POWER_BIN_DIR/BE_SP_INT_ADD
echo "finished BE_SP_INT_ADD..."
cd ..

cd BE_SP_INT_DIV
rm -f powerdatafile
$POWER_BIN_DIR/BE_SP_INT_DIV
echo "finished BE_SP_INT_DIV..."
cd ..

cd BE_SP_INT_FP
rm -f powerdatafile
$POWER_BIN_DIR/BE_SP_INT_FP
echo "finished BE_SP_INT_FP..."
cd ..

cd BE_TEXTURE_ACCESS
rm -f powerdatafile
$POWER_BIN_DIR/BE_TEXTURE_ACCESS
echo "finished BE_TEXTURE_ACCESS..."
cd ..

cd FE_INST_CASH_HIT
rm -f powerdatafile
$POWER_BIN_DIR/FE_INST_CASH_HIT
echo "finished FE_INST_CASH_HIT..."
cd ..

cd FE_INST_CASH_HIT_MISS
rm -f powerdatafile
$POWER_BIN_DIR/FE_INST_CASH_HIT_MISS
echo "finished FE_INST_CASH_HIT_MISS..."
cd ..

cd FE_INST_CASH_MISS
rm -f powerdatafile
$POWER_BIN_DIR/FE_INST_CASH_MISS
echo "finished FE_INST_CASH_MISS..."
cd ..

cd mul_add_tex_1
rm -f powerdatafile
$POWER_BIN_DIR/mul_add_tex_1
echo "finished mul_add_tex_1..."
cd ..

cd mul_add_tex_2
rm -f powerdatafile
$POWER_BIN_DIR/mul_add_tex_2
echo "finished mul_add_tex_2..."
cd ..

cd mul_mem_1
rm -f powerdatafile
$POWER_BIN_DIR/mul_mem_1
echo "finished mul_mem_1..."
cd ..

cd mul_mem_2
rm -f powerdatafile
$POWER_BIN_DIR/mul_mem_2
echo "finished mul_mem_2..."
cd ..

cd PP_FP_MEM
rm -f powerdatafile
$POWER_BIN_DIR/PP_FP_MEM
echo "finished PP_FP_MEM..."
cd ..

echo "Finished running all benchmarks..."