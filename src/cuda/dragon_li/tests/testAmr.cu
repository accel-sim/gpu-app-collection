#include <string>

#include <hydrazine/interface/ArgumentParser.h>
#include <hydrazine/interface/debug.h>

#include <dragon_li/amr/types.h>
#include <dragon_li/amr/settings.h>
#include <dragon_li/amr/amrReg.h>
#include <dragon_li/amr/amrCdp.h>
#include <dragon_li/amr/amrCpu.h>

#undef REPORT_BASE
#define REPORT_BASE 0


int main(int argc, char **argv) {
	
	//Basic Types and Settings
	typedef dragon_li::util::Types<
							int 			//SizeType
							> _Types;
	typedef dragon_li::util::Settings< 
				_Types,						//types
				256, 						//THREADS
				104,						//CTAS
				5,							//CDP_THREADS_BITS
				32							//CDP_THRESHOLD
				> _Settings;



	typedef dragon_li::amr::Types<
							_Types, 		//Basic Types
							float			//DataType
							> Types;
	typedef dragon_li::amr::Settings<
				_Settings, 					//Basic Settings
				Types,						//AMR Types
				32,							//GRID_REFINE_SIZE
				4,							//GRID_REFINE_X
				4							//GRID_REFINE_Y
				> Settings;

	hydrazine::ArgumentParser parser(argc, argv);
	parser.description("Dragon Li AMR");

	bool verbose;
	parser.parse("-v", "--v1", verbose, false, "Verbose, display information");

	bool veryVerbose;
	parser.parse("", "--v2", veryVerbose, false, "Very verbose, display extended information");

	bool verify;
	parser.parse("-e", "--verify", verify, false, "Verify results against CPU implementation");

	bool cdp; //use CDP
	parser.parse("", "--cdp", cdp, false, "Use Cuda Dynamic Parallelism");

	Settings::SizeType maxGridDataSize;
	parser.parse("-s", "--maxGridDataSize", maxGridDataSize, 4*1024*1024, "Max Grid Size (cell count)");

	Settings::SizeType maxRefineLevel;
	parser.parse("-r", "--maxRefineLevel", maxRefineLevel, 4, "Max level to refine the grid"); 

	Settings::DataType maxGridValue;
	parser.parse("-g", "--maxGridValue", maxGridValue, 6000.0, "Max grid value");

	Settings::DataType gridRefineThreshold;
	parser.parse("-t", "--gridRefineThreshold", gridRefineThreshold, 100.0, "Refine threshold for grid");

	parser.parse();


	if(!cdp) {
		dragon_li::amr::AmrReg< Settings > amrReg;
		dragon_li::amr::AmrReg< Settings >::UserConfig amrRegConfig(
														verbose,
														veryVerbose,
														maxGridDataSize,
														maxRefineLevel,
														maxGridValue,
														gridRefineThreshold
														);
	
		if(amrReg.setup(amrRegConfig))
			return -1;
	
		if(amrReg.refine())
			return -1;
	
		if(verify) {
			dragon_li::amr::AmrCpu<Settings>::amrCpu(amrReg.getStartGridValue(), gridRefineThreshold);
			if(!amrReg.verifyResult(dragon_li::amr::AmrCpu<Settings>::cpuAmrData)) {
				std::cout << "Verify correct!\n";
			}
			else {
				std::cout << "Incorrect!\n";
			}
		}
	
		if(amrReg.displayResult())
			return -1;
	}
	else {
#ifdef ENABLE_CDP
		dragon_li::amr::AmrCdp< Settings > amrCdp;
		dragon_li::amr::AmrCdp< Settings >::UserConfig amrCdpConfig(
														verbose,
														veryVerbose,
														maxGridDataSize,
														maxRefineLevel,
														maxGridValue,
														gridRefineThreshold
														);
	

		if(amrCdp.setup(amrCdpConfig))
			return -1;
	
		if(amrCdp.refine())
			return -1;
	
		if(verify) {
			dragon_li::amr::AmrCpu<Settings>::amrCpu(amrCdp.getStartGridValue(), gridRefineThreshold);
			if(!amrCdp.verifyResult(dragon_li::amr::AmrCpu<Settings>::cpuAmrData)) {
				std::cout << "Verify correct!\n";
			}
			else {
				std::cout << "Incorrect!\n";
			}
		}
	
		if(amrCdp.displayResult())
			return -1;
#else
        std::cout << "CDP is not supported! Is CDP enabled in scons?\n";
#endif
	}

	return 0;
}
