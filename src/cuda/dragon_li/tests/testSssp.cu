#include <string>
#include <climits>

#include <hydrazine/interface/ArgumentParser.h>
#include <hydrazine/interface/debug.h>

#include <dragon_li/util/graphCsr.h>
#include <dragon_li/util/graphCsrDevice.h>

#include <dragon_li/sssp/types.h>
#include <dragon_li/sssp/settings.h>
#include <dragon_li/sssp/ssspReg.h>
#include <dragon_li/sssp/ssspCpu.h>
#include <dragon_li/sssp/ssspCdp.h>

#undef REPORT_BASE
#define REPORT_BASE 0

int main(int argc, char **argv) {
	
	hydrazine::ArgumentParser parser(argc, argv);
	parser.description("Dragon Li SSSP");

	std::string inputGraphFile;
	parser.parse("-g", "--graph", inputGraphFile, "graphs/sample.gr", "Input graph"); 

	std::string graphFormat;
	parser.parse("-f", "--format", graphFormat, "gr", "Input graph format, default to 'gr'");

	bool displayGraph;
	parser.parse("", "--display", displayGraph, false, "Display input graph");

	bool verbose;
	parser.parse("-v", "--v1", verbose, false, "Verbose, display information");

	bool veryVerbose;
	parser.parse("", "--v2", veryVerbose, false, "Very verbose, display extended information");

	double frontierScaleFactor;
	parser.parse("", "--sf", frontierScaleFactor, 1.0, "Frontier scale factor, default 1.0");

	int startVertexId;
	parser.parse("-s", "--source", startVertexId, 0, "The source vertex ID, default 0");

	bool verify;
	parser.parse("-e", "--verify", verify, false, "Verify results against CPU implementation");

	bool cdp; //use CDP
	parser.parse("", "--cdp", cdp, false, "Use Cuda Dynamic Parallelism");


	parser.parse();

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



	typedef dragon_li::sssp::Types<
							_Types, 		//Basic Types
							int,			//VertexIdType
							int				//EdgeWeightType
							> Types;
	typedef dragon_li::sssp::Settings<
				_Settings, 					//Basic Settings
				Types,						//SSSP Types
				INT_MAX						//Max weight
				> Settings;

	dragon_li::util::GraphCsr< Types > graph;

	if(graph.buildFromFile(inputGraphFile, graphFormat))
		return -1;

	if(displayGraph) {
		if(graph.displayCsr(veryVerbose))
			return -1;
	}

	dragon_li::util::GraphCsrDevice< Types > graphDev;
	if(graphDev.setup(graph))
		return -1;
	
	if(!cdp) {
		dragon_li::sssp::SsspReg< Settings > ssspReg;
		dragon_li::sssp::SsspReg< Settings >::UserConfig ssspRegConfig(
														verbose,
														veryVerbose,
														frontierScaleFactor,
														startVertexId);
	
		if(ssspReg.setup(graphDev, ssspRegConfig))
			return -1;
	
		if(ssspReg.search())
			return -1;
	
		if(verify) {
			dragon_li::sssp::SsspCpu<Settings>::ssspCpu(graph, startVertexId, INT_MAX);
			if(!ssspReg.verifyResult(dragon_li::sssp::SsspCpu<Settings>::cpuSearchDistance)) {
				std::cout << "Verify correct!\n";
			}
			else {
				std::cout << "Incorrect!\n";
			}
		}
	
		if(ssspReg.displayResult())
			return -1;
	}

	else {
#ifdef ENABLE_CDP
		dragon_li::sssp::SsspCdp< Settings > ssspCdp;
		dragon_li::sssp::SsspCdp< Settings >::UserConfig ssspCdpConfig(
														verbose,
														veryVerbose,
														frontierScaleFactor,
														startVertexId);
	
		if(ssspCdp.setup(graphDev, ssspCdpConfig))
			return -1;
	
		if(ssspCdp.search())
			return -1;
	
		if(verify) {
			dragon_li::sssp::SsspCpu<Settings>::ssspCpu(graph, startVertexId, INT_MAX);
			if(!ssspCdp.verifyResult(dragon_li::sssp::SsspCpu<Settings>::cpuSearchDistance)) {
				std::cout << "CDP Verify correct!\n";
			}
			else {
				std::cout << "CDP Incorrect!\n";
			}
		}
	
		if(ssspCdp.displayResult())
			return -1;
#else
        std::cout << "CDP is not supported! Is CDP enabled in scons?\n";
#endif
    }

	return 0;
}
