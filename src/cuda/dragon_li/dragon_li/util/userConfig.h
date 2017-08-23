#pragma once

namespace dragon_li {
namespace util {

class UserConfig {

public:
	bool verbose;
	bool veryVerbose;
	
	UserConfig(
		bool _verbose,
		bool _veryVerbose) :
			verbose(_verbose),
			veryVerbose(_veryVerbose) {}
};

}
}
